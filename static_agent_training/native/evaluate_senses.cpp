// Fixed sensory bank evaluator. Shares only RNG and sampling with the table model.
#include "evaluate.cpp"
#ifndef MAX_R
#define MAX_R 256
#endif
#define SPARAMS int B,int Q,int R,int E,double* bias,double* weights,double* gates,double* prediction,const int* successor,const int* sizes,const unsigned char* frames,const int* moves,const int* starts,const int* goals,unsigned int* embed_draws,unsigned int* output_draws,int learning,int clamp_boundaries,double prediction_rate,double embedding_rate,double threshold,double* metrics,unsigned char* revised
#define SARGS B,Q,R,E,bias,weights,gates,prediction,successor,sizes,frames,moves,starts,goals,embed_draws,output_draws,learning,clamp_boundaries,prediction_rate,embedding_rate,threshold,metrics,revised
DEVICE double ssigmoid(double v){return v>=0?1/(1+exp(-v)):exp(v)/(1+exp(v));}
DEVICE double sloss(double v,int bit){return (v>0?v:0)-v*bit+log1p(exp(-fabs(v)));}
// Compensated sums match the reference's fsum within the replay tolerance.
DEVICE void sadd(double v,double& sum,double& correction){double t=sum+v;correction+=fabs(sum)>=fabs(v)?(sum-t)+v:(v-t)+sum;sum=t;}
DEVICE void sensory_logits(double* logits,const double* bias,const double* weights,const int* active,int n,int count,int Q){
    for(int j=0;j<count;++j){double sum=0,c=0;for(int i=0;i<n;++i)sadd(weights[active[i]*Q+j],sum,c);logits[j]=bias[j]+(sum+c);}
}
DEVICE void sensory_one(int lane,SPARAMS){
    double* b=bias+(long long)lane*Q;
    double* w=weights+(long long)lane*(R+1)*Q;
    double* g=gates+(long long)lane*Q*10;
    double* p=prediction+(long long)lane*Q*R;
    const int* edges=successor+lane*Q;
    unsigned char* rev=revised+lane*Q;
    int count=sizes[lane];
    int stride=1;unsigned int* erng=embed_draws+lane*625;unsigned int* orng=output_draws+lane*625;

    double logits[MAX_Q];int active[MAX_R+1],sources[16];unsigned char pending_bits[16][MAX_R];
    double emitted_loss[16];
    for(int e=0;e<E;++e){
        double* stats=metrics+((long long)lane*E+e)*12;
        int position=starts[e],goal=goals[e],actions=0,internal=0,pending=0,pair=-1;bool observe=true;
        while(true){
            if(observe){
                if(count==0){stats[1]=1;break;}
                const unsigned char* frame=frames+((position+20)*41+goal+20)*(R+1);
                int n=0;for(int i=0;i<=R;++i)if(frame[i])active[n++]=i;
                sensory_logits(logits,b,w,active,n,count,Q);
                pair=sample(logits,count,random_uniform(erng,stride));observe=false;
            }
            pair=edges[pair];if(pair<0){stats[1]=1;stats[8]+=pending;break;}
            if(learning&&rev[pair]&&edges[pair]>=0){
                int n=1;active[0]=0;for(int i=0;i<R;++i)if(p[pair*R+i]>=0)active[n++]=i+1;
                sensory_logits(logits,b,w,active,n,count,Q);
                double largest=logits[0],total=0;for(int j=1;j<count;++j)if(logits[j]>largest)largest=logits[j];
                for(int j=0;j<count;++j)total+=exp(logits[j]-largest);
                int target=edges[pair];if(exp(logits[target]-largest)/total<threshold){
                    double delta=embedding_rate/(1+n);b[target]+=delta;
                    for(int i=0;i<n;++i)w[active[i]*Q+target]+=delta;stats[10]+=1;
                }
            }
            int token=sample(g+pair*10,10,random_uniform(orng,stride));stats[2]+=1;++internal;
            if(token==9){
                sources[pending]=pair;for(int i=0;i<R;++i)pending_bits[pending][i]=random_uniform(orng,stride)<ssigmoid(p[pair*R+i]);
                ++pending;stats[4]+=1;if(internal>=16){stats[1]=4;stats[8]+=pending;break;}continue;
            }
            ++actions;stats[3]+=1;position+=moves[token];internal=0;
            if(clamp_boundaries){if(position<-20)position=-20;if(position>20)position=20;}
            if(position<-20||position>20){stats[1]=2;stats[8]+=pending;break;}
            const unsigned char* observed=frames+((position+20)*41+goal+20)*(R+1)+1;
            // Capture emission losses before sequential updates to repeated source rows.
            for(int k=0;k<pending;++k){
                double loss=0,c=0;bool correct=true;
                for(int i=0;i<R;++i){sadd(sloss(p[sources[k]*R+i],observed[i]),loss,c);if(pending_bits[k][i]!=observed[i])correct=false;}
                emitted_loss[k]=loss+c;stats[5]+=1;if(correct)stats[6]+=1;
            }
            for(int k=0;k<pending;++k){
                stats[7]+=emitted_loss[k];if(learning){
                    double loss=0,c=0;double* row=p+sources[k]*R;
                    for(int i=0;i<R;++i){sadd(sloss(row[i],observed[i]),loss,c);row[i]-=prediction_rate*(ssigmoid(row[i])-observed[i]);}
                    stats[11]+=loss+c;rev[sources[k]]=1;stats[9]+=1;
                }
            }
            pending=0;if(position==goal){stats[0]=1;stats[1]=0;break;}if(actions>=12){stats[1]=3;break;}observe=true;
        }
    }
}
extern "C" void evaluate_senses_cpu(SPARAMS,int workers){
    #pragma omp parallel for schedule(static) num_threads(workers)
    for(int lane=0;lane<B;++lane)sensory_one(lane,SARGS);
}
