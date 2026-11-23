#include "evaluate_senses.cpp"
DEVICE void agent_y_one(int lane,SPARAMS,double* affinity_data,double* split_data,double* baselines,const double* router_rates,int* group_counts){
    double* b=bias+(long long)lane*Q;
    double* w=weights+(long long)lane*(R+1)*Q;
    double* g=gates+(long long)lane*Q*10;
    double* p=prediction+(long long)lane*Q*R;
    const int* edges=successor+lane*Q;
    unsigned char* rev=revised+lane*Q;
    const int count=sizes[lane],stride=1;
    unsigned int* erng=embed_draws+lane*625;
    unsigned int* orng=output_draws+lane*625;
    // At most one full-width expectation per region per synchronous round.
    const size_t capacity=(size_t)16*R;
    std::vector<int> local_regions(R),chosen_groups(R),decision_counts(R),route_features(R+2);
    std::vector<double> route_probs((size_t)R*(R+1)),scores(R+1);
    double* affinity=affinity_data+(size_t)lane*R*R;
    double* splits=split_data+(size_t)lane*R*(R+2);
    double& baseline=baselines[lane];
    const double router_rate=router_rates[lane];
    int feature_count=0;
    region_map=local_regions.data();
    std::vector<int> pairs(R),sources(capacity);
    std::vector<unsigned char> bits(capacity*R);
    std::vector<double> emitted_loss(capacity);
    double logits[MAX_Q];int active[MAX_R+1];
    for(int e=0;e<E;++e){
        double* stats=metrics+((long long)lane*E+e)*12;
        int position=starts[e],goal=goals[e],actions=0,internal=0,observations=0;
        size_t pending=0;
        bool observe=true;
        while(true){
            if(observe){
                if(count==0){stats[1]=1;break;}
                const unsigned char* frame=frames+(
#ifdef LIGHTS_OUT
                    position
#else
                    (position+20)*41+goal+20
#endif
                )*(R+1);
                feature_count=0;
                for(int i=0;i<=R;++i)if(frame[i])route_features[feature_count++]=i;
                route_features[feature_count++]=R+1;
                std::vector<std::vector<int>> groups;
                for(int receptor=0;receptor<R;++receptor){
                    int n=groups.size();
                    for(int k=0;k<n;++k){
                        double sum=0,c=0;
                        for(int j:groups[k])sadd(affinity[(size_t)receptor*R+j],sum,c);
                        scores[k]=(sum+c)/groups[k].size();
                    }
                    double sum=0,c=0;
                    for(int i=0;i<feature_count;++i)sadd(splits[(size_t)receptor*(R+2)+route_features[i]],sum,c);
                    scores[n]=sum+c;
                    double largest=scores[0];for(int k=1;k<=n;++k)largest=std::max(largest,scores[k]);
                    sum=0;c=0;double* probs=route_probs.data()+(size_t)receptor*(R+1);
                    for(int k=0;k<=n;++k){probs[k]=exp(scores[k]-largest);sadd(probs[k],sum,c);}
                    double total=sum+c;for(int k=0;k<=n;++k)probs[k]/=total;
                    double draw=random_uniform(erng,1),cumulative=0;int selected=n;
                    for(int k=0;k<=n;++k){cumulative+=probs[k];if(draw<cumulative){selected=k;break;}}
                    decision_counts[receptor]=n;chosen_groups[receptor]=selected;
                    if(selected==n)groups.push_back({});
                    groups[selected].push_back(receptor);local_regions[receptor]=selected;
                }
                sensory_partitions=groups.size();
                group_counts[((size_t)lane*E+e)*12+observations++]=sensory_partitions;
                for(int region=0;region<sensory_partitions;++region){
                    int n=0;if(frame[0])active[n++]=0;
                    for(int i=0;i<R;++i)if(region_map[i]==region&&frame[i+1])active[n++]=i+1;
                    sensory_logits(logits,b,w,active,n,count,Q);
                    pairs[region]=embedding_argmax(logits,count,erng);
                }
                observe=false;
            }
            int votes[9]={0},proposals=0,surviving=0;
            ++internal;
            for(int region=0;region<sensory_partitions;++region){
                int pair=pairs[region];if(pair<0)continue;
                pair=edges[pair];pairs[region]=pair;if(pair<0)continue;
                ++surviving;
                if(learning&&rev[pair]&&edges[pair]>=0){
                    // Full modal prediction, masked without renumbering features.
                    int n=1;active[0]=0;
                    for(int i=0;i<R;++i)if(region_map[i]==region&&p[pair*R+i]>=0)active[n++]=i+1;
                    sensory_logits(logits,b,w,active,n,count,Q);
                    double largest=logits[0],total=0;
                    for(int j=1;j<count;++j)if(logits[j]>largest)largest=logits[j];
                    for(int j=0;j<count;++j)total+=exp(logits[j]-largest);
                    int target=edges[pair];
                    if(exp(logits[target]-largest)/total<threshold){
                        double delta=embedding_rate/(1+n);b[target]+=delta;
                        for(int i=0;i<n;++i)w[active[i]*Q+target]+=delta;
                        stats[10]+=1;
                    }
                }
                int token=sample(g+pair*10,10,random_uniform(orng,stride));stats[2]+=1;
                if(token==9){
                    sources[pending]=pair;
                    for(int i=0;i<R;++i)bits[pending*R+i]=random_uniform(orng,stride)<ssigmoid(p[pair*R+i]);
                    ++pending;stats[4]+=1;
                }else{++votes[token];++proposals;}
            }
            if(!proposals){
                if(!surviving){stats[1]=1;stats[8]+=pending;break;}
                if(internal>=16){stats[1]=4;stats[8]+=pending;break;}
                continue;
            }
            // Resolve only after every emission, including full predictions.
            int chosen=-1,largest=0,ties=0,distinct=0;
            for(int token=0;token<9;++token){
                if(votes[token])++distinct;
                if(votes[token]>largest){largest=votes[token];chosen=token;ties=1;}
                else if(votes[token]==largest&&votes[token])++ties;
            }
            if((action_policy==0&&distinct>1)||(action_policy==1&&ties>1))chosen=-1;
            ++actions;stats[3]+=1;internal=0;
            if(chosen>=0){
#ifdef LIGHTS_OUT
                position^=moves[chosen];
#else
                position+=moves[chosen];
                if(clamp_boundaries){if(position<-20)position=-20;if(position>20)position=20;}
                if(position<-20||position>20){stats[1]=2;stats[8]+=pending;break;}
#endif
            }
            const unsigned char* observed=frames+(
#ifdef LIGHTS_OUT
                position
#else
                (position+20)*41+goal+20
#endif
            )*(R+1)+1;
            // Capture emission losses before sequential updates to repeated source rows.
            for(size_t k=0;k<pending;++k){
                double loss=0,c=0;bool correct=true;
                for(int i=0;i<R;++i){sadd(sloss(p[sources[k]*R+i],observed[i]),loss,c);if(bits[k*R+i]!=observed[i])correct=false;}
                emitted_loss[k]=loss+c;stats[5]+=1;if(correct)stats[6]+=1;
            }
            for(size_t k=0;k<pending;++k){
                stats[7]+=emitted_loss[k];if(learning){
                    double loss=0,c=0;double* row=p+sources[k]*R;
                    for(int i=0;i<R;++i){sadd(sloss(row[i],observed[i]),loss,c);row[i]-=prediction_rate*(ssigmoid(row[i])-observed[i]);}
                    stats[11]+=loss+c;rev[sources[k]]=1;stats[9]+=1;
                }
            }

            if(learning&&pending){
                double sum=0,c=0;
                for(size_t k=0;k<pending;++k)sadd(emitted_loss[k]/R,sum,c);
                double loss=(sum+c)/pending;
                double previous=std::isnan(baseline)?loss:baseline;
                double advantage=previous-loss;
                for(int receptor=0;receptor<R;++receptor){
                    int n=decision_counts[receptor];
                    const double* probs=route_probs.data()+(size_t)receptor*(R+1);
                    for(int k=0;k<=n;++k){
                        double delta=router_rate*advantage*((chosen_groups[receptor]==k?1.:0.)-probs[k]);
                        if(k==n){
                            for(int i=0;i<feature_count;++i)splits[(size_t)receptor*(R+2)+route_features[i]]+=delta;
                        }else{
                            int members=0;for(int j=0;j<receptor;++j)if(local_regions[j]==k)++members;
                            for(int j=0;j<receptor;++j)if(local_regions[j]==k)affinity[(size_t)receptor*R+j]+=delta/members;
                        }
                    }
                }
                baseline=.95*previous+.05*loss;
            }
            pending=0;
            if(position==goal){stats[0]=1;stats[1]=0;break;}
            if(actions>=12){stats[1]=3;break;}
            observe=true;
        }
    }
}

extern "C" void evaluate_agent_y_cpu(SPARAMS,double* affinity_data,double* split_data,double* baselines,const double* router_rates,int* group_counts,int workers){
    #pragma omp parallel for schedule(static) num_threads(workers)
    for(int lane=0;lane<B;++lane)agent_y_one(lane,SARGS,affinity_data,split_data,baselines,router_rates,group_counts);
}
