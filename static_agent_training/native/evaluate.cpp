// One independent lifetime per CPU lane.
#include <math.h>

#define DEVICE inline
#ifdef LIGHTS_OUT
#ifndef PUZZLE_STATES
#define PUZZLE_STATES 512
#endif
#define WORLD_STATES PUZZLE_STATES
#define WORLD_INDEX(x) (x)
#else
#define WORLD_STATES 41
#define WORLD_INDEX(x) ((x)+20)
#endif
#define WORLD_OUTPUTS (9+WORLD_STATES)

#ifndef MAX_Q
#define MAX_Q 256
#endif
#define PARAMS int B, int Q, int S, int E, double* embedding, double* output, const int* successor, const int* sizes, const int* obs_rows, const int* goal_rows, const int* expect_rows, const int* output_values, const int* target_columns, const int* starts, const int* goals, unsigned int* embed_draws, unsigned int* output_draws, int learning, int clamp_boundaries, double prediction_rate, double embedding_rate, double threshold, double* metrics, unsigned char* revised
#define ARGS B,Q,S,E,embedding,output,successor,sizes,obs_rows,goal_rows,expect_rows,output_values,target_columns,starts,goals,embed_draws,output_draws,learning,clamp_boundaries,prediction_rate,embedding_rate,threshold,metrics,revised

// Python's random.Random uses MT19937 and combines 27 + 26 random bits.
// Initial states come directly from getstate(), preserving reference streams.
DEVICE unsigned int random_word(unsigned int* state, int stride) {
    unsigned int index=state[624*stride];
    if(index>=624) {
        for(int i=0;i<624;++i) {
            const unsigned int value=(state[i*stride]&0x80000000U)|(state[((i+1)%624)*stride]&0x7fffffffU);
            state[i*stride]=state[((i+397)%624)*stride]^(value>>1)^((value&1U)?0x9908b0dfU:0U);
        }
        index=0;
    }
    unsigned int value=state[index*stride];
    state[624*stride]=index+1;
    value^=value>>11;
    value^=(value<<7)&0x9d2c5680U;
    value^=(value<<15)&0xefc60000U;
    value^=value>>18;
    return value;
}
DEVICE double random_uniform(unsigned int* state,int stride) {
    const unsigned int high=random_word(state,stride)>>5;
    const unsigned int low=random_word(state,stride)>>6;
    return (high*67108864.0+low)*(1.0/9007199254740992.0);
}

DEVICE int sample(const double* logits, int count, double uniform) {
    double largest = logits[0], total = 0, weights[MAX_Q > WORLD_OUTPUTS ? MAX_Q : WORLD_OUTPUTS];
    for (int j=1; j<count; ++j) if (logits[j]>largest) largest=logits[j];
    for (int j=0; j<count; ++j) {weights[j]=exp(logits[j]-largest);total+=weights[j];}
    double cumulative=0;
    for (int j=0; j<count; ++j) {cumulative+=weights[j]/total;if(uniform<cumulative)return j;}
    return count-1;
}

DEVICE void evaluate_one(int lane, PARAMS) {
    double* emb=embedding+(long long)lane*S*Q;
    double* out=output+(long long)lane*Q*WORLD_OUTPUTS;
    const int* edges=successor+lane*Q;
    unsigned char* rev=revised+lane*Q;
    const int count=sizes[lane];
    const int rng_stride=1;
    unsigned int* erng=embed_draws+lane*625;
    unsigned int* orng=output_draws+lane*625;

    double logits[MAX_Q], probabilities[WORLD_STATES];
    int pending_sources[16], pending_predictions[16];
    double pending_losses[16][WORLD_STATES];
    for (int episode=0; episode<E; ++episode) {
        double* stats=metrics+((long long)lane*E+episode)*11;
        int position=starts[episode], goal=goals[episode], actions=0, internal=0, pending=0, pair=-1;
        bool observe=true;
        const int context=goal_rows[WORLD_INDEX(goal)];
        while (true) {
            if(observe) {
                if(count==0){stats[1]=1;break;}
                const int signal=obs_rows[WORLD_INDEX(position)];
                for(int j=0;j<count;++j) logits[j]=emb[signal*Q+j]+(context<0?0:emb[context*Q+j]);
                pair=sample(logits,count,random_uniform(erng,rng_stride));
                observe=false;
            }
            pair=edges[pair];
            if(pair<0){stats[1]=1;stats[8]+=pending;break;}
            // Delayed compatibility check, at the same encounter as the reference.
            if(learning && rev[pair] && edges[pair]>=0) {
                double* row=out+pair*WORLD_OUTPUTS;
                int best=0;
                for(int j=1;j<WORLD_STATES;++j) if(row[9+j]>row[9+best]) best=j;
                const int signal=expect_rows[best], target=edges[pair];
                double largest=-1.7976931348623157e308, total=0;
                for(int j=0;j<count;++j){logits[j]=emb[signal*Q+j]+(context<0?0:emb[context*Q+j]);if(logits[j]>largest)largest=logits[j];}
                for(int j=0;j<count;++j) total+=exp(logits[j]-largest);
                if(exp(logits[target]-largest)/total<threshold){emb[signal*Q+target]+=embedding_rate;stats[10]+=1;}
            }
            double* row=out+pair*WORLD_OUTPUTS;
            int token=sample(row,WORLD_OUTPUTS,random_uniform(orng,rng_stride));
            stats[2]+=1;
            ++internal;
            if(token>=9) {
                pending_sources[pending]=pair;
                pending_predictions[pending]=output_values[token];
                double largest=row[9], total=0;
                for(int j=1;j<WORLD_STATES;++j) if(row[9+j]>largest) largest=row[9+j];
                for(int j=0;j<WORLD_STATES;++j) total+=exp(row[9+j]-largest);
                const double normalizer=log(total);
                for(int j=0;j<WORLD_STATES;++j) pending_losses[pending][j]=largest-row[9+j]+normalizer;
                ++pending;stats[4]+=1;
                if(internal>=16){stats[1]=4;stats[8]+=pending;break;}
                continue;
            }
            ++actions;stats[3]+=1;
#ifdef LIGHTS_OUT
            position^=output_values[token];
            internal=0;
#else
            position+=output_values[token];
            internal=0;
            if(clamp_boundaries){if(position<-20)position=-20;if(position>20)position=20;}
            if(position<-20 || position>20){stats[1]=2;stats[8]+=pending;break;}
#endif
            const int target=target_columns[WORLD_INDEX(position)];
            for(int k=0;k<pending;++k) {
                stats[5]+=1;
                if(pending_predictions[k]==position) stats[6]+=1;
                stats[7]+=pending_losses[k][target];
                if(learning) {
                    double* source=out+pending_sources[k]*WORLD_OUTPUTS;
                    double largest=source[9], total=0;
                    for(int j=1;j<WORLD_STATES;++j) if(source[9+j]>largest) largest=source[9+j];
                    for(int j=0;j<WORLD_STATES;++j){probabilities[j]=exp(source[9+j]-largest);total+=probabilities[j];}
                    for(int j=0;j<WORLD_STATES;++j) source[9+j]-=prediction_rate*(probabilities[j]/total-(j==target?1:0));
                    rev[pending_sources[k]]=1;stats[9]+=1;
                }
            }
            pending=0;
            if(position==goal){stats[0]=1;stats[1]=0;break;}
            if(actions>=12){stats[1]=3;break;}
            observe=true;
        }
    }
}

extern "C" void evaluate_cpu(PARAMS, int workers) {
    #pragma omp parallel for schedule(static) num_threads(workers)
    for(int lane=0;lane<B;++lane)evaluate_one(lane,ARGS);
}
