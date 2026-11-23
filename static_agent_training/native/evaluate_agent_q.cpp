#include "evaluate_agent_w.cpp"

namespace {
void q_embed(int pairs, const double* weights, int signal, int region, int regions,
             std::vector<double>& probabilities, std::vector<double>& logs) {
    double peak = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < pairs; ++row) {
        const double* w = weights + row * 9;
        double sum = 0;
        for (int bit = region; bit < 8; bit += regions)
            if (signal & (128 >> bit)) sum += w[bit + 1];
        logs[row] = w[0] + sum;
        peak = std::max(peak, logs[row]);
    }
    double norm = 0;
    for (double& value : logs) {
        value -= peak;
        norm += std::exp(value);
    }
    const double log_norm = std::log(norm);
    for (int row = 0; row < pairs; ++row) {
        logs[row] -= log_norm;
        probabilities[row] = std::exp(logs[row]);
    }
}
}

extern "C" {
struct QTrace {
    int round, region, source, successor, action, sample, observation, position, actions;
    double empirical_error, structural_error;
};

int q_evaluate(int pairs, int regions, int episodes, int round_limit,
               const int* successors, const int* tasks, const int* moves,
               const std::uint32_t* state, double* embedding, double* expectations,
               double exploration, double empirical_rate, double structural_rate,
               int* stats, QTrace* trace) {
    try {
        Random rng(state);
        std::vector<double> probabilities(pairs), logs(pairs);
        std::vector<Contributions> empirical(pairs * action_count * 8), structural(pairs * 9);
        int successes = 0, rounds = 0, actions = 0, correct = 0;
        for (int episode = 0; episode < episodes; ++episode) {
            int position = tasks[episode * 2];
            const int goal = tasks[episode * 2 + 1];
            int remaining = 12;
            std::array<int, maximum_regions> active;
            for (int region = 0; region < regions; ++region) {
                q_embed(pairs, embedding, (goal - position) & 255, region, regions, probabilities, logs);
                active[region] = w_sample(probabilities.begin(), probabilities.end(), rng);
            }
            for (int round = 0; round < round_limit && remaining && position != goal; ++round) {
                std::array<int, maximum_regions> order;
                int count = 0;
                for (int region = 0; region < regions; ++region) {
                    if (active[region] >= 0 && successors[active[region]] >= 0)
                        order[count++] = region;
                }
                if (!count) break;
                for (int i = 0; i < count && remaining && position != goal; ++i) {
                    const int region = order[(i + round % count) % count];
                    const int source = active[region], target = successors[source];
                    int mask = 0;
                    for (int bit = region; bit < 8; bit += regions) mask |= 128 >> bit;
                    // Marginalize ignored bits analytically. Their probabilities sum to one.
                    std::array<int, 256> signals{};
                    std::array<double, 256> compatible{};
                    int combinations = 0;
                    for (int signal = 0; signal < 256; ++signal) {
                        if (signal & ~mask) continue;
                        signals[combinations] = signal;
                        q_embed(pairs, embedding, signal, region, regions, probabilities, logs);
                        compatible[combinations++] = probabilities[target];
                    }
                    std::array<double, action_count> policy{};
                    double total = 0;
                    for (int action = 0; action < action_count; ++action) {
                        const int base = (source * action_count + action) * 8;
                        std::array<double,8> ones{}, zeros{};
                        z_probabilities(8, expectations + base, ones.data());
                        for (int bit = region; bit < 8; bit += regions) {
                            const double logit = -expectations[base + bit];
                            z_probabilities(1, &logit, &zeros[bit]);
                        }
                        for (int j = 0; j < combinations; ++j) {
                            double mass = 1;
                            for (int bit = region; bit < 8; bit += regions)
                                mass *= (signals[j] & (128 >> bit)) ? ones[bit] : zeros[bit];
                            policy[action] += mass * compatible[j];
                        }
                        total += policy[action];
                    }
                    for (double& value : policy)
                        value = total > 0 ? (1 - exploration) * value / total + exploration / action_count
                                          : 1.0 / action_count;
                    const int action = w_sample(policy.begin(),policy.end(),rng);
                    const int base = (source * action_count + action) * 8;
                    std::array<double, 8> p{};
                    z_probabilities(8, expectations + base, p.data());
                    int sample = 0;
                    for (double prob : p) sample = (sample << 1) | (rng.uniform() < prob);
                    position = std::clamp(position + moves[action], -20, 20);
                    --remaining;
                    const int observed = (goal - position) & 255;
                    correct += sample == observed;
                    double loss = 0, structural_error = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const bool actual = observed & (128 >> bit);
                        const double z = expectations[base + bit];
                        if (trace) loss += std::max(actual ? -z : z, 0.) + std::log1p(std::exp(-std::abs(z)));
                        if (empirical_rate) empirical[base + bit].add(empirical_rate * (actual - p[bit]));
                    }
                    if (structural_rate || trace) {
                        q_embed(pairs, embedding, sample, region, regions, probabilities, logs);
                        structural_error = -logs[target];
                        if (structural_rate)
                            for (int row = 0; row < pairs; ++row) {
                                const double delta = structural_rate * ((row == target) - probabilities[row]);
                                structural[row * 9].add(delta);
                                for (int bit = region; bit < 8; bit += regions)
                                    if (sample & (128 >> bit)) structural[row * 9 + bit + 1].add(delta);
                            }
                    }
                    if (trace) trace[actions] = {rounds,region,source,target,action,sample,observed,
                                                position,12-remaining,loss,structural_error};
                    ++actions;
                    active[region] = target;
                }
                if (empirical_rate)
                    for (std::size_t i = 0; i < empirical.size(); ++i)
                        if (empirical[i].count) expectations[i] += empirical[i].sum();
                if (structural_rate)
                    for (std::size_t i = 0; i < structural.size(); ++i)
                        if (structural[i].count) embedding[i] += structural[i].sum();
                ++rounds;
            }
            successes += position == goal;
        }
        stats[0]=successes; stats[1]=rounds; stats[2]=actions; stats[3]=correct;
        return 0;
    } catch (...) { return 1; }
}
}
