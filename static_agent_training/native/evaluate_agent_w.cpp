// Reuse the seeded RNG, softmax embedding and deterministic sum from Agent Z.
#include "evaluate_agent_z.cpp"

namespace {
int w_sensor(int byte, int rotate) {
    return ((byte << rotate) | (byte >> (width - rotate))) & 255;
}

void w_distribution(const double* logits, std::array<double, action_count>& p) {
    const double peak = *std::max_element(logits, logits + action_count);
    double total = 0;
    for (int i = 0; i < action_count; ++i) {
        p[i] = std::exp(logits[i] - peak);
        total += p[i];
    }
    for (double& value : p) value /= total;
}

template<class Iterator>
int w_sample(Iterator first, Iterator last, Random& rng) {
    double total = 0;
    for (auto i = first; i != last; ++i) total += *i;
    const double draw = rng.uniform() * total;
    double running = 0;
    for (auto i = first; i != last - 1; ++i) {
        running += *i;
        if (draw < running) return static_cast<int>(i - first);
    }
    return static_cast<int>(last - first - 1);
}
}

extern "C" {
struct WDiagnostics {
    std::uint64_t action_counts[action_count];
    std::uint64_t starts[41];
    std::uint64_t solved[41];
    std::uint64_t pair_starts[41 * 41];
    std::uint64_t pair_solved[41 * 41];
    std::uint64_t ends[4]; // goal, action limit, exhausted chains, round limit
    std::uint64_t compatibility_bins[20];
    std::uint64_t unique_first;
    std::uint64_t tied_first;
    std::uint64_t action_sign[3]; // negative, zero, positive
    double compatibility_sum;
    double action_entropy_sum;
};

struct WTrace {
    int episode;
    int round;
    int region;
    int pair;
    int action;
    int feedback;
    int successor;
    double compatibility;
    double probability;
    double entropy;
};

int w_evaluate(int pairs, int regions, int episodes, int round_limit, int action_limit, int puzzle, int sensor_rotate,
               const int* successors, const int* tasks, const int* moves,
               const std::uint32_t* state, double* embedding, double* continuation,
               double action_rate, double structural_rate, int* stats, WTrace* trace,
               std::uint32_t* final_state, WDiagnostics* diagnostics,
               std::uint8_t* changed_actions, std::uint8_t* changed_embedding) {
    try {
        Random rng(state);
        std::vector<double> embedded(pairs);
        std::vector<Contributions> action_updates(pairs * action_count);
        std::vector<Contributions> structural_updates(pairs * (width + 1));
        int successes = 0, rounds = 0, actions = 0;
        for (int episode = 0; episode < episodes; ++episode) {
            int position = tasks[2 * episode];
            const int goal = tasks[2 * episode + 1];
            int remaining = action_limit;
            std::array<int, maximum_regions> active;
            const auto initial = signal_bits(w_sensor(puzzle ? position : ((goal - position) & 255), sensor_rotate));
            for (int region = 0; region < regions; ++region) {
                z_embedding(pairs, width, region, regions, embedding, initial.data(), embedded.data());
                active[region] = w_sample(embedded.begin(), embedded.end(), rng);
            }
            for (int round = 0; round < round_limit && remaining && position != goal; ++round) {
                std::array<int, maximum_regions> order;
                int count = 0;
                for (int region = 0; region < regions; ++region)
                    if (active[region] >= 0) order[count++] = region;
                if (!count) break;
                for (int i = 0; i < count && remaining && position != goal; ++i) {
                    const int region = order[(i + round % count) % count];
                    const int pair = active[region];
                    auto* logits = continuation + pair * action_count;
                    std::array<double, action_count> distribution;
                    w_distribution(logits, distribution);
                    const int action = w_sample(distribution.begin(), distribution.end(), rng);
                    position = puzzle ? position ^ moves[action] : std::clamp(position + moves[action], -20, 20);
                    --remaining;
                    const int feedback = w_sensor(puzzle ? position : ((goal - position) & 255), sensor_rotate);
                    const int successor = successors[pair];
                    active[region] = successor;
                    double compatibility = -1;
                    if (successor >= 0 && (action_rate || structural_rate || trace || diagnostics)) {
                        const auto measured = signal_bits(feedback);
                        z_embedding(pairs, width, region, regions, embedding, measured.data(), embedded.data());
                        compatibility = embedded[successor];
                        if (diagnostics) {
                            diagnostics->compatibility_sum += compatibility;
                            ++diagnostics->compatibility_bins[std::min(19, static_cast<int>(compatibility * 20))];
                            const double best = *std::max_element(embedded.begin(), embedded.end());
                            if (compatibility == best) {
                                if (std::count(embedded.begin(), embedded.end(), best) == 1)
                                    ++diagnostics->unique_first;
                                else
                                    ++diagnostics->tied_first;
                            }
                        }
                        if (action_rate) {
                            const double scale = action_rate * (compatibility - 1.0 / pairs);
                            if (diagnostics)
                                ++diagnostics->action_sign[(scale > 0) - (scale < 0) + 1];
                            for (int j = 0; j < action_count; ++j)
                                action_updates[pair * action_count + j].add(
                                    scale * ((j == action) - distribution[j]));
                        }
                        if (structural_rate) {
                            for (int row = 0; row < pairs; ++row) {
                                const double delta = structural_rate * ((row == successor) - embedded[row]);
                                const int base = row * (width + 1);
                                structural_updates[base].add(delta);
                                for (int bit = region; bit < width; bit += regions)
                                    if (measured[bit]) structural_updates[base + bit + 1].add(delta);
                            }
                        }
                    }
                    if (trace || diagnostics) {
                        double entropy = 0;
                        for (double p : distribution)
                            if (p > 0) entropy -= p * std::log(p);
                        if (diagnostics) {
                            ++diagnostics->action_counts[action];
                            diagnostics->action_entropy_sum += entropy;
                        }
                        if (trace)
                            trace[actions] = {episode, round, region, pair, action, feedback,
                                              successor, compatibility, distribution[action], entropy};
                    }
                    ++actions;
                }
                if (action_rate)
                    for (std::size_t i = 0; i < action_updates.size(); ++i)
                        if (action_updates[i].count) {
                            const double before = continuation[i];
                            continuation[i] += action_updates[i].sum();
                            if (changed_actions && before != continuation[i])
                                changed_actions[i / action_count] = 1;
                        }
                if (structural_rate)
                    for (std::size_t i = 0; i < structural_updates.size(); ++i)
                        if (structural_updates[i].count) {
                            const double before = embedding[i];
                            embedding[i] += structural_updates[i].sum();
                            if (changed_embedding && before != embedding[i])
                                changed_embedding[i / (width + 1)] = 1;
                        }
                ++rounds;
            }
            successes += position == goal;
            if (diagnostics) {
                const int start = tasks[2 * episode] + (puzzle ? 0 : 20);
                ++diagnostics->starts[start];
                diagnostics->solved[start] += position == goal;
                const int task = start * 41 + goal + (puzzle ? 0 : 20);
                ++diagnostics->pair_starts[task];
                diagnostics->pair_solved[task] += position == goal;
                const bool chain_exhausted = std::all_of(active.begin(), active.begin() + regions,
                                                         [](int pair) { return pair < 0; });
                const int reason = position == goal ? 0 : !remaining ? 1 :
                                   chain_exhausted ? 2 : 3;
                ++diagnostics->ends[reason];
            }
        }
        if (final_state) rng.save(final_state);
        stats[0] = successes;
        stats[1] = rounds;
        stats[2] = actions;
        return 0;
    } catch (...) {
        return 1;
    }
}
}
