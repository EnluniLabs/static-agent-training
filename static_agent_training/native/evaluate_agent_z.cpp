#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void z_probabilities(int count, const double* logits, double* output) {
    for (int i = 0; i < count; ++i) {
        const double z = std::exp(-std::abs(logits[i]));
        output[i] = logits[i] >= 0 ? 1.0 / (1.0 + z) : z / (1.0 + z);
    }
}

extern "C" void z_embedding(int count, int bits, int region, int regions,
                           const double* weights, const double* signal, double* output) {
    double peak = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < count; ++i) {
        const double* row = weights + i * (bits + 1);
        double input = 0;
        for (int bit = region; bit < bits; bit += regions) {
            if (signal[bit]) input += row[bit + 1];
        }
        output[i] = row[0] + input;
        peak = std::max(peak, output[i]);
    }
    double total = 0;
    for (int i = 0; i < count; ++i) {
        output[i] = std::exp(output[i] - peak);
        total += output[i];
    }
    for (int i = 0; i < count; ++i) output[i] /= total;
}

extern "C" void z_gradient(int count, const double* probabilities, const double* targets,
                          double rate, double* output) {
    for (int i = 0; i < count; ++i) output[i] = rate * (targets[i] - probabilities[i]);
}


namespace {
constexpr int width = 8;
constexpr int maximum_regions = 8;
constexpr int action_base = 0x70;
constexpr int action_count = 9;

class Random {
    std::array<std::uint32_t, 624> state;
    int index;

    std::uint32_t next() {
        if (index == 624) {
            for (int i = 0; i < 624; ++i) {
                const auto combined = (state[i] & 0x80000000u) |
                                      (state[(i + 1) % 624] & 0x7fffffffu);
                state[i] = state[(i + 397) % 624] ^ (combined >> 1) ^
                           ((combined & 1) ? 0x9908b0dfu : 0u);
            }
            index = 0;
        }
        auto value = state[index++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        return value ^ (value >> 18);
    }

public:
    explicit Random(const std::uint32_t* initial) : index(static_cast<int>(initial[624])) {
        std::copy_n(initial, 624, state.begin());
    }

    void save(std::uint32_t* output) const {
        std::copy(state.begin(), state.end(), output);
        output[624] = static_cast<std::uint32_t>(index);
    }

    double uniform() {
        const auto high = next() >> 5;
        const auto low = next() >> 6;
        return (high * 67108864.0 + low) / 9007199254740992.0;
    }
};

std::array<double, width> signal_bits(int signal) {
    std::array<double, width> result{};
    for (int bit = 0; bit < width; ++bit) result[bit] = (signal >> (7 - bit)) & 1;
    return result;
}

struct Emission {
    int pair;
    int signal;
    std::array<double, width> probabilities;
};

class NumberLineAdapter {
    int position;
    const int goal;
    const int* moves;
    int remaining = 12;

    static bool is_action(int signal) {
        return signal >= action_base && signal < action_base + action_count;
    }

public:
    NumberLineAdapter(int start, int goal, const int* moves)
        : position(start), goal(goal), moves(moves) {}

    int location() const { return position; }
    int actions_taken() const { return 12 - remaining; }
    bool success() const { return position == goal; }
    bool done() const { return !remaining || success(); }
    int measurement() const { return (goal - position) & 255; }

    struct Feedback {
        int observation = -1;
        std::array<bool, maximum_regions> recipients{};
    };

    Feedback interpret(const std::array<Emission, maximum_regions>& emitted, int regions) {
        Feedback feedback;
        bool has_expectation = false;
        std::array<int, action_count> actions{};
        for (int region = 0; region < regions; ++region) {
            const auto& event = emitted[region];
            if (event.pair < 0) continue;
            if (is_action(event.signal)) {
                ++actions[event.signal - action_base];
            } else {
                feedback.recipients[region] = true;
                has_expectation = true;
            }
        }
        for (int code = 0; code < action_count; ++code) {
            for (int i = 0; i < actions[code] && !done(); ++i) {
                position = std::clamp(position + moves[code], -20, 20);
                --remaining;
            }
        }
        if (has_expectation) feedback.observation = measurement();
        return feedback;
    }
};

// A parameter has at most one contribution per region. Sort by numerical
// value, sum an error-free expansion, then round once as in the reference.
struct Contributions {
    std::array<double, maximum_regions> values{};
    int count = 0;

    void add(double value) { values[count++] = value; }

    double sum() {
        std::sort(values.begin(), values.begin() + count);
        std::array<double, maximum_regions> partials{};
        int size = 0;
        for (int i = 0; i < count; ++i) {
            double current = values[i];
            int written = 0;
            for (int j = 0; j < size; ++j) {
                double part = partials[j];
                if (std::abs(current) < std::abs(part)) std::swap(current, part);
                const double high = current + part;
                const double low = part - (high - current);
                if (low) partials[written++] = low;
                current = high;
            }
            size = written;
            if (current) partials[size++] = current;
        }
        double high = size ? partials[--size] : 0;
        double low = 0;
        while (size) {
            const double current = high;
            const double part = partials[--size];
            high = current + part;
            low = part - (high - current);
            if (low) break;
        }
        if (size && ((low < 0 && partials[size - 1] < 0) ||
                     (low > 0 && partials[size - 1] > 0))) {
            const double twice = low * 2;
            const double rounded = high + twice;
            if (rounded - high == twice) high = rounded;
        }
        count = 0;
        return high;
    }
};
}

extern "C" {
struct ZTrace {
    int round;
    int region;
    int pair;
    int successor;
    int signal;
    int observation;
    int mismatches;
    int position;
    int actions;
    double compatibility;
};

int z_evaluate(int pairs, int regions, int episodes, int round_limit,
               const int* successors, const int* tasks, const int* moves,
               const std::uint32_t* state, double* embedding, double* continuation,
               double empirical_rate, double structural_rate, double threshold,
               int* stats, ZTrace* trace, const int* stream) {
    try {
        Random rng(state);
        std::vector<double> embedded(pairs);
        std::vector<Contributions> empirical(pairs * width);
        std::vector<Contributions> structural(pairs * (width + 1));
        int successes = 0, rounds = 0, traces = 0, actions = 0, tested = 0, correct = 0;
        for (int episode = 0; episode < episodes; ++episode) {
            NumberLineAdapter world(tasks[2 * episode], tasks[2 * episode + 1], moves);
            std::array<int, maximum_regions> active;
            const auto initial = signal_bits(stream ? stream[episode] : world.measurement());
            for (int region = 0; region < regions; ++region) {
                z_embedding(pairs, width, region, regions, embedding, initial.data(), embedded.data());
                for (int i = 1; i < pairs; ++i) embedded[i] += embedded[i - 1];
                const double draw = rng.uniform() * embedded.back();
                active[region] = static_cast<int>(
                    std::upper_bound(embedded.begin(), embedded.end() - 1, draw) - embedded.begin());
            }

            for (int iteration = 0; iteration < round_limit && (stream || !world.done()); ++iteration) {
                std::array<Emission, maximum_regions> emitted{};
                bool any = false;
                for (int region = 0; region < regions; ++region) {
                    const int pair = active[region];
                    auto& event = emitted[region];
                    event.pair = pair;
                    if (pair < 0) continue;
                    any = true;
                    z_probabilities(width, continuation + pair * width, event.probabilities.data());
                    for (const double probability : event.probabilities)
                        event.signal = (event.signal << 1) | (rng.uniform() < probability);
                    active[region] = successors[pair];
                }
                if (!any) break;

                const auto feedback = stream ? NumberLineAdapter::Feedback{} : world.interpret(emitted, regions);
                const auto measured = feedback.observation < 0
                    ? std::array<double, width>{} : signal_bits(feedback.observation);
                for (int region = 0; region < regions; ++region) {
                    const auto& event = emitted[region];
                    const int pair = event.pair;
                    if (pair < 0) continue;
                    const int successor = active[region];
                    ZTrace record{rounds, region, pair, successor, event.signal, -1, -1,
                                  world.location(), world.actions_taken(), -1};
                    if (feedback.recipients[region]) {
                        record.observation = feedback.observation;
                        const auto expected = signal_bits(event.signal);
                        record.mismatches = 0;
                        for (int bit = 0; bit < width; ++bit)
                            record.mismatches += expected[bit] != measured[bit];
                        ++tested;
                        correct += record.mismatches == 0;
                        if (empirical_rate) {
                            for (int bit = 0; bit < width; ++bit)
                                empirical[pair * width + bit].add(
                                    empirical_rate * (measured[bit] - event.probabilities[bit]));
                        }
                        if (successor >= 0 && (structural_rate || trace)) {
                            z_embedding(pairs, width, region, regions, embedding, expected.data(), embedded.data());
                            record.compatibility = embedded[successor];
                            if (structural_rate && record.compatibility < threshold) {
                                for (int row = 0; row < pairs; ++row) {
                                    const double delta = structural_rate * ((row == successor) - embedded[row]);
                                    const int base = row * (width + 1);
                                    structural[base].add(delta);
                                    for (int bit = region; bit < width; bit += regions)
                                        if (expected[bit]) structural[base + bit + 1].add(delta);
                                }
                            }
                        }
                    }
                    if (trace) trace[traces++] = record;
                }

                for (std::size_t i = 0; i < empirical.size(); ++i)
                    if (empirical[i].count) continuation[i] += empirical[i].sum();
                for (std::size_t i = 0; i < structural.size(); ++i)
                    if (structural[i].count) embedding[i] += structural[i].sum();
                ++rounds;
            }
            successes += !stream && world.success();
            actions += world.actions_taken();
        }
        stats[0] = successes;
        stats[1] = rounds;
        stats[2] = traces;
        stats[3] = actions;
        stats[4] = tested;
        stats[5] = correct;
        return 0;
    } catch (...) {
        return 1;
    }
}
}
