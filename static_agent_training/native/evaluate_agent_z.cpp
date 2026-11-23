#include <algorithm>
#include <cmath>
#include <limits>

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
