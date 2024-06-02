#include "1D_2L/model_inference.h"
#include <vector>
#include <chrono>

int main(int argc, const char* argv[]) {
    vector<double> inputs{1, 2, 3};
    int iterations_count = 10000;

    const auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iterations_count; i++) {
        vector<double> predictions = inference_1d_2l::calc_predictions(inputs[0], inputs[1], inputs[2]);
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff = end - start;
    cout << "average predict time=" << diff.count() / iterations_count;
}