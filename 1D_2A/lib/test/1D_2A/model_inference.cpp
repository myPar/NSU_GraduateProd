#include "model_inference.h"
using namespace inference_1d_2a;

const char* inference_1d_2a::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_1d_2a::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_1d_2a::INPUTS[4] = {"ro_well", "ro_formation", "rad_well", "kanisotrop"};
const string inference_1d_2a::OUTPUTS[5] = {"A04M01N", "A10M01N", "A20M05N", "A40M05N", "A80M10N"};

// output attributes inference statistic:
const double inference_1d_2a::OUTPUT_MEANS[5] = {1.1696066630906665, 2.1549688040161925, 2.8295580213737046, 3.2474607202004897, 3.547436036617802};
const double inference_1d_2a::OUTPUT_STDS[5] = {1.8981222952922183, 2.255211124359088, 2.6552511938592653, 3.0011638205846554, 3.3244960293369976};

const double inference_1d_2a::OUTPUT_MINS[5] = {-2.2750642972209936, -2.0179301353187635, -1.953984183017199, -1.8572676912982564, -1.7621519117873663};
const double inference_1d_2a::OUTPUT_MAXES[5] = {2.835495775774292, 2.636742904204633, 2.4430786527659194, 2.289199537124314, 2.105355682717262};

// input attributes inference statistic:
const double inference_1d_2a::INPUT_MEANS[4] = {-1.7269386497605308, 3.4538777271899646, 0.08936170212765956, 2.183226068969074};
const double inference_1d_2a::INPUT_STDS[4] = {1.7269388050893328, 3.389313407249595, 0.03429870385983213, 1.276033218172284};

const double inference_1d_2a::INPUT_MINS[4] = {-1.6666667792427492, -1.6984156165290543, -1.4391710639966195, -0.9272690178582171};
const double inference_1d_2a::INPUT_MAXES[4] = {1.666667010837201, 1.6984155647788115, 3.225728246888976, 2.207445614202356};

std::vector<double> inference_1d_2a::calc_predictions(double ro_well, double ro_formation, double rad_well, double kanisotrop) {
    double normalize_interval[2] = {-1, 1};

    // inputs forward transform:
    const int inputs_count = 4;
    const int outputs_count = 5;
    double inputs[inputs_count] = {log(ro_well), log(ro_formation), rad_well, kanisotrop};
    double outputs[outputs_count] = {0, 0, 0, 0, 0};

    for (int i = 0; i < inputs_count; i++) {
        double value = inputs[i];
        double min = inference_1d_2a::INPUT_MINS[i];
        double max = inference_1d_2a::INPUT_MAXES[i];

        double mean = inference_1d_2a::INPUT_MEANS[i];
        double std = inference_1d_2a::INPUT_STDS[i];

        value = inputs[i] = scale(value, mean, std);
        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{inputs_count};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2], (float) inputs[3]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < outputs_count; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_1d_2a::OUTPUT_MINS[i], inference_1d_2a::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // scaling backward
        value = outputs[i] = value * inference_1d_2a::OUTPUT_STDS[i] + inference_1d_2a::OUTPUT_MEANS[i];
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + outputs_count);
}

double inference_1d_2a::scale(double attribute_value, double attribute_mean, double attribute_std) {
    return (attribute_value - attribute_mean) / attribute_std;
}

double inference_1d_2a::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

