#include "model_inference.h"
using namespace inference_pz_2a;

const char* inference_pz_2a::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_pz_2a::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_pz_2a::INPUTS[INPUTS_COUNT] = {"ro_well", "ro_formation", "r_well", "lambda1"};
const string inference_pz_2a::OUTPUTS[OUTPUTS_COUNT] = {"rok"};

// output attributes inference statistic:
const double inference_pz_2a::OUTPUT_MINS[OUTPUTS_COUNT] = {-2.360173952403574};
const double inference_pz_2a::OUTPUT_MAXES[OUTPUTS_COUNT] = {9.975631454308614};

const double inference_pz_2a::INPUT_MINS[INPUTS_COUNT] = {-4.605170185988091, -2.3025850929940455, 0.04, 1.0};
const double inference_pz_2a::INPUT_MAXES[INPUTS_COUNT] = {6.907755278982137, 9.210340371976184, 0.2, 5.0};

std::vector<double> inference_pz_2a::calc_predictions(double ro_well, double ro_formation, double r_well, double lambda1) {
    double normalize_interval[2] = {0, 1}; // sigmoid interval

    // inputs forward transform:
    double inputs[INPUTS_COUNT] = {log(ro_well), log(ro_formation), r_well, lambda1};
    double outputs[OUTPUTS_COUNT];

    for (int i = 0; i < INPUTS_COUNT; i++) {
        double value = inputs[i];
        double min = inference_pz_2a::INPUT_MINS[i];
        double max = inference_pz_2a::INPUT_MAXES[i];

        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{INPUTS_COUNT};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < OUTPUTS_COUNT; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_pz_2a::OUTPUT_MINS[i], inference_pz_2a::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + OUTPUTS_COUNT);
}

double inference_pz_2a::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

