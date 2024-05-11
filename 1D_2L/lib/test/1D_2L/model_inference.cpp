#include "model_inference.h"
using namespace inference_1d_2l;

const char* inference_1d_2l::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_1d_2l::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_1d_2l::INPUTS[INPUTS_COUNT] = {"ro_well", "ro_formation", "d_well"};
const string inference_1d_2l::OUTPUTS[OUTPUTS_COUNT] = {"A04M01N", "A10M01N", "A20M05N", "A40M05N", "A80M10N"};

// output attributes inference statistic:
const double inference_1d_2l::OUTPUT_MINS[OUTPUTS_COUNT] = {-3.1487785634443775, -2.395997275136031, -2.3656803894149374, -2.332715488563936, -2.3167114011156187};
const double inference_1d_2l::OUTPUT_MAXES[OUTPUTS_COUNT] = {9.556833658307777, 9.818430627995768, 10.010232109966116, 10.153456075492668, 10.289429994155428};

const double inference_1d_2l::INPUT_MINS[INPUTS_COUNT] = {-4.605170185988091, -2.3025850929940455, 0.08};
const double inference_1d_2l::INPUT_MAXES[INPUTS_COUNT] = {6.907755278982137, 9.210340371976184, 0.4};

std::vector<double> inference_1d_2l::calc_predictions(double ro_well, double ro_formation, double d_well) {
    double normalize_interval[2] = {0, 1}; // sigmoid interval

    // inputs forward transform:
    double inputs[INPUTS_COUNT] = {log(ro_well), log(ro_formation), d_well};
    double outputs[OUTPUTS_COUNT] = {0, 0, 0, 0, 0};

    for (int i = 0; i < INPUTS_COUNT; i++) {
        double value = inputs[i];
        double min = inference_1d_2l::INPUT_MINS[i];
        double max = inference_1d_2l::INPUT_MAXES[i];

        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{INPUTS_COUNT};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < OUTPUTS_COUNT; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_1d_2l::OUTPUT_MINS[i], inference_1d_2l::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + OUTPUTS_COUNT);
}

double inference_1d_2l::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

