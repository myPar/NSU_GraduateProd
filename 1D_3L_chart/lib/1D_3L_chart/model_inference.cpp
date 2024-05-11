#include "model_inference.h"
using namespace inference_1d_3l_chart;

const char* inference_1d_3l_chart::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_1d_3l_chart::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_1d_3l_chart::INPUTS[INPUTS_COUNT] = {"AO/d", "ro_formation", "invasion_zone_ro", "D/d"};
const string inference_1d_3l_chart::OUTPUTS[OUTPUTS_COUNT] = {"rok"};

// output attributes inference statistic:
const double inference_1d_3l_chart::OUTPUT_MINS[OUTPUTS_COUNT] = {-2.4100879017239056};
const double inference_1d_3l_chart::OUTPUT_MAXES[OUTPUTS_COUNT] = {10.536228993573161};

const double inference_1d_3l_chart::INPUT_MINS[INPUTS_COUNT] = {-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, 1};
const double inference_1d_3l_chart::INPUT_MAXES[INPUTS_COUNT] = {6.907755278982137, 9.210340371976184, 9.210340371976184, 51};

std::vector<double> inference_1d_3l_chart::calc_predictions(double AO_d, double ro_formation, double invasion_zone_ro, double D_d) {
    double normalize_interval[2] = {0, 1}; // sigmoid interval

    // inputs forward transform:
    double inputs[INPUTS_COUNT] = {log(AO_d), log(ro_formation), log(invasion_zone_ro), D_d};
    double outputs[OUTPUTS_COUNT];

    for (int i = 0; i < INPUTS_COUNT; i++) {
        double value = inputs[i];
        double min = inference_1d_3l_chart::INPUT_MINS[i];
        double max = inference_1d_3l_chart::INPUT_MAXES[i];

        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{INPUTS_COUNT};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2], (float) inputs[3]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < OUTPUTS_COUNT; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_1d_3l_chart::OUTPUT_MINS[i], inference_1d_3l_chart::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + OUTPUTS_COUNT);
}

double inference_1d_3l_chart::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

