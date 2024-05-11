#include "model_inference.h"
using namespace inference_1d_3l;

const char* inference_1d_3l::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_1d_3l::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_1d_3l::INPUTS[INPUTS_COUNT] = {"ro_well", "ro_formation", "d_well", "invasion_zone_ro", "invasion_zone_h"};
const string inference_1d_3l::OUTPUTS[OUTPUTS_COUNT] = {"A04M01N", "A10M01N", "A20M05N", "A40M05N", "A80M10N"};

// output attributes inference statistic:
const double inference_1d_3l::OUTPUT_MINS[OUTPUTS_COUNT] = {-3.151572379658513, -2.4601808670160583, -2.4499442210414473, -2.425190110742185, -2.4101914620269955};
const double inference_1d_3l::OUTPUT_MAXES[OUTPUTS_COUNT] = {9.556727578146436, 9.817798936675766, 10.010232109966116, 10.153456075492668, 10.280258218069612};

const double inference_1d_3l::INPUT_MINS[INPUTS_COUNT] = {-4.605170185988091, -2.3025850929940455, 0.08, -2.3025850929940455, 0.0};
const double inference_1d_3l::INPUT_MAXES[INPUTS_COUNT] = {6.907755278982137, 9.210340371976184, 0.4, 9.210340371976184, 2.5};

std::vector<double> inference_1d_3l::calc_predictions(double ro_well, double ro_formation, double d_well, double invasion_zone_ro, double invasion_zone_h) {
    double normalize_interval[2] = {0, 1}; // sigmoid interval

    // inputs forward transform:
    double inputs[INPUTS_COUNT] = {log(ro_well), log(ro_formation), d_well, log(invasion_zone_ro), invasion_zone_h};
    double outputs[OUTPUTS_COUNT];

    for (int i = 0; i < INPUTS_COUNT; i++) {
        double value = inputs[i];
        double min = inference_1d_3l::INPUT_MINS[i];
        double max = inference_1d_3l::INPUT_MAXES[i];

        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{INPUTS_COUNT};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < OUTPUTS_COUNT; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_1d_3l::OUTPUT_MINS[i], inference_1d_3l::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + OUTPUTS_COUNT);
}

double inference_1d_3l::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

