#include "model_inference.h"
using namespace inference_pz_3l;

const char* inference_pz_3l::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_pz_3l::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// input attributes names:
const string inference_pz_3l::INPUTS[INPUTS_COUNT] = {"ro_well", "ro_formation", "r_well", "invasion_zone_h", "invasion_zone_ro"};
const string inference_pz_3l::OUTPUTS[OUTPUTS_COUNT] = {"PZ"};

// output attributes inference statistic:
const double inference_pz_3l::OUTPUT_MINS[OUTPUTS_COUNT] = {-2.3905249008422538};
const double inference_pz_3l::OUTPUT_MAXES[OUTPUTS_COUNT] = {9.63855800053032};

const double inference_pz_3l::INPUT_MINS[INPUTS_COUNT] = {-4.605170185988091, -2.3025850929940455, 0.04, 0.0, -2.3025850929940455};
const double inference_pz_3l::INPUT_MAXES[INPUTS_COUNT] = {6.907755278982137, 9.210340371976184, 0.2, 2.4, 9.210340371976184};

std::vector<double> inference_pz_3l::calc_predictions(double ro_well, double ro_formation, double r_well, double invasion_zone_h, double invasion_zone_ro) {
    double normalize_interval[2] = {0, 1}; // sigmoid interval

    // inputs forward transform:
    double inputs[INPUTS_COUNT] = {log(ro_well), log(ro_formation), r_well, invasion_zone_h, log(invasion_zone_ro),};
    double outputs[OUTPUTS_COUNT];

    for (int i = 0; i < INPUTS_COUNT; i++) {
        double value = inputs[i];
        double min = inference_pz_3l::INPUT_MINS[i];
        double max = inference_pz_3l::INPUT_MAXES[i];

        inputs[i] = normalize(value, min, max, normalize_interval);
    }
    keras2cpp::Tensor input_tensor{INPUTS_COUNT};
    input_tensor.data_ = {(float) inputs[0], (float) inputs[1], (float) inputs[2], (float) inputs[3], (float) inputs[4]};
    keras2cpp::Tensor output_tensor = model(input_tensor);

    // backward transform for output attributes:
    for (int i = 0; i < OUTPUTS_COUNT; i++) {
        double value = outputs[i] = output_tensor.data_[i];
        double dst_interval[2] = {inference_pz_3l::OUTPUT_MINS[i], inference_pz_3l::OUTPUT_MAXES[i]};
        // normalize backward
        value = outputs[i] = normalize(value, normalize_interval[0], normalize_interval[1], dst_interval);
        // take exp (log^-1)
        outputs[i] = pow(M_E, value);
    }

    return std::vector<double>(outputs, outputs + OUTPUTS_COUNT);
}

double inference_pz_3l::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

