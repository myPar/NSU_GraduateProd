#include "model_inference.h"
using namespace inference_1d_2l_chart;

const char* inference_1d_2l_chart::DEFAULT_MODEL_PATH = "model_dir/keras2cpp.model";

keras2cpp::Model inference_1d_2l_chart::model = keras2cpp::Model::load(DEFAULT_MODEL_PATH);
// src attribute's intervals:
const double inference_1d_2l_chart::AO_D_MIN = 0.5;
const double inference_1d_2l_chart::AO_D_MAX = 500;

const double inference_1d_2l_chart::LAMBDA_MIN = 1;
const double inference_1d_2l_chart::LAMBDA_MAX = 5;

const double inference_1d_2l_chart::RO_FORMATION_MIN = 0.1;
const double inference_1d_2l_chart::RO_FORMATION_MAX = 10000;

// dst attribute interval (rok):
const double inference_1d_2l_chart::ROK_MIN = 0.0916247;
const double inference_1d_2l_chart::ROK_MAX = 41448.2;

float inference_1d_2l_chart::calc_prediction(float AO_d, float ro_formation, float lambda) {
    double normalize_interval[2] = {0, 1};
    // transform attributes
    AO_d = normalize(log(AO_d), log(AO_D_MIN), log(AO_D_MAX), normalize_interval);
    lambda = normalize(lambda, LAMBDA_MIN, LAMBDA_MAX, normalize_interval);
    ro_formation = normalize(log(ro_formation), log(RO_FORMATION_MIN), log(RO_FORMATION_MAX), normalize_interval);

    keras2cpp::Tensor input_tensor{3};
    input_tensor.data_ = {AO_d, lambda, ro_formation};
    keras2cpp::Tensor output_tensor = model(input_tensor);
    double rok_prediction = output_tensor.data_[0];

    // backward transform for rok attribute
    double rok_dst_interval[2] = {log(ROK_MIN), log(ROK_MAX)};

    rok_prediction = normalize(rok_prediction, normalize_interval[0], normalize_interval[1], rok_dst_interval);
    rok_prediction = pow(M_E, rok_prediction);

    return rok_prediction;
}


double inference_1d_2l_chart::normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

