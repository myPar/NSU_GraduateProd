#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <string>
#include <stdexcept>
#include "../keras2cpp/model.h"
#include "../keras2cpp/baseLayer.h"

using namespace std;
namespace inference_1d_2l_chart {
    extern const char* DEFAULT_MODEL_PATH;
    extern keras2cpp::Model model;
    const int INPUTS_COUNT = 3;
    const int OUTPUTS_COUNT = 1;
    // src attribute's intervals:
    extern const double AO_D_MIN;
    extern const double AO_D_MAX;

    extern const double LAMBDA_MIN;
    extern const double LAMBDA_MAX;

    extern const double RO_FORMATION_MIN;
    extern const double RO_FORMATION_MAX;

    // dst attribute interval (rok):
    extern const double ROK_MIN;
    extern const double ROK_MAX;

    extern const string INPUTS[INPUTS_COUNT];
    extern const string OUTPUTS[OUTPUTS_COUNT];

    double normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]);
    // main calc method
    vector<double> calc_predictions(float AO_d, float ro_formation, float lambda);
}
