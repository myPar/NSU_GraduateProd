#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>
#include "../keras2cpp/model.h"
#include "../keras2cpp/baseLayer.h"

// 'ro_well', 'ro_formation', 'rad_well', 'kanisotrop'
// 'A04M01N', 'A10M01N', 'A20M05N', 'A40M05N', 'A80M10N'

using namespace std;
namespace inference_1d_2a {
    extern const char* DEFAULT_MODEL_PATH;
    extern keras2cpp::Model model;

    extern const string INPUTS[4];
    extern const string OUTPUTS[5];

    extern const double OUTPUT_MEANS[5];
    extern const double OUTPUT_STDS[5];

    extern const double OUTPUT_MAXES[5];
    extern const double OUTPUT_MINS[5];

    extern const double INPUT_MEANS[4];
    extern const double INPUT_STDS[4];

    extern const double INPUT_MAXES[4];
    extern const double INPUT_MINS[4];

    double normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]);
    double scale(double attribute_value, double attribute_mean, double attribute_std);
    // main calc method
    std::vector<double> calc_predictions(double ro_well, double ro_formation, double rad_well, double kanisotrop);
}
