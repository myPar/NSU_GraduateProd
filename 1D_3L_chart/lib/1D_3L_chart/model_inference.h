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
namespace inference_1d_3l_chart {
    extern const char* DEFAULT_MODEL_PATH;
    extern keras2cpp::Model model;
    const int INPUTS_COUNT = 4;
    const int OUTPUTS_COUNT = 1;

    extern const string INPUTS[INPUTS_COUNT];
    extern const string OUTPUTS[OUTPUTS_COUNT];

    extern const double OUTPUT_MAXES[OUTPUTS_COUNT];
    extern const double OUTPUT_MINS[OUTPUTS_COUNT];

    extern const double INPUT_MAXES[INPUTS_COUNT];
    extern const double INPUT_MINS[INPUTS_COUNT];

    double normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]);
    // main calc method
    std::vector<double> calc_predictions(double AO_d, double ro_formation, double invasion_zone_ro, double D_d);
}
