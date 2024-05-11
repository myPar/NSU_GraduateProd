#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cmath>
#include <string>
#include <stdexcept>

#define _USE_MATH_DEFINES

using namespace std;

const char* DEFAULT_MODEL_PATH = "../model_dir/model.pt";
// execution status consts:
const int SUCCESS = 0;
const int FAILED = -1;

// exit status consts
const int EXIT = 0;
const int CONTINUE = 1;

// src attribute's intervals:
const double AO_D_MIN = 0.5;
const double AO_D_MAX = 500;

const double LAMBDA_MIN = 1;
const double LAMBDA_MAX = 5;

const double RO_FORMATION_MIN = 0.1;
const double RO_FORMATION_MAX = 10000;

// dst attribute interval (rok):
const double ROK_MIN = 0.0916247;
const double ROK_MAX = 41448.2;

double normalize(double attribute_value, double attribute_min, double attribute_max, double normalize_interval[2]) {
    double interval_min = normalize_interval[0];
    double delta = normalize_interval[1] - normalize_interval[0];
    double result = ((attribute_value - attribute_min) / (attribute_max - attribute_min)) * delta + interval_min;

    return result;
}

double read_double(string const &str, const char* attribute_name) {
    double result;

    try {
        result = std::stod(str);
    }
    catch(std::invalid_argument const &e) {
        cerr << attribute_name << ": invalid input\n";
        
        throw std::exception();
    }
    catch(std::out_of_range const &e) {
        cerr << attribute_name << ": input is too big for double value\n";

        throw std::exception();
    }

    return result;
}

int init_model_path(int argc, const char* argv[], const char **path_to_model) {
    if (argc == 1) {
        (*path_to_model) = DEFAULT_MODEL_PATH;
    }
    else if (argc == 2) {
        (*path_to_model) = argv[1];
    }
    else {
        cerr << "invalid args count, should be 1 - path to model .pt file" << endl;

        return FAILED;
    }
    return SUCCESS;
}

int parse_exit_status() {
    string exit_status;
    cout << "type 'e'/'exit' to exit the program or type 'c'/'continue' to continue\n";
    cin >> exit_status;

    if (exit_status.compare("e") == SUCCESS || exit_status.compare("exit") == SUCCESS) {
        return EXIT;
    }
    else if (!(exit_status.compare("c") == SUCCESS || exit_status.compare("continue") == SUCCESS)) {
        cout << "invalid input" << endl;

        return FAILED;
    }
    return CONTINUE;
}

int load_model(const char* path, torch::jit::script::Module &module) {
    cout << "loading the model...\n";

    try {
        cout << "path=" << path << endl;
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        cerr << "can't load the model:\n";
        cerr << e.what();

        return FAILED;
    }
    cout << "model is loaded successfully\n";

    return SUCCESS;
}

int main(int argc, const char* argv[]) {
    const char* path_to_model;
    if (init_model_path(argc, argv, &path_to_model) == FAILED) {return FAILED;}

    // load model:
    torch::jit::script::Module module;
    load_model(path_to_model, module);

    float AO_d;
    float ro_formation;
    float lambda;

    string AO_d_input;
    string ro_formation_input;
    string lambda_input;

    double normalize_interval[2] = {0, 1};
    string exit_status;

    while (true) {
        // check exit status:
        int exit_status = parse_exit_status();
        if (exit_status == EXIT) {break;}
        else if (exit_status == FAILED) {return FAILED;}

        // reading input attribute's values
        try {
            cout << "enter AO/d: ";
            cin >> AO_d_input;
            AO_d = read_double(AO_d_input, "AO/d");

            cout << "enter lambda: ";
            cin >> lambda_input;
            lambda = read_double(lambda_input, "lambda");
            
            cout << "enter ro_formation: ";
            cin >> ro_formation_input;
            ro_formation = read_double(ro_formation_input, "ro_formation");
        }
        catch (std::exception &e) {
            continue;
        }
        // transform attributes
        AO_d = normalize(log(AO_d), log(AO_D_MIN), log(AO_D_MAX), normalize_interval);
        lambda = normalize(lambda, LAMBDA_MIN, LAMBDA_MAX, normalize_interval);
        ro_formation = normalize(log(ro_formation), log(RO_FORMATION_MIN), log(RO_FORMATION_MAX), normalize_interval);

        // make predictions
        float input_tensor_data[] = {AO_d, lambda, ro_formation};
        torch::Tensor input_tensor = torch::from_blob(input_tensor_data, {3}, torch::kFloat32);
        vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        float rok_prediction = module.forward(inputs).toTensor()[0].item<float>();

        // backward transform for rok attribute
        double rok_dst_interval[2] = {log(ROK_MIN), log(ROK_MAX)};

        rok_prediction = normalize(rok_prediction, normalize_interval[0], normalize_interval[1], rok_dst_interval);
        rok_prediction = pow(M_E, rok_prediction);

        cout << "prediction=" << rok_prediction << endl;
    }

    return SUCCESS;
}
