#include "1D_3L_chart/model_inference.h"
using namespace inference_1d_3l_chart;
// execution status consts:
const int SUCCESS = 0;
const int FAILED = -1;

// exit status consts
const int EXIT = 0;
const int CONTINUE = 1;

double read_double(string const &str, string const &attribute_name) {
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


int main(int argc, const char* argv[]) {
    const char* path_to_model;
    if (init_model_path(argc, argv, &path_to_model) == FAILED) {return FAILED;}

    double inputs[INPUTS_COUNT];
    string inputs_strings[INPUTS_COUNT];

    string exit_status;

    while (true) {
        // check exit status:
        int exit_status = parse_exit_status();
        if (exit_status == EXIT) {break;}
        else if (exit_status == FAILED) {return FAILED;}

        // reading input attribute's values
        try {
            for (int i = 0; i < INPUTS_COUNT; i++) {
                cout << "enter " << INPUTS[i] << ": ";
                cin >> inputs_strings[i];
                inputs[i] = read_double(inputs_strings[i], INPUTS[i]);
            }
        }
        catch (std::exception &e) {
            continue;
        }
        std::vector<double> result = calc_predictions(inputs[0], inputs[1], inputs[2], inputs[3]);

        for (int i = 0; i < OUTPUTS_COUNT; i++) {
            cout << OUTPUTS[i] << "=" << result[i] << " ";
        }
        cout << endl;
    }

    return SUCCESS;
}
