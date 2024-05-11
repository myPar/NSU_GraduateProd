#include "pz_2a/model_inference.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

const int ERROR = 1;
const int INPUTS_ATTR_COUNT = 4;
const int OUTPUTS_ATTR_COUNT = 1;
const string PREDICTIONS_FILE_NAME = "test_predictions.csv";

double parse_double(const string &str, const int pos) {
    double result;

    try {
        result = std::stod(str);
    }
    catch(std::invalid_argument const &e) {
        cerr << "pos - " << pos << ": invalid input - " << e.what() << endl;
        
        throw std::exception();
    }
    catch(std::out_of_range const &e) {
        cerr << "pos - " << pos << ": input is too big for double value - " << e.what() << endl;;

        throw std::exception();
    }

    return result;
}

vector<double> parse_inputs(const string &line) {
    vector<double> result;
    string token;
    stringstream input(line);

    for (int i = 0; i < INPUTS_ATTR_COUNT; i++) {
        getline(input, token, ',');

        if (!input.good()) {
            cerr << "error while reading file at pos - " << i << endl;

            throw std::exception();
        }
        result.push_back(parse_double(token, i));
    }

    return result;
}

void write_header(ofstream &ofs) {
    for (int i = 0; i < OUTPUTS_ATTR_COUNT; i++) {
        const string token = inference_pz_2a::OUTPUTS[i];
        ofs << token;
        
        if (i < OUTPUTS_ATTR_COUNT - 1) {
            ofs << ",";
        }
    }
    ofs << endl;
}

void write_line(ofstream &ofs, vector<double> &content_line) {
    size_t size = content_line.size();

    for (unsigned int i = 0; i < size; i++) {
        ofs << content_line[i];
        if (i < size - 1) {
            ofs << ",";
        }
    }

    ofs << endl;
}

int main(int argc, const char* argv[]) {
    if (argc <= 1) {
        cerr << "no input file specified" << endl;

        return ERROR;
    }
    // read input attributes data from here:
    string input_file_name = string(argv[1]);
    ifstream test_data_file(input_file_name);
    if (!test_data_file.is_open()) {
        cerr << "error open the input file" << endl;
        return ERROR;
    }
    // write results here:
    ofstream output_file(PREDICTIONS_FILE_NAME); 
    if (!output_file.is_open()) {
        cerr << "can't open the output file" << endl;
        return ERROR;
    }
    string line;
    write_header(output_file);  // write attribute names in the first string
    
    // parse input file and write predictions to output line by line:
    int line_pos = 0;
    bool is_first = true;

    while (true) {
        getline(test_data_file, line);

        // skip header:
        if (is_first) {
            is_first = false;
            line_pos++;

            continue;
        }
        vector<double> inputs;

        try {
            inputs = parse_inputs(line);
        }
        catch (std::exception &e) {
            cerr << "exception at line - " << line_pos << endl;

            return ERROR;
        }
        vector<double> predictions = inference_pz_2a::calc_predictions(inputs[0], inputs[1], inputs[2], inputs[3]);
        write_line(output_file, predictions);
        line_pos++;
    }

    // close files:
    test_data_file.close();
    output_file.close();
}