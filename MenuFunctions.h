#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

void input_parameter_rates(double& learning_rate, double& regularization_rate);
void generate_border_line();
void update_weights_and_biases_file(std::string weights_and_biases_file_name, double*** weights, double** biases, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
void update_mv_or_ss_file(std::string mv_or_ss_file_name, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons);