#pragma once
#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

double* input_new_features(std::string* feature_names, bool* not_normalize, int number_of_features);
void update_batch_size_and_regen_new_neural_network(NeuralNetwork*& neural_network, const int* number_of_neurons_each_hidden_layer,
	int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers, int number_of_features,
	std::string weights_and_biases_file_name, std::string means_and_vars_file_name, std::string scales_and_shifts_file_name,
	int number_of_samples);
void input_parameter_rates(NeuralNetwork* neural_network);
void generate_border_line();
void update_weights_and_biases_file(std::string weights_and_biases_file_name, double*** weights, double** biases, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
void update_mv_or_ss_file(std::string mv_or_ss_file_name, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons);