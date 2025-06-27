#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <cctype>

// validating the neural network files by essentially just calling all of the validation methods
void validate_neural_network_files(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features,
	int net_number_of_neurons_in_hidden_layers, std::string weights_and_biases_file_name, std::string means_and_vars_file_name,
	std::string scales_and_shifts_file_name);

// weights and biases file methods
void generate_weights_and_biases_file(std::string weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features);
void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);
void validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_name,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features);
bool check_line_weights_and_biases_file(std::fstream& weights_and_biases_file, int number_of_features);

// methods for saving and generating running means and running variances, and generating the shifts and scales of each neuron
void generate_means_and_vars_file(std::string means_and_vars_file_name, int net_number_of_neurons_in_hidden_layers);
void generate_scales_and_shifts_file(std::string scales_and_shifts_file_name, int net_number_of_neurons_in_hidden_layers);

	// verifying means and variances file OR shifts and scales file
void validate_mv_or_ss_file(std::fstream& mv_or_ss_file, std::string means_and_vars_file_name, int net_number_of_neurons_in_hidden_layers);
int find_error_mv_or_ss_file(std::fstream& mv_or_ss_file, int net_number_of_neurons_in_hidden_layers);

// dataset file methods
void parse_dataset_file(std::fstream& dataset_file, double** training_features, double* target_values, std::string* feature_names,
	std::string& target_name, int number_of_features, int number_of_samples);
void validate_dataset_file(std::fstream& dataset_file, std::string dataset_file_name, int number_of_features);
int find_error_dataset_file(std::fstream& dataset_file, int number_of_features);

// miscallaneous methods
void randomize_training_samples(double** training_features, double* target_values, int number_of_samples);
int count_number_of_samples(std::string dataset_file_name);
int count_number_of_features(std::string dataset_file_name);
void identify_not_normalize_feature_columns(std::string* feature_names, bool* not_normalize, int number_of_features);