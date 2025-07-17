#pragma once
#include <iostream>
#include <filesystem>
#include <unordered_set>
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <cmath>

// all the options that can be selected by user
enum MenuOptions
{
	FIRST_OPTION = '1',
	LAST_OPTION = '9',

	ALL_SAMPLE_TRAIN_OPTION = '1',
	FIVE_FOLD_TRAIN_OPTION = '2',
	SAVE_NETWORK_STATE_OPTION = '3',
	RANDOMIZE_SAMPLES_OPTION = '4',
	PREDICT_PROVIDED_FEATURES_OPTION = '5',
	PREDICT_RANDOM_FEATURES_OPTION = '6',
	PRINT_TRAINING_LOGS_OPTION = '7',
	SAVE_TRAINING_LOGS_OPTION = '8',
	EXIT_PROGRAM_OPTION = '9',
};

class TrainingLogList;
class NeuralNetwork;

MenuOptions get_option();
void generate_border_line();

// menu-selected options
void save_neural_network_state(NeuralNetwork& neural_network, std::filesystem::path weights_and_biases_file_path, 
	std::filesystem::path means_and_vars_file_path, std::filesystem::path scales_and_shifts_file_path, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features, int net_number_of_neurons_in_hidden_layers);
void all_sample_train_network_option(NeuralNetwork& neural_network, TrainingLogList& log_list, double** all_features_normalized, double* log_transformed_target_values,
	int number_of_samples);
void k_fold_train_network_option(NeuralNetwork& neural_network, TrainingLogList& log_list, double** training_features,
	bool* not_normalize, double* log_transformed_target_values, int number_of_samples);
void predict_with_provided_features_option(NeuralNetwork& neural_network, std::string* feature_names, std::string target_name,
	double* all_features_means, double* all_features_variances, bool* not_normalize, int number_of_features);
void predict_with_random_features_option(NeuralNetwork& neural_network, std::string* feature_names, std::string target_name,
	double** training_features, double** all_features_normalized, double* target_values, double* log_transformed_target_values,
	int* sample_numbers, int number_of_samples, int number_of_features);

// helper methods
double* input_new_features(std::string* feature_names, bool* not_normalize, int number_of_features);
void input_batch_size(int& new_batch_size, int number_of_samples, bool using_all_samples, int number_of_folds = 1);
void input_rates(double& new_learning_rate, double& new_regularization_rate, int& new_patience, int& new_number_of_epochs);
void input_number_of_folds(int& number_of_folds);
void input_session_name(std::string& new_session_name);
void randomize_training_samples(double** training_features, double* target_values, double* log_transformed_target_values,
	int* sample_numbers, int number_of_samples);
void update_weights_and_biases_file(std::filesystem::path weights_and_biases_file_path, double*** weights, double** biases, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
void update_mv_or_ss_file(std::filesystem::path mv_or_ss_file_path, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons);