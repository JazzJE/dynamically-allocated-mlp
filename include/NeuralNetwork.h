#pragma once
#include <fstream>
#include <random>
#include <limits>
#include <string>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <omp.h>
#include <filesystem>
#include "TrainingLogAndList.h"

class TrainingLogList;
class DenseLayer;
class OutputLayer;

class NeuralNetwork
{
private:

	// parameters and other variables
	const int net_number_of_neurons_in_hidden_layers;
	const int number_of_features;
	int batch_size;
	int patience;
	int number_of_epochs;

	// helper method to create dynamically allocated version of the array so nn components can access it
	const int* create_dynamically_allocated_number_of_neurons_each_hidden_layer_array(const int* number_of_neurons_each_hidden_layer,
		int number_of_hidden_layers);
	const int* const number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;

	// need these for saving state of nn
	double*** const network_weights;
	double** const network_biases;
	double* const network_running_means;
	double* const network_running_variances;
	double* const network_scales;
	double* const network_shifts;

	// controls for the training
	double* const learning_rate;
	double* const regularization_rate;

	DenseLayer** const hidden_layers;
	OutputLayer* const output_layer;
	
	// used for saving and loading states of nn
	class SavedStateLoader
	{
	private:

		// store pointers to the network weights, biases, means & variances, and scales & shifts
		double*** const current_weights;
		double** const current_biases;
		double* const current_running_means;
		double* const current_running_variances;
		double* const current_scales;
		double* const current_shifts;

		// use these to interact with the pointers
		const int* const number_of_neurons_each_hidden_layer;
		const int number_of_hidden_layers;
		const int net_number_of_neurons_in_hidden_layers;
		const int number_of_features;

		// store current saved when asked to do so
		// this MUST go after the above variables since these pointers use them to create adequate memory in the 
		// constructor initializer list
		double*** const saved_weights;
		double** const saved_biases;
		double* const saved_running_means;
		double* const saved_running_variances;
		double* const saved_scales;
		double* const saved_shifts;

	public:

		SavedStateLoader(double*** network_weights, double** network_biases, double* network_means, double* network_variances, 
			double* network_scales, double* network_shifts, const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers,
			int net_number_of_neurons_in_hidden_layers, int network_number_of_features);

		~SavedStateLoader();

		// methods to save the best state
		void save_current_state();
		void write_to_saved_weights();
		void write_to_saved_biases();
		void write_to_saved_means_and_variances();
		void write_to_saved_scales_and_shifts();

		// methods to load best state
		void load_saved_state();
		void write_to_current_weights();
		void write_to_current_biases();
		void write_to_current_means_and_variances();
		void write_to_current_scales_and_shifts();
	};

	SavedStateLoader ss_loader;

	// validating the neural network files by essentially just calling all of the validation methods
	void validate_neural_network_files(std::filesystem::path weights_and_biases_file_path, std::filesystem::path means_and_vars_file_path,
		std::filesystem::path scales_and_shifts_file_path, std::string weights_and_biases_file_name, std::string means_and_vars_file_name,
		std::string scales_and_shifts_file_name);

	void generate_weights_and_biases_file(std::filesystem::path weights_and_biases_file_path);
	void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);
	void validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_name);
	int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file);

	// mv or ss means means and variances OR scales and shifts files
	void generate_means_and_vars_file(std::filesystem::path means_and_vars_file_path);
	void generate_scales_and_shifts_file(std::filesystem::path scales_and_shifts_file_path);
	void validate_mv_or_ss_file(std::fstream& mv_or_ss_file, std::filesystem::path means_and_vars_file_path,
		std::string means_and_vars_file_name, void (NeuralNetwork::*generate_mv_or_ss_file)(std::filesystem::path));
	int find_error_mv_or_ss_file(std::fstream& mv_or_ss_file);

	void parse_weights_and_biases_file(std::filesystem::path weights_and_biases_file_path);
	// helper method for parsing weights and biases
	void parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file_name, int number_of_features, int number_of_neurons, int layer_index);
	void parse_mv_or_ss_file(std::filesystem::path mv_or_ss_file_path, double* means_or_scales, double* variances_or_shifts);

	// training components and its helper methods
	double early_stop_training(double** training_features_normalized, double* log_transformed_target_values,
		int number_of_samples, int lower_cross_validation_index = -1, int higher_cross_validation_index = -1);
	void train_network(double** normalized_batch_input_features, double* log_transformed_target_values);
	int* select_random_batch_indices(int number_of_samples, int lower_validation_index, int higher_validation_index); // select random batch samples for training
	void calculate_training_predictions(double** normalized_batch_input_features) const; // compute the predictions of all the samples; need to do this so we can have the input features of each layer and can thus apply gradient descent
	void backpropagate_derived_values(double* log_transformed_target_values);
	void update_arrays_using_batch_size();
	void update_parameters();

public:

	NeuralNetwork(const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers,
		int number_of_features, std::filesystem::path weights_and_biases_file_path, std::filesystem::path means_and_vars_file_path,
		std::filesystem::path scales_and_shifts_file_path, std::string weights_and_biases_file_name, std::string means_and_vars_file_name,
		std::string scales_and_shifts_file_name);

	~NeuralNetwork();

	// split data set into 5 different folds and train the network to as far as it can get
	// note that the network is reset to its initial state after training on each fold
	void k_fold_train(TrainingLogList& log_list, double** training_features, bool* not_normalize, double* log_transformed_target_values, 
		int number_of_samples, int number_of_folds);

	// this method is explicitly used to train on all samples of the data set
	// user at the end can decide to continue using the current state for the duration of the program
	// user must decide to save the neural network state in the menu themselves; this is to allow more flexibility
	void all_sample_train(TrainingLogList& log_list, double** normalized_training_features, double* log_transformed_target_values, int number_of_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double* input_features) const;

	// accessor methods for updating the neural network files
	double*** get_network_weights() const;
	double** get_network_biases() const;
	double* get_network_running_means() const;
	double* get_network_running_variances() const;
	double* get_network_scales() const;
	double* get_network_shifts() const;

	// setter methods for updating rates
	void set_learning_rate(double new_learning_rate);
	void set_regularization_rate(double new_regularization_rate);
	void set_patience(int new_patience);
	void set_number_of_epochs(int new_number_of_epochs);
	void set_batch_size(int new_batch_size);
	
};