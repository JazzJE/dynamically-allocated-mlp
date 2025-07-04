#pragma once
#include "DenseLayer.h"
#include "OutputLayer.h"
#include "MemoryFunctions.h"
#include "StatisticsFunctions.h"
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <omp.h>

class NeuralNetwork
{
private:

	// used for deleting network and training
	const int net_number_of_neurons_in_hidden_layers;

	const int network_number_of_features;
	const int batch_size;

	const int* number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;

	double*** const network_weights;
	double** const network_biases;
	double* const network_running_means;
	double* const network_running_variances;
	double* const network_scales;
	double* const network_shifts;

	// controls for the training method
	double* const network_learning_rate;
	double* const network_regularization_rate;
	int patience; // how many times the nn should fail in decreasing the mse before stopping training of a fold
	int prompt_epoch; // number of epochs before prompting user if they'd like to stop training a fold

	DenseLayer** const hidden_layers;
	OutputLayer* const output_layer;
	
	// this will save and write out to the neural network the best state of the program
	// only really will be used during training
	struct BestStateLoader
	{
		// store pointers to the network weights, biases, means & variances, and scales & shifts
		double*** const current_weights;
		double** const current_biases;
		double* const current_running_means;
		double* const current_running_variances;
		double* const current_scales;
		double* const current_shifts;

		// use these to interact with the pointers
		const int* number_of_neurons_each_hidden_layer;
		const int number_of_hidden_layers;
		const int net_number_of_neurons_in_hidden_layers;
		const int number_of_features;

		// store pointers to the best states
		double*** const best_weights;
		double** const best_biases;
		double* const best_running_means;
		double* const best_running_variances;
		double* const best_scales;
		double* const best_shifts;

		BestStateLoader(double*** network_weights, double** network_biases, double* network_means, double* network_variances, 
			double* network_scales, double* network_shifts, const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers,
			int net_number_of_neurons_in_hidden_layers, int network_number_of_features);

		~BestStateLoader();

		// methods to save the best state
		void save_current_state();
		void write_to_best_weights();
		void write_to_best_biases();
		void write_to_best_means_and_variances();
		void write_to_best_scales_and_shifts();

		// methods to load best state
		void load_best_state();
		void write_to_current_weights();
		void write_to_current_biases();
		void write_to_current_means_and_variances();
		void write_to_current_scales_and_shifts();
	};

	// for initializing the neural network and parsing the data in its corresponding files
	
	// validating the neural network files by essentially just calling all of the validation methods
	void validate_neural_network_files(std::string weights_and_biases_file_name, std::string means_and_vars_file_name,
		std::string scales_and_shifts_file_name);

	void generate_weights_and_biases_file(std::string weights_and_biases_file_name);
	void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);
	void validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_name);
	int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file);

	// mv or ss means means and variances OR scales and shifts files
	void generate_means_and_vars_file(std::string means_and_vars_file_name);
	void generate_scales_and_shifts_file(std::string scales_and_shifts_file_name);
	void validate_mv_or_ss_file(std::fstream& mv_or_ss_file, std::string means_and_vars_file_name, void (NeuralNetwork::*generate_mv_or_ss_file)(std::string));
	int find_error_mv_or_ss_file(std::fstream& mv_or_ss_file);

	void parse_weights_and_biases_file(std::string weights_and_biases_file_name);
	// helper method for parsing weights and biases
	void parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file_name, int number_of_features, int number_of_neurons, int layer_index);
	void parse_mv_or_ss_file(std::string mv_or_ss_file_name, double* means_or_scales, double* variances_or_shifts);


	// training components
	void early_stop_training(BestStateLoader& bs_loader, double** training_features_normalized, double* log_transformed_target_values,
		int lower_cross_validation_index, int higher_cross_validation_index, int number_of_samples);
	void train_network(double** normalized_batch_input_features, double* log_transformed_target_values);
	int* select_random_batch_indices(int number_of_samples, int lower_validation_index = -1, int higher_validation_index = -1); // select random batch samples for training
	void calculate_training_predictions(double** normalized_batch_input_features) const; // compute the predictions of all the samples; need to do this so we can have the input features of each layer and can thus apply gradient descent
	void backpropagate_derived_values(double* log_transformed_target_values);
	void update_parameters();

public:

	// initialize all the variables
	NeuralNetwork(const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers,
		int number_of_features, int batch_size, double learning_rate, double regularization_rate, int patience, int prompt_epoch, 
		std::string weights_and_biases_file_path, std::string means_and_vars_file_path, std::string scales_and_shifts_file_path);

	// destructor to delete neural network
	~NeuralNetwork();

	// train the neural network five times based on the number of training samples
	void five_fold_train(double** training_features, bool* not_normalize, double* log_transformed_target_values, int number_of_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double* input_features) const;

	// get the number of neurons in each hidden layer and turn it into a dynamic array so that the best state loader
	// can access it
	int* get_number_of_neurons_each_hidden_layer() const;

	// mutator/setter methods for rates
	void set_regularization_rate(double r_rate);
	void set_learning_rate(double l_rate);
	void set_patience(int p);
	void set_prompt_epoch(int i);

	// accessor methods for updating the neural network files
	double*** get_network_weights() const;
	double** get_network_biases() const;
	double* get_network_running_means() const;
	double* get_network_running_variances() const;
	double* get_network_scales() const;
	double* get_network_shifts() const;
	double get_learning_rate() const;
	double get_regularization_rate() const;
	int get_patience() const;
	int get_prompt_epoch() const;
};

