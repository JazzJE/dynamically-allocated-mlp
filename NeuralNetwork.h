#pragma once
#include "DenseLayer.h"
#include "OutputLayer.h"
#include "MemoryFunctions.h"
#include "StatisticsFunctions.h"
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <omp.h>

class NeuralNetwork
{
private:

	// used for deleting network and training
	const int net_number_of_neurons_in_hidden_layers;

	const int network_number_of_features;
	const int batch_size;

	double* const network_learning_rate;
	double* const network_regularization_rate;

	const int* number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;

	double*** const network_weights;
	double** const network_biases;
	double* const network_running_means;
	double* const network_running_variances;
	double* const network_scales;
	double* const network_shifts;

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
	void parse_weights_and_biases_file(std::string weights_and_biases_file_name, double*** weights, double** biases,
		const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
	void parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file_name, int number_of_features, int number_of_neurons,
		double*** weights, double** biases, int layer_index);
	// parse the means and variances OR scales and shifts file
	void parse_mv_or_ss_file(std::string mv_or_ss_file_name, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons_in_hidden_layers);

	// training components
	void early_stop_training(BestStateLoader& bs_loader, double** training_features_normalized, double* target_values,
		int lower_cross_validation_index, int higher_cross_validation_index, int number_of_samples);
	void train_network(double** normalized_batch_input_features, double* target_values);
	int* select_random_batch_indices(int number_of_samples, int lower_validation_index = -1, int higher_validation_index = -1); // select random batch samples for training
	void calculate_training_predictions(double** normalized_batch_input_features); // compute the predictions of all the samples; need to do this so we can have the input features of each layer and can thus apply gradient descent
	void backpropagate_derived_values(double* target_values);
	void update_parameters();

public:

	// initialize all the variables
	NeuralNetwork(const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers,
		int number_of_features, int batch_size, double learning_rate, double regularization_rate, std::string weights_and_biases_file_name,
		std::string means_and_vars_file_name, std::string scales_and_shifts_file_name);

	// destructor to delete neural network
	~NeuralNetwork();

	// train the neural network five times based on the number of training samples
	void five_fold_train(double** training_features, bool* not_normalize, double* target_values, int number_of_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double* input_features);

	// get the number of neurons in each hidden layer and turn it into a dynamic array so that the best state loader
	// can access it
	int* get_number_of_neurons_each_hidden_layer();

	// mutator/setter methods for rates
	void set_regularization_rate(double r_rate);
	void set_learning_rate(double l_rate);

	// accessor methods for updating the neural network files
	double*** get_network_weights();
	double** get_network_biases();
	double* get_network_running_means();
	double* get_network_running_variances();
	double* get_network_scales();
	double* get_network_shifts();

};

