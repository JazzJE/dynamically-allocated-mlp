#pragma once
#include "Neuron.h"
#include "OutputNeuron.h"
#include "MemoryFunctions.h"
#include <iostream>
class DenseLayer
{
protected:

	Neuron** const neurons;
	const int number_of_neurons;

	// for normal prediction and computation
	const int number_of_features;
	double* const layer_input_features;
	double* const layer_activation_array;

	// to pass in the input features quickly into each neuron
	const int batch_size;
	double** const training_layer_input_features;
	double** const training_layer_activation_arrays;

	// each neurons' derived values within this layer
		// these are the derived values for the weights and biases
	double** const layer_linear_transform_derived_values;
		// these are the derived values for the scales and shifts
	double** const layer_affinal_transform_derived_values;
	// also for derived value computation
	double** const layer_weights;
	double** const layer_linear_transform_values;

public:

	// constructor that will initialize each neuron inside of the layer
	DenseLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
		double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, int number_of_features, 
		int number_of_neurons, double* learning_rate, double* regularization_rate);

	// delete all the dynamically allocated objects
	~DenseLayer();

	// layer will go through each neuron and compute its activation values
	void compute_activation_array();

	// for backpropagation and gradient descent
	void training_compute_activation_arrays();
	void calculate_derived_values(double** next_layer_derived_values, double** next_layer_weights, 
		int number_of_neurons_next_layer);
	void update_parameters();

	// getter/accessor methods
	double** get_training_layer_input_features() const;
	double** get_training_layer_activation_arrays() const;
	double* get_layer_activation_array() const;
	double* get_layer_input_features() const;

	// getter/accessors specifically for training
	double** get_layer_linear_transform_derived_values() const;
	double** get_layer_weights() const;
	int get_number_of_neurons() const;

};

