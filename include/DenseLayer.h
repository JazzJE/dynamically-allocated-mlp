#pragma once
#include "MemoryFunctions.h"
#include <cmath>
#include <omp.h>
class DenseLayer
{
protected:

	double* learning_rate;
	double* regularization_rate;

	// for normalization
	double* const training_means;
	double* const training_variances;
	double* const running_means;
	double* const running_variances;
	double* const scales;
	double* const shifts;

	const int number_of_neurons;

	// for normal prediction and computation
	const int number_of_features;
	double* const input_features;
	double* const activation_array;

	// to pass in the input features quickly into each neuron
	const int batch_size;
	double** const training_input_features;
	double** const training_activation_arrays;

	// each neurons' derived values within this layer
		// these are the derived values for the weights and biases
	double** const linear_transform_derived_values;
		// these are the derived values for the scales and shifts
	double** const affine_transform_derived_values;
	
	// used for training computations
	double** const layer_weights;
	double* const layer_biases;
	double** const linear_transform_values;
	double** const normalized_values;

	const double momentum = 0.9;

	// methods for updating parameters
	void mini_batch_gd_weights_and_bias();
	void mini_batch_gd_scales_and_shift();
	void update_running_mean_and_variance();

	// for normal input features
	void linear_transform();
	void normalize_linear_transform_value();
	void relu_activation_function();

	// for training computations
	void training_linear_transform();
	void training_normalize_linear_transform_value();
	void training_relu_activation_function();


public:

	// constructor that will initialize each neuron inside of the layer
	DenseLayer(double** layer_weights, double* layer_biases, double* running_means, double* running_variances, double* scales, 
		double* shifts, double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, int number_of_features, 
		int number_of_neurons, double* learning_rate, double* regularization_rate);

	// delete all the dynamically allocated objects
	~DenseLayer();

	// layer will go through each neuron and compute its activation values
	void compute_activation_array();

	// layer will go through each samples' features and compute activation values
	void training_compute_activation_arrays();

	// for backpropagation and gradient descent
	void calculate_derived_values(double** next_layer_derived_values, double** next_layer_weights, int number_of_neurons_next_layer);
	void update_parameters();

	// getter/accessor methods
	double** get_training_input_features() const;
	double* get_input_features() const;
	double** get_linear_transform_derived_values() const;
	double** get_layer_weights() const;
	int get_number_of_neurons() const;

};

