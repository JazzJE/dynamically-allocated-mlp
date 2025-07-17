#pragma once
#include <cmath>
#include <omp.h>

class DenseLayer
{
protected:

	double** const layer_weights;
	double* const layer_biases;

	double* const learning_rate;
	double* const regularization_rate;

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

	// used for training computations; these all rely on batch size
	int batch_size;
	double** training_input_features;
	double** training_activation_arrays;
	// these are the derived values for the weights and biases
	double** linear_transform_derived_values;
	// these are the derived values for the scales and shifts
	double** affine_transform_derived_values;
	double** linear_transform_values;
	double** normalized_values;

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

	DenseLayer(double** layer_weights, double* layer_biases, double* running_means, double* running_variances, double* scales, 
		double* shifts, double* layer_activation_array, int number_of_features, int number_of_neurons, double* learning_rate, 
		double* regularization_rate);

	~DenseLayer();

	// both methods used for connecting all of the layers to each other
	void deallocate_arrays_using_batch_size();
	void allocate_arrays_using_batch_size(int new_batch_size, double** new_training_activation_arrays);

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

