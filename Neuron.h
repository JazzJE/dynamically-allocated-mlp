#pragma once
#include "StatisticsFunctions.h"
#include <iostream>
class Neuron
{
protected:

	double* const neuron_weights;
	double* const neuron_bias;

	// for normal guessing and computation
	const int number_of_features;
	double* const input_features;
	double* const activation_array;

	// for training with batch gradient descent
	double** const training_input_features;
	double** const training_activation_arrays;
	double* const linear_transform_values;
	double* const normalized_values;
	double* const learning_rate;
	double* const regularization_rate;

	// derived values of the linear transform function
	double* const linear_transform_derived_values;
	// derived values of the affinal transform function
	double* const affinal_transform_derived_values;

	double training_mean;
	double training_variance;

	const double momentum;
	const int batch_size;

	// the neuron number is used to access the associated column of the activation arrays
	const int neuron_number;

	double* const running_mean;
	double* const running_variance;
	double* const scale;
	double* const shift;

	// methods for updating parameters
	void mini_batch_gd_weights_and_bias();
	void mini_batch_gd_scales_and_shift();
	void update_running_mean_and_variance();

public:

	// constructor
	Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
		double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
		double* linear_transform_derived_values, double* affinal_transform_derived_values, double* linear_transform_values, 
		int number_of_features, int batch_size, int neuron_number, double* learning_rate, double* regularization_rate);

	// delete the dynamic memory still left, which is the derived values
	~Neuron();

	// methods for calculating activation values for normal input features
	virtual void compute_activation_value();
	void linear_transform();
	void normalize_activation_value();
	void affinal_transform();
	void relu_activation_function();

	// methods to compute values for training
	virtual void training_compute_activation_values();
	virtual void training_linear_transform();
	void training_normalize_activation_value();
	void training_affinal_transform();
	void training_relu_activation_function();

	// apply minibatch gradient descent to the neuron's weights, biases, and the scales and shifts
	// also update the running means and variances
	virtual void update_parameters();

	// acessor/getter methods
	double get_training_mean() const;
	double get_training_variance() const;
	double get_scale_value() const;
};

