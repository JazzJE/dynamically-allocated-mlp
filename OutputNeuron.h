#pragma once
#include "Neuron.h"
#include <iostream>
// functionally will have the same attributes as a regular neuron; HOWEVER, this neuron will only calculate using the linear
// transformation function with no other modifications to the activation
class OutputNeuron : public Neuron
{
public:

	OutputNeuron(double* neuron_weights, double* neuron_bias, double** training_input_features, 
		double** training_activation_arrays, double* input_features, double* activation_array,
		double* linear_transform_derived_values, int number_of_features, int batch_size, int neuron_number, 
		double* learning_rate, double* regularization_rate);

	// these methods will be different in that they will only use the linear transformation method, and that linear transformation
	// method will not need to update the linear transform array
	void compute_activation_value() override;
	void training_compute_activation_values() override;
	void training_linear_transform() override;
	void update_parameters() override;

};

