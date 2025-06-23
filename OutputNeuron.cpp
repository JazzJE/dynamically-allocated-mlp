#include "OutputNeuron.h"
OutputNeuron::OutputNeuron(double* neuron_weights, double* neuron_bias,
	double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
	double* linear_transform_derived_values, int number_of_features, int batch_size, int neuron_number, double* learning_rate, 
	double* regularization_rate) : 

	Neuron(neuron_weights, neuron_bias, nullptr, nullptr, training_input_features, training_activation_arrays, 
		input_features, activation_array, linear_transform_derived_values, nullptr, nullptr, number_of_features, 
		batch_size, neuron_number, learning_rate, regularization_rate)
{ 
	// the output neuron will not use batch normalization
	delete[] normalized_values;
}