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

// only do linear transformation to compute the activation value
void OutputNeuron::compute_activation_value()
{
	linear_transform();
}

// only do linear transformations to compute each training activation value
void OutputNeuron::training_compute_activation_values()
{
	training_linear_transform();
}

// do the exact same thing as any other layer, but don't update the linear transform array as it doesn't exist since we won't use the 
// batch normalization formula here
void OutputNeuron::training_linear_transform()
{
	for (int s = 0; s < batch_size; s++)
	{
		training_activation_arrays[s][neuron_number] = 0;
		for (int f = 0; f < number_of_features; f++)
			training_activation_arrays[s][neuron_number] += (training_input_features[s][f] * neuron_weights[f]);

		training_activation_arrays[s][neuron_number] += (*neuron_bias);
	}
}

// the only parameters that should be updated by the output neuron are the weights and biases
void OutputNeuron::update_parameters()
{
	mini_batch_gd_weights_and_bias();
}