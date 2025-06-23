#include "OutputLayer.h"
OutputLayer::OutputLayer(double** layer_weights, double* layer_biases,
	double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, 
	int number_of_features, int number_of_neurons, double* layer_learning_rate, double* layer_regularization_rate) :

	DenseLayer(layer_weights, layer_biases, nullptr, nullptr, training_layer_activation_arrays, 
		layer_activation_array, batch_size, number_of_features, number_of_neurons, layer_learning_rate, layer_regularization_rate)
{ 
	neurons[0] = new OutputNeuron(layer_weights[0], &layer_biases[0], training_layer_input_features, training_layer_activation_arrays,
		layer_input_features, layer_activation_array, layer_linear_transform_derived_values[0], number_of_features, batch_size, 0,
		layer_learning_rate, layer_regularization_rate);

	// delete the scales and shifts & the linear transform arrays as we won't use them
	deallocate_memory_for_training_features(layer_linear_transform_values, 1);
	deallocate_memory_for_training_features(layer_affinal_transform_derived_values, 1);
}

// delete the output neuron array
OutputLayer::~OutputLayer()
{
	delete[] layer_activation_array;
	deallocate_memory_for_training_features(training_layer_activation_arrays, batch_size);
}

// for the singular output neuron
void OutputLayer::compute_activation_array()
{ (*neurons)->compute_activation_value(); }

// for the outputs of the training arrays
void OutputLayer::training_compute_activation_arrays()
{ (*neurons)->training_compute_activation_values(); }