#include "OutputLayer.h"
#include "MemoryFunctions.h"

OutputLayer::OutputLayer(double** layer_weights, double* layer_biases, double* layer_activation_array, int number_of_features, 
	int number_of_neurons, double* layer_learning_rate, double* layer_regularization_rate) :

	DenseLayer(layer_weights, layer_biases, nullptr, nullptr, nullptr, nullptr, layer_activation_array, number_of_features, 
		number_of_neurons, layer_learning_rate, layer_regularization_rate)
{ }

// deallocate the output array of the output layer since the output layer's activation arrays do not have a layer pointing to them
// other than the output layer
OutputLayer::~OutputLayer()
{
	deallocate_memory_for_2D_array(training_activation_arrays, batch_size);
	delete[] activation_array;
}

// output layer must get rid of its output arrays
void OutputLayer::deallocate_arrays_using_batch_size()
{
	deallocate_memory_for_2D_array(training_activation_arrays, batch_size);
	deallocate_memory_for_2D_array(training_input_features, batch_size);
	deallocate_memory_for_2D_array(linear_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(affine_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(linear_transform_values, number_of_neurons);
	deallocate_memory_for_2D_array(normalized_values, number_of_neurons);
}

// only do linear transformation to compute the activation value
void OutputLayer::compute_activation_array()
{ linear_transform(); }

// only do linear transformations to compute each training activation value
void OutputLayer::training_compute_activation_arrays()
{ training_linear_transform(); }

// do the exact same thing as any other layer, but don't update the linear transform array as it doesn't exist since we won't use the 
// batch normalization formula here
void OutputLayer::training_linear_transform()
{
	for (int s = 0; s < batch_size; s++)
	{
		training_activation_arrays[s][0] = 0;
		for (int f = 0; f < number_of_features; f++)
			training_activation_arrays[s][0] += (training_input_features[s][f] * layer_weights[0][f]);

		training_activation_arrays[s][0] += layer_biases[0];
	}
}

// the only parameters that should be updated by the output neuron are the weights and biases, 
// as we don't use scales and shifts or means and variances
void OutputLayer::update_parameters()
{ mini_batch_gd_weights_and_bias(); }

// getter methods
double** OutputLayer::get_training_activation_arrays() const
{ return training_activation_arrays; }
double* OutputLayer::get_activation_array() const
{ return activation_array; }