#include "Neuron.h"
Neuron::Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
	double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
	double* linear_transform_derived_values, double* affinal_transform_derived_values, 
	double* training_linear_transform_values, int number_of_features, int batch_size, int neuron_number,
	double* learning_rate, double* regularization_rate) :

	neuron_weights(neuron_weights), neuron_bias(neuron_bias), 
	
	// ternary operator since the output neuron will pass in the running mean and variance & scale and shift as nullptr objects
	running_mean(mean_and_variance ? &mean_and_variance[0] : nullptr), running_variance(mean_and_variance ? &mean_and_variance[1] : nullptr),
	scale(scale_and_shift ? &scale_and_shift[0] : nullptr), shift(scale_and_shift ? &scale_and_shift[1] : nullptr),
	
	number_of_features(number_of_features), momentum(0.9),
	neuron_number(neuron_number), batch_size(batch_size),
	
	// link to the n-1th layer and the n+1th layer
	input_features(input_features), activation_array(activation_array),
	training_input_features(training_input_features), training_activation_arrays(training_activation_arrays),

	// these will store the normalized values that are outputted from the batch normalization to apply gradient descent on the 
	// scales and shifts faster
	// it should only be accessed and updated by the neuron, not the layer
	normalized_values(new double[batch_size]),

	// the linear transform values will be used by the layer for computing derived values, while this neuron will compute these values
	// and pass them into this array to save time when backpropagating
	linear_transform_values(training_linear_transform_values),

	// each neuron will have access to its derived values that will be managed by the layer; used to also save time
		// weights and biases derived values
	linear_transform_derived_values(linear_transform_derived_values),
	// scales and shifts derived values
	affinal_transform_derived_values(affinal_transform_derived_values),

	learning_rate(learning_rate),
	regularization_rate(regularization_rate)

{
	training_mean = 0;
	training_variance = 0;
}