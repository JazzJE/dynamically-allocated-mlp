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

Neuron::~Neuron()
{
	
}

// methods for normal computation and prediction
// compute activation value of neuron for normal predictions
void Neuron::compute_activation_value()
{
	// calculate activation value
	linear_transform();

	// normalize output according to running mean and running variance
	normalize_activation_value();

	// affinal transform implementation
	affinal_transform();

	// relu function implementation
	relu_activation_function();
}

// calculate the activation value of the linear transform
void Neuron::linear_transform()
{
	activation_array[neuron_number] = 0;
	for (int f = 0; f < number_of_features; f++)
		activation_array[neuron_number] += neuron_weights[f] * input_features[f];

	activation_array[neuron_number] += *neuron_bias;
}

// normalize the activation value according to the running means and running variance
void Neuron::normalize_activation_value()
{ activation_array[neuron_number] = (activation_array[neuron_number] - (*running_mean)) / (sqrt(*running_variance + 1e-5)); }

// scale the normalized activation value using the scale and shift
void Neuron::affinal_transform()
{ activation_array[neuron_number] = (*scale) * activation_array[neuron_number] + (*shift); }

// apply a relu activation function implementation to the output; if negative number, just make 0
void Neuron::relu_activation_function()
{ if (activation_array[neuron_number] <= 0) activation_array[neuron_number] = 0; }


// methods for training
void Neuron::training_compute_activation_values()
{
	// calculate the activation values of each linear transform of each sample in the batch
	training_linear_transform();

	// normalize each activation value in the batch
	training_normalize_activation_value();

	// transform each sample's activation value according to the current values of the scale and shift
	training_affinal_transform();

	// ensure value is above 0, else make it 0
	training_relu_activation_function();
}

// for each sample, calculate the activation value
void Neuron::training_linear_transform()
{
	for (int s = 0; s < batch_size; s++)
	{
		linear_transform_values[s] = 0;
		for (int f = 0; f < number_of_features; f++)
			linear_transform_values[s] += (training_input_features[s][f] * neuron_weights[f]);

		linear_transform_values[s] += (*neuron_bias);
	}
}

// normalize each output activation value
void Neuron::training_normalize_activation_value()
{
	training_mean = 0;
	training_variance = 0;

	// calculate mean of the activation values
	for (int s = 0; s < batch_size; s++)
		training_mean += linear_transform_values[s];
	training_mean /= batch_size;

	// calculate standard deviation of activation values
	for (int s = 0; s < batch_size; s++)
		training_variance += pow(linear_transform_values[s] - training_mean, 2);
	training_variance /= batch_size;

	// calculate all the normalized activation values using the aforementioned
	// pass them into the normalized values array
	for (int s = 0; s < batch_size; s++)
		normalized_values[s] = (linear_transform_values[s] - training_mean) / (sqrt(training_variance + 1e-5));
}

// transform the normalized values; must calculate the mean and variance at this step
void Neuron::training_affinal_transform()
{
	for (int s = 0; s < batch_size; s++)
		training_activation_arrays[s][neuron_number] = (*scale) * normalized_values[s] + (*shift);
}

// go through all the normalized activations and just set them to 0 if less than or equal to 0
void Neuron::training_relu_activation_function()
{
	for (int s = 0; s < batch_size; s++)
		if (training_activation_arrays[s][neuron_number] <= 0) training_activation_arrays[s][neuron_number] = 0;
}

// method to call the necessary functions to update the neuron's parameters
void Neuron::update_parameters()
{
	mini_batch_gd_weights_and_bias();

	mini_batch_gd_scales_and_shift();

	update_running_mean_and_variance();
}

// update the weights and biases based on the average change in the activation value
void Neuron::mini_batch_gd_weights_and_bias()
{
	double average_linear_derived_value = 0;

	// calculate average value of the derivatives of the linear transformation
	for (int s = 0; s < batch_size; s++)
		average_linear_derived_value += linear_transform_derived_values[s];
	average_linear_derived_value /= batch_size;

	// update the value of the bias using the average and gradient descent
	*neuron_bias = *neuron_bias - ((*learning_rate) * average_linear_derived_value);

	// update the values of the weights
	// the input feature stays constant regardless of the derived value
	for (int w = 0; w < number_of_features; w++)
		neuron_weights[w] = neuron_weights[w] - (*learning_rate) * (average_linear_derived_value * input_features[w] + *regularization_rate * neuron_weights[w]);
}

// update the sclaes and shifts based on the average change in the linear transform derived values
void Neuron::mini_batch_gd_scales_and_shift()
{
	double average_affinal_derived_value = 0;

	// calculate average value of the derivatives of the weights
	for (int s = 0; s < batch_size; s++)
		average_affinal_derived_value += affinal_transform_derived_values[s];
	average_affinal_derived_value /= batch_size;

	// update the value of the shift
	*shift = *shift - (*learning_rate * average_affinal_derived_value);

	// update the value of the scale
	// the derived value of the scale varies depending on the specific sample we're interacting with
	average_affinal_derived_value = 0;
	for (int s = 0; s < batch_size; s++)
		average_affinal_derived_value += affinal_transform_derived_values[s] * normalized_values[s];
	average_affinal_derived_value /= batch_size;

	*scale = *scale - (*learning_rate * average_affinal_derived_value);
}

// use exponential moving averages to update the running means and variances
void Neuron::update_running_mean_and_variance()
{
	*running_mean = momentum * training_mean + (1 - momentum) * (*running_mean);
	*running_variance = momentum * training_variance + (1 - momentum) * (*running_variance);
}

// return the training mean of the neuron
double Neuron::get_training_mean() const
{ return training_mean;  }
// return the training mean
double Neuron::get_training_variance() const 
{ return training_variance;  }
// return the scale's value
double Neuron::get_scale_value() const
{ return *scale; }