#include "DenseLayer.h"

// each neuron should store...
	// the weights of the neuron
	// the memory address of the bias value of the neuron
	// the number of weights/features
	// a pointer to the input features of the layer, such that when the layer's input features is updated, all the neurons are effectively
		// updated as well
DenseLayer::DenseLayer(double** layer_weights, double* layer_biases, double* running_means, double* running_variances, double* scales,
	double* shifts, double** training_layer_activation_values, double* layer_activation_array, int batch_size, int number_of_features,
	int number_of_neurons, double* layer_learning_rate, double* layer_regularization_rate) :

	number_of_features(number_of_features), number_of_neurons(number_of_neurons), batch_size(batch_size), layer_weights(layer_weights),
	layer_biases(layer_biases), running_means(running_means), running_variances(running_variances), scales(scales), shifts(shifts),

	// assign the activation arrays/output arrays that we are outputting to as the input arrays/feature arrays of the n + 1th layer
	training_activation_arrays(training_layer_activation_values),
	activation_array(layer_activation_array),

	// create new inputs that can then be used for the n - 1th layer
	training_input_features(allocate_memory_for_2D_array(batch_size, number_of_features)),
	input_features(new double[number_of_features]),

	// allocate memory for linear transformation values and derived values; used primarily for training
	linear_transform_derived_values(allocate_memory_for_2D_array(number_of_neurons, batch_size)),
	affine_transform_derived_values(allocate_memory_for_2D_array(number_of_neurons, batch_size)),

	// for training computation and eventually gradient descent
	linear_transform_values(allocate_memory_for_2D_array(number_of_neurons, batch_size)),
	normalized_values(allocate_memory_for_2D_array(number_of_neurons, batch_size)),

	// for the batch normalization formula
	training_means(new double[number_of_neurons]),
	training_variances(new double[number_of_neurons]),

	learning_rate(layer_learning_rate),
	regularization_rate(layer_regularization_rate),

	momentum(0.9)

{ }

// the neural network will delete everything else, and the activation arrays of this lth layer are the input features of the l + 1th layer,
// so do not need to deallocate the output features up until the output layer
DenseLayer::~DenseLayer()
{
	deallocate_memory_for_2D_array(training_input_features, batch_size);
	delete[] input_features;

	deallocate_memory_for_2D_array(linear_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(affine_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(linear_transform_values, number_of_neurons);
	deallocate_memory_for_2D_array(normalized_values, number_of_neurons);

	delete[] training_means;
	delete[] training_variances;
}

// methods for normal computation and prediction
// compute activation value of neuron for normal predictions
void DenseLayer::compute_activation_array()
{
	// calculate activation value
	linear_transform();

	// normalize output according to running mean and running variance
	normalize_activation_value();

	// affine transform implementation
	affine_transform();

	// relu function implementation
	relu_activation_function();
}

// calculate the activation value of the linear transform
void DenseLayer::linear_transform()
{
	for (int n = 0; n < number_of_neurons; n++)
	{
		activation_array[n] = 0;
		for (int f = 0; f < number_of_features; f++)
			activation_array[n] += layer_weights[n][f] * input_features[f];

		activation_array[n] += layer_biases[n];
	}
}

// normalize the activation value according to the running means and running variance
void DenseLayer::normalize_activation_value()
{
	for (int n = 0; n < number_of_neurons; n++)
		activation_array[n] = (activation_array[n] - running_means[n]) / (sqrt(running_variances[n] + 1e-5));
}

// scale the normalized activation value using the scale and shift
void DenseLayer::affine_transform()
{
	for (int n = 0; n < number_of_neurons; n++)
		activation_array[n] = scales[n] * activation_array[n] + shifts[n];
}

// apply a relu activation function implementation to the output; if negative number, just make 0
void DenseLayer::relu_activation_function()
{
	for (int n = 0; n < number_of_neurons; n++)
		if (activation_array[n] <= 0) activation_array[n] = 0;
}


// methods for training
void DenseLayer::training_compute_activation_arrays()
{
	// calculate the activation values of each linear transform of each sample in the batch
	training_linear_transform();

	// normalize each activation value in the batch
	training_normalize_activation_value();

	// transform each sample's activation value according to the current values of the scale and shift
	training_affine_transform();

	// ensure value is above 0, else make it 0
	training_relu_activation_function();
}

// for each sample, calculate the activation value
void DenseLayer::training_linear_transform()
{
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
		{
			linear_transform_values[n][s] = 0;
			for (int f = 0; f < number_of_features; f++)
				linear_transform_values[n][s] += (training_input_features[s][f] * layer_weights[n][f]);

			linear_transform_values[n][s] += layer_biases[n];
		}
}

// normalize each output activation value
void DenseLayer::training_normalize_activation_value()
{
	for (int n = 0; n < number_of_neurons; n++)
	{
		training_means[n] = 0;
		training_variances[n] = 0;

		// calculate mean of the activation values
		for (int s = 0; s < batch_size; s++)
			training_means[n] += linear_transform_values[n][s];
		training_means[n] /= batch_size;

		// calculate standard deviation of activation values
		for (int s = 0; s < batch_size; s++)
			training_variances[n] += pow(linear_transform_values[n][s] - training_means[n], 2);
		training_variances[n] /= batch_size;

		// calculate all the normalized activation values using the aforementioned
		// pass them into the normalized values array
		for (int s = 0; s < batch_size; s++)
			normalized_values[n][s] = (linear_transform_values[n][s] - training_means[n]) / (sqrt(training_variances[n] + 1e-5));
	}
}

// transform the normalized values; must calculate the mean and variance at this step
void DenseLayer::training_affine_transform()
{
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
			training_activation_arrays[s][n] = scales[n] * normalized_values[n][s] + shifts[n];
}

// go through all the normalized activations and just set them to 0 if less than or equal to 0
void DenseLayer::training_relu_activation_function()
{
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
			(training_activation_arrays[s][n] > 0) ? training_activation_arrays[s][n] : 0;
}

// calculate each neurons' derived values, provided the number of neurons in the next layer, the weights of the next layer, 
// and the derived values of the next layer
void DenseLayer::calculate_derived_values(double** next_layer_derived_values, double** next_layer_weights, int number_of_neurons_next_layer)
{
	// set all the derived values to 0
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
			affine_transform_derived_values[n][s] = 0;

	// current layer neuron
	for (int cln = 0; cln < number_of_neurons; cln++)
		// for the current sample
		for (int s = 0; s < batch_size; s++)
		{
			
			// for each neuron's derived value of the current sample
			for (int nln = 0; nln < number_of_neurons_next_layer; nln++)
			{
				// relu function implementation derivative
				if (training_activation_arrays[s][cln] <= 0)
					affine_transform_derived_values[cln][s] += 0;
				else
					affine_transform_derived_values[cln][s] +=
					next_layer_derived_values[nln][s] *
					next_layer_weights[nln][cln];
			}

			// net derivative of the linear transform formula
			linear_transform_derived_values[cln][s] =
				affine_transform_derived_values[cln][s] *
				scales[cln] *

				// derivative of the batch normalization formula
				1 / sqrt(training_variances[cln] + 1e-5) *
				(1 - 1.0 / batch_size -
					(pow(linear_transform_values[cln][s] - training_means[cln], 2.0)) /
					(batch_size * (training_variances[cln] + 1e-5))
				);
		}
}


// method to call the necessary functions to update the neuron's parameters
void DenseLayer::update_parameters()
{
	mini_batch_gd_weights_and_bias();

	mini_batch_gd_scales_and_shift();

	update_running_mean_and_variance();
}

// update the weights and biases based on the average change in the activation value
void DenseLayer::mini_batch_gd_weights_and_bias()
{
	double average_derived_value = 0;

	for (int n = 0; n < number_of_neurons; n++)
	{
		average_derived_value = 0;

		// calculate average value of the derivatives of the linear transformation
		for (int s = 0; s < batch_size; s++)
			average_derived_value += linear_transform_derived_values[n][s];
		average_derived_value /= batch_size;

		// update the value of the bias using the average and gradient descent
		layer_biases[n] = layer_biases[n] - ((*learning_rate) * average_derived_value);

		// get the average linear derived value in relation to the weights for each weight
		for (int f = 0; f < number_of_features; f++)
		{
			average_derived_value = 0;
			for (int s = 0; s < batch_size; s++)
				average_derived_value += linear_transform_derived_values[n][s] * training_input_features[s][f];
			average_derived_value /= batch_size;

			// update the values of the weights
			layer_weights[n][f] = layer_weights[n][f] - (*learning_rate) *
				((average_derived_value + *regularization_rate * layer_weights[n][f]));
		}
	}
}

// update the sclaes and shifts based on the average change in the linear transform derived values
void DenseLayer::mini_batch_gd_scales_and_shift()
{
	for (int n = 0; n < number_of_neurons; n++)
	{
		double average_affine_derived_value = 0;

		// calculate average value of the derivatives of the weights
		for (int s = 0; s < batch_size; s++)
			average_affine_derived_value += affine_transform_derived_values[n][s];
		average_affine_derived_value /= batch_size;

		// update the value of the shift
		shifts[n] = shifts[n] - (*learning_rate * average_affine_derived_value);

		// update the value of the scale
		// the derived value of the scale varies depending on the specific sample we're interacting with
		average_affine_derived_value = 0;
		for (int s = 0; s < batch_size; s++)
			average_affine_derived_value += affine_transform_derived_values[n][s] * normalized_values[n][s];
		average_affine_derived_value /= batch_size;

		scales[n] = scales[n] - (*learning_rate * average_affine_derived_value);
	}
}

// use exponential moving averages to update the running means and variances
void DenseLayer::update_running_mean_and_variance()
{
	for (int n = 0; n < number_of_neurons; n++)
	{
		running_means[n] = momentum * running_means[n] + (1 - momentum) * training_means[n];
		running_variances[n] = momentum * running_variances[n] + (1 - momentum) * training_variances[n];
	}
}


// return where the layer inputs will be stored
double* DenseLayer::get_input_features() const
{ return input_features; }
// return where the training layer input features will be stored
double** DenseLayer::get_training_input_features() const
{ return training_input_features; }


// return this layer's derived values
double** DenseLayer::get_linear_transform_derived_values() const
{ return linear_transform_derived_values; }
// return this layer's weight values
double** DenseLayer::get_layer_weights() const
{ return layer_weights; }
// return the number of neurons in this layer
int DenseLayer::get_number_of_neurons() const
{ return number_of_neurons; }