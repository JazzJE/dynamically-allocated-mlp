#include "DenseLayer.h"
#include "MemoryFunctions.h"
#include "Constants.h"

DenseLayer::DenseLayer(double** layer_weights, double* layer_biases, double* running_means, double* running_variances, double* scales,
	double* shifts, double* layer_activation_array, int number_of_features, int number_of_neurons, double* layer_learning_rate, 
	double* layer_regularization_rate) :

	number_of_features(number_of_features), number_of_neurons(number_of_neurons), layer_weights(layer_weights),
	layer_biases(layer_biases), running_means(running_means), running_variances(running_variances), scales(scales), shifts(shifts),

	// assign the activation arrays/output arrays that we are outputting to as the input arrays/feature arrays of the n + 1th layer
	activation_array(layer_activation_array),

	// create new inputs that can then be used for the n - 1th layer
	input_features(new double[number_of_features]),

	training_means(new double[number_of_neurons]),
	training_variances(new double[number_of_neurons]),

	learning_rate(layer_learning_rate),
	regularization_rate(layer_regularization_rate),

	// all things related to batch size will initially be set to nothing; the user will update the batch size of the nn
	training_activation_arrays(nullptr), training_input_features(nullptr), linear_transform_derived_values(nullptr), 
	affine_transform_derived_values(nullptr), normalized_values(nullptr), linear_transform_values(nullptr), batch_size(0)
{ }

// the neural network will delete everything else, and the activation arrays of this lth layer are the input features of the l + 1th layer,
// so do not need to deallocate the output features up until the output layer
DenseLayer::~DenseLayer()
{
	delete[] input_features;
	deallocate_memory_for_2D_array(training_input_features, batch_size);
	deallocate_memory_for_2D_array(linear_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(affine_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(linear_transform_values, number_of_neurons);
	deallocate_memory_for_2D_array(normalized_values, number_of_neurons);

	delete[] training_means;
	delete[] training_variances;
}

// updating the batch size of a layer requires a lot of alterations to dynamically allocated memory
void DenseLayer::update_arrays_using_batch_size(int new_batch_size, double** new_training_activation_arrays)
{
	// delete all dynamically allocated memory that relied on batch size
	deallocate_memory_for_2D_array(training_input_features, batch_size);
	deallocate_memory_for_2D_array(linear_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(affine_transform_derived_values, number_of_neurons);
	deallocate_memory_for_2D_array(linear_transform_values, number_of_neurons);
	deallocate_memory_for_2D_array(normalized_values, number_of_neurons);

	// recreate the arrays that relied on batch size but w/ the new batch size
	training_input_features = allocate_memory_for_2D_array(new_batch_size, number_of_features);
	linear_transform_derived_values = allocate_memory_for_2D_array(number_of_neurons, new_batch_size);
	affine_transform_derived_values = allocate_memory_for_2D_array(number_of_neurons, new_batch_size);
	linear_transform_values = allocate_memory_for_2D_array(number_of_neurons, new_batch_size);
	normalized_values = allocate_memory_for_2D_array(number_of_neurons, new_batch_size);

	training_activation_arrays = new_training_activation_arrays;
	batch_size = new_batch_size;
}

// methods for normal computation and prediction
// compute activation value of neuron for normal predictions
void DenseLayer::compute_activation_array()
{
	linear_transform();
	normalize_linear_transform_value();
	relu_activation_function();
}

// calculate the value of the linear transform
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

// normalize the linear transform according to the running means and running variance
void DenseLayer::normalize_linear_transform_value()
{
	for (int n = 0; n < number_of_neurons; n++)
	{
		activation_array[n] = (activation_array[n] - running_means[n]) / (sqrt(running_variances[n] + Constants::epsilon));
		activation_array[n] = scales[n] * activation_array[n] + shifts[n];
	}
}

// apply a relu activation function implementation to the output of the normalize linear transform value; if negative number, just make 0
void DenseLayer::relu_activation_function()
{
	for (int n = 0; n < number_of_neurons; n++)
		if (activation_array[n] <= 0) activation_array[n] = 0;
}

// methods for training
void DenseLayer::training_compute_activation_arrays()
{
	training_linear_transform();
	training_normalize_linear_transform_value();
	training_relu_activation_function();
}

// for each sample, calculate the linear transform value value
void DenseLayer::training_linear_transform()
{
	#pragma omp parallel for
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
		{
			double computed_value = 0;
			for (int f = 0; f < number_of_features; f++)
				computed_value += (training_input_features[s][f] * layer_weights[n][f]);
			computed_value += layer_biases[n];

			linear_transform_values[n][s] = computed_value;
		}
}

// normalize each output activation value
void DenseLayer::training_normalize_linear_transform_value()
{
	#pragma omp parallel for
	for (int n = 0; n < number_of_neurons; n++)
	{
		training_means[n] = 0;
		training_variances[n] = 0;

		// calculate means of linear transform values
		for (int s = 0; s < batch_size; s++)
			training_means[n] += linear_transform_values[n][s];
		training_means[n] /= batch_size;

		// calculate variances of linear transform values
		for (int s = 0; s < batch_size; s++)
			training_variances[n] += (linear_transform_values[n][s] - training_means[n]) * (linear_transform_values[n][s] - training_means[n]);
		training_variances[n] /= batch_size;

		// calculate all the normalized values using the aforementioned values
		// pass them into the normalized values array for backpropagation first
		for (int s = 0; s < batch_size; s++)
		{
			normalized_values[n][s] = (linear_transform_values[n][s] - training_means[n]) / (sqrt(training_variances[n] + Constants::epsilon));
			training_activation_arrays[s][n] = scales[n] * normalized_values[n][s] + shifts[n];
		}
	}
}

// go through all the normalized values and just set them to 0 if less than or equal to 0
void DenseLayer::training_relu_activation_function()
{
	#pragma omp parallel for collapse(2)
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
			training_activation_arrays[s][n] = (training_activation_arrays[s][n] > 0) ? training_activation_arrays[s][n] : 0;
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
	{
		// for the current sample
		for (int s = 0; s < batch_size; s++)
			// for each neuron's derived value of the current sample
			for (int nln = 0; nln < number_of_neurons_next_layer; nln++)
				// relu function implementation derivative
				if (training_activation_arrays[s][cln] > 0)
					affine_transform_derived_values[cln][s] +=
					next_layer_derived_values[nln][s] *
					next_layer_weights[nln][cln];

		double affine_derived_value_sum = 0.0; // ∑∂L/∂y_j; summation of derivative of loss in relation to each batch normalization value
		double affine_derived_value_difference_sum = 0.0; // ∑(x_j - μ)∂L/∂y_j; summation of derivatibe of loss in relation to each normalization value times the difference of the linear transformation value and mean

		#pragma omp parallel for reduction(+:affine_derived_value_sum, affine_derived_value_difference_sum)
		for (int s = 0; s < batch_size; s++)
		{
			affine_derived_value_sum += affine_transform_derived_values[cln][s];
			affine_derived_value_difference_sum += affine_transform_derived_values[cln][s] *
				(linear_transform_values[cln][s] - training_means[cln]);
		}

		// Then calculate gradient for each sample
		for (int s = 0; s < batch_size; s++)
		{
			linear_transform_derived_values[cln][s] = scales[cln] * 1.0 / sqrt(training_variances[cln] + Constants::epsilon) * (
				affine_transform_derived_values[cln][s] - affine_derived_value_sum / batch_size - 
				(linear_transform_values[cln][s] - training_means[cln]) * 
				affine_derived_value_difference_sum / (batch_size * (training_variances[cln] + Constants::epsilon))
			);
		}
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
	#pragma omp parallel for
	for (int n = 0; n < number_of_neurons; n++)
	{
		double average_derived_value = 0;

		// calculate average value of the derivatives of the linear transformation
		for (int s = 0; s < batch_size; s++)
			average_derived_value += linear_transform_derived_values[n][s];
		average_derived_value /= batch_size;

		layer_biases[n] = layer_biases[n] - ((*learning_rate) * average_derived_value);

		// get the average linear derived value in relation to the weights for each weight
		for (int f = 0; f < number_of_features; f++)
		{
			average_derived_value = 0;
			for (int s = 0; s < batch_size; s++)
				average_derived_value += linear_transform_derived_values[n][s] * training_input_features[s][f];
			average_derived_value /= batch_size;

			layer_weights[n][f] = layer_weights[n][f] - (*learning_rate) *
				((average_derived_value + (*regularization_rate) * layer_weights[n][f]));
		}
	}
}

// update the scales and shifts based on the average change in the linear transform derived values
void DenseLayer::mini_batch_gd_scales_and_shift()
{
	#pragma omp parallel for
	for (int n = 0; n < number_of_neurons; n++)
	{
		double average_affine_derived_value = 0;

		// calculate average value of the derivatives of the weights
		for (int s = 0; s < batch_size; s++)
			average_affine_derived_value += affine_transform_derived_values[n][s];
		average_affine_derived_value /= batch_size;

		// update the value of the shift
		shifts[n] = shifts[n] - ((*learning_rate) * average_affine_derived_value);

		// update the value of the scale
		// the derived value of the scale varies depending on the specific sample we're interacting with
		average_affine_derived_value = 0;
		for (int s = 0; s < batch_size; s++)
			average_affine_derived_value += affine_transform_derived_values[n][s] * normalized_values[n][s];
		average_affine_derived_value /= batch_size;

		scales[n] = scales[n] - ((*learning_rate) * average_affine_derived_value);
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

// getter methods
double* DenseLayer::get_input_features() const
{ return input_features; }
double** DenseLayer::get_training_input_features() const
{ return training_input_features; }
double** DenseLayer::get_linear_transform_derived_values() const
{ return linear_transform_derived_values; }
double** DenseLayer::get_layer_weights() const
{ return layer_weights; }
int DenseLayer::get_number_of_neurons() const
{ return number_of_neurons; }