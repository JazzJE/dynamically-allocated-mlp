#include "DenseLayer.h"

// each neuron should store...
	// the weights of the neuron
	// the memory address of the bias value of the neuron
	// the number of weights/features
	// a pointer to the input features of the layer, such that when the layer's input features is updated, all the neurons are effectively
		// updated as well
DenseLayer::DenseLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
	double** training_layer_activation_values, double* layer_activation_array, int batch_size, int number_of_features, 
	int number_of_neurons, double* layer_learning_rate, double* layer_regularization_rate) : 
	
	number_of_features(number_of_features), number_of_neurons(number_of_neurons), batch_size(batch_size), 
	neurons(new Neuron*[number_of_neurons]), layer_weights(layer_weights),
	
	// assign the activation arrays/output arrays that we are outputting to as the input arrays/feature arrays of the n + 1th layer
	training_layer_activation_arrays(training_layer_activation_values),
	layer_activation_array(layer_activation_array),

	// create new inputs that can then be used for the n - 1th layer
	training_layer_input_features(allocate_memory_for_training_features(batch_size, number_of_features)),
	layer_input_features(new double[number_of_features]),

	// allocate memory for linear transformation values and derived values; used primarily for training
	layer_linear_transform_derived_values(allocate_memory_for_training_features(number_of_neurons, batch_size)),
	layer_affinal_transform_derived_values(allocate_memory_for_training_features(number_of_neurons, batch_size)),
	layer_linear_transform_values(allocate_memory_for_training_features(number_of_neurons, batch_size))

{
	// if there is no memory for the layer means and variances or scales and shifts, that means this is an output layer
	// and thus don't create a new default neuron
	if (layer_means_and_variances != nullptr && layer_scales_and_shifts != nullptr)
	{
		for (int n = 0; n < number_of_neurons; n++)
			neurons[n] = new Neuron(layer_weights[n], &layer_biases[n], layer_means_and_variances[n], layer_scales_and_shifts[n],
				training_layer_input_features, training_layer_activation_arrays, layer_input_features, layer_activation_array,
				layer_linear_transform_derived_values[n], layer_affinal_transform_derived_values[n], layer_linear_transform_values[n],
				number_of_features, batch_size, n, layer_learning_rate, layer_regularization_rate);
	}
}

// deallocate the neurons and the input features; the nth layer will have its output layer activation arrays be deallocated by the 
// (n + 1)th layer as they refer to the same thing
DenseLayer::~DenseLayer()
{
	for (int n = 0; n < number_of_neurons; n++)
		delete neurons[n];
	delete[] neurons;

	delete[] layer_input_features;
	deallocate_memory_for_training_features(training_layer_input_features, batch_size);
}

// calculate each neuron's activation values
void DenseLayer::compute_activation_array()
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n]->compute_activation_value();
}

// calculate each sample's activation_values
void DenseLayer::training_compute_activation_arrays()
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n]->training_compute_activation_values();
}

// calculate each neurons' derived values, provided the number of neurons in the next layer, the weights of the next layer, 
// and the derived values of the next layer
void DenseLayer::calculate_derived_values(double** next_layer_derived_values, double** next_layer_weights, int number_of_neurons_next_layer)
{
	// set all the derived values to 0
	for (int n = 0; n < number_of_neurons; n++)
		for (int s = 0; s < batch_size; s++)
			layer_affinal_transform_derived_values[n][s] = 0;

	// current layer neuron
	for (int cln = 0; cln < number_of_neurons; cln++)
	{
		// current neurons' scale value, training mean, and training variance
		double cln_scale_value = neurons[cln]->get_scale_value();
		double cln_training_mean = neurons[cln]->get_training_mean();
		double cln_training_variance = neurons[cln]->get_training_variance();

		// for the current sample
		for (int s = 0; s < batch_size; s++)
		{
			
			// for each neuron's derived value of the current sample
			for (int nln = 0; nln < number_of_neurons_next_layer; nln++)
			{
				// relu function implementation derivative
				if (training_layer_activation_arrays[s][cln] <= 0)
					layer_affinal_transform_derived_values[cln][s] += 0;
				else
					layer_affinal_transform_derived_values[cln][s] +=
					next_layer_derived_values[nln][s] *
					next_layer_weights[nln][cln];
			}

			// net derivative of the linear transform formula
			layer_linear_transform_derived_values[cln][s] =
				layer_affinal_transform_derived_values[cln][s] *
				cln_scale_value *

				// derivative of the batch normalization formula
				1 / sqrt(cln_training_variance + 1e-5) *
				(1 - 1.0 / batch_size -
					(pow(layer_linear_transform_values[cln][s] - cln_training_mean, 2.0)) /
					(batch_size * (cln_training_variance + 1e-5))
				);
		}
	}
}

// apply gradient descent to all of the weights and values inside of the neurons
void DenseLayer::update_parameters()
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n]->update_parameters();
}

// return where the layer inputs will be stored
double* DenseLayer::get_layer_input_features() const
{ return layer_input_features; }
// return the activation array
double* DenseLayer::get_layer_activation_array() const
{ return layer_activation_array; }

// return where the training layer input features will be stored
double** DenseLayer::get_training_layer_input_features() const
{ return training_layer_input_features; }
// return the training activation array
double** DenseLayer::get_training_layer_activation_arrays() const
{ return training_layer_activation_arrays; }

// return this layer's derived values
double** DenseLayer::get_layer_linear_transform_derived_values() const
{ return layer_linear_transform_derived_values; }
// return this layer's weight values
double** DenseLayer::get_layer_weights() const
{ return layer_weights; }
// return the number of neurons in this layer
int DenseLayer::get_number_of_neurons() const
{ return number_of_neurons; }