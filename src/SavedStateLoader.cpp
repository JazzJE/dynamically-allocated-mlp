#include "NeuralNetwork.h"
#include "MemoryFunctions.h"

// initialize with the network's pointers
NeuralNetwork::SavedStateLoader::SavedStateLoader(double*** network_weights, double** network_biases, double* network_running_means,
	double* network_running_variances, double* network_scales, double* network_shifts, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int net_number_of_neurons_in_hidden_layers, int network_number_of_features) :

	current_weights(network_weights), current_biases(network_biases), current_running_means(network_running_means),
	current_running_variances(network_running_variances), current_scales(network_scales), current_shifts(network_shifts),
	number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer), number_of_hidden_layers(number_of_hidden_layers),
	net_number_of_neurons_in_hidden_layers(net_number_of_neurons_in_hidden_layers), number_of_features(network_number_of_features),

	// allocate memory to the pointers that will save the initial state of the neural network
	saved_weights(allocate_memory_for_weights(number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features)),
	saved_biases(allocate_memory_for_biases(number_of_neurons_each_hidden_layer, number_of_hidden_layers)),
	saved_running_means(new double[net_number_of_neurons_in_hidden_layers]),
	saved_running_variances(new double[net_number_of_neurons_in_hidden_layers]),
	saved_scales(new double[net_number_of_neurons_in_hidden_layers]),
	saved_shifts(new double[net_number_of_neurons_in_hidden_layers])
{
}

// delete all the dynamically allocated objects that are part of the bs loader
NeuralNetwork::SavedStateLoader::~SavedStateLoader()
{
	deallocate_memory_for_weights(saved_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(saved_biases, number_of_hidden_layers);
	delete[] saved_running_means;
	delete[] saved_running_variances;
	delete[] saved_scales;
	delete[] saved_shifts;
}

// update the best state to the current state of the neural network
void NeuralNetwork::SavedStateLoader::save_current_state()
{
	write_to_saved_weights();
	write_to_saved_biases();
	write_to_saved_means_and_variances();
	write_to_saved_scales_and_shifts();
}

// copy the values of the current weights to the best weights pointer
void NeuralNetwork::SavedStateLoader::write_to_saved_weights()
{
	// for the first layer
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
		for (int f = 0; f < number_of_features; f++)
			saved_weights[0][n][f] = current_weights[0][n][f];

	// for each layer after
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			for (int f = 0; f < number_of_neurons_each_hidden_layer[l - 1]; f++)
				saved_weights[l][n][f] = current_weights[l][n][f];

	// for output layer
	for (int f = 0; f < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; f++)
		saved_weights[number_of_hidden_layers][0][f] = current_weights[number_of_hidden_layers][0][f];
}

// copy the values of the current biases to the best biases pointer
void NeuralNetwork::SavedStateLoader::write_to_saved_biases()
{
	// copy each layers' biases into the best biases
	for (int l = 0; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			saved_biases[l][n] = current_biases[l][n];

	// copy the output layer's bias into the best bias
	saved_biases[number_of_hidden_layers][0] = current_biases[number_of_hidden_layers][0];
}

// copy the values of the current running means and variances to the best means and variances pointer
void NeuralNetwork::SavedStateLoader::write_to_saved_means_and_variances()
{
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		saved_running_means[n] = current_running_means[n];
		saved_running_variances[n] = current_running_variances[n];
	}
}

// copy the values of the current scales and shifts to the scales and shifts pointer
void NeuralNetwork::SavedStateLoader::write_to_saved_scales_and_shifts()
{
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		saved_scales[n] = current_scales[n];
		saved_shifts[n] = current_shifts[n];
	}
}

// update the current state of the neural network to the best state
void NeuralNetwork::SavedStateLoader::load_saved_state()
{
	write_to_current_weights();
	write_to_current_biases();
	write_to_current_means_and_variances();
	write_to_current_scales_and_shifts();
}

// copy the values of the best weights into the current weights
void NeuralNetwork::SavedStateLoader::write_to_current_weights()
{
	// for the first layer
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
		for (int f = 0; f < number_of_features; f++)
			current_weights[0][n][f] = saved_weights[0][n][f];

	// for each subsequent layer
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			for (int f = 0; f < number_of_neurons_each_hidden_layer[l - 1]; f++)
				current_weights[l][n][f] = saved_weights[l][n][f];

	// for output layer
	for (int f = 0; f < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; f++)
		current_weights[number_of_hidden_layers][0][f] = saved_weights[number_of_hidden_layers][0][f];
}

// copy the values of the best biases into the current biases
void NeuralNetwork::SavedStateLoader::write_to_current_biases()
{
	// copy the best biases into the layers' biases
	for (int l = 0; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			current_biases[l][n] = saved_biases[l][n];

	// copy the output layer's best bias into the output layer's bias
	current_biases[number_of_hidden_layers][0] = saved_biases[number_of_hidden_layers][0];
}

// copy the values of the best means and variances into the current means and variances
void NeuralNetwork::SavedStateLoader::write_to_current_means_and_variances()
{
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		current_running_means[n] = saved_running_means[n];
		current_running_variances[n] = saved_running_variances[n];
	}
}

// copy the values of the best scales and shifts into the current scales and shifts
void NeuralNetwork::SavedStateLoader::write_to_current_scales_and_shifts()
{
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		current_scales[n] = saved_scales[n];
		current_shifts[n] = saved_shifts[n];
	}
}