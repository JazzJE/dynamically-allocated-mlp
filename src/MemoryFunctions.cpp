#include "MemoryFunctions.h"

// methods to allocate memory for weights

	// allocate 3d array for weights via 3d pointer
double*** allocate_memory_for_weights(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	double*** weights = new double** [number_of_hidden_layers + 1];

	// allocate memory for the first layer
	// number of features is provided by the dataset
	weights[0] = new double* [number_of_neurons_each_hidden_layer[0]];
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
		weights[0][n] = new double[number_of_features];

	// allocate memory for each subsequent layer
	for (int l = 1; l < number_of_hidden_layers; l++)
	{
		weights[l] = new double* [number_of_neurons_each_hidden_layer[l]];

		// number of features of given layer l is the number of neurons in the previous layer (l - 1)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			weights[l][n] = new double[number_of_neurons_each_hidden_layer[l - 1]];
	}

	// allocate memory for output layer with only one neuron
	// the number_of_hidden_layers is equal to the index of the last/output layer pointer
	// number of features is the number of neurons in the last hidden layer
	weights[number_of_hidden_layers] = new double*;
	weights[number_of_hidden_layers][0] = new double[number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]];

	return weights;
}

// allocate 2d array for biases
double** allocate_memory_for_biases(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers)
{
	double** biases = new double* [number_of_hidden_layers + 1];

	// allocate bias values for each hidden layer
	for (int l = 0; l < number_of_hidden_layers; l++)
		biases[l] = new double[number_of_neurons_each_hidden_layer[l]];

	// allocate memory for last layer/output layer with one neuron
	// number_of_hidden_layers is the index of the output layer
	biases[number_of_hidden_layers] = new double;

	return biases;
}

// allocate 2d array for training samples via 2d pointer
double** allocate_memory_for_2D_array(int number_of_rows, int number_of_columns)
{
	double** training_samples = new double* [number_of_rows];
	for (int i = 0; i < number_of_rows; i++)
		training_samples[i] = new double[number_of_columns];

	return training_samples;
}

// deallocate the memory in the provided weights pointer
void deallocate_memory_for_weights(double*** weights, const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers)
{
	if (weights != nullptr)
	{
		for (int l = 0; l < number_of_hidden_layers; l++)
		{
			for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
				delete[] weights[l][n];
			delete[] weights[l];
		}

		// deallocate output layer weights
		delete[] weights[number_of_hidden_layers][0];
		delete[] weights[number_of_hidden_layers];

		delete[] weights;
	}

	weights = nullptr;
}

// deallocate memory in provided bias pointer
void deallocate_memory_for_biases(double** biases, int number_of_hidden_layers)
{
	if (biases != nullptr)
	{
		for (int l = 0; l < number_of_hidden_layers; l++)
			delete[] biases[l];

		// deallocate output layer bias
		delete[] biases[number_of_hidden_layers];

		delete[] biases;
	}
}

// deallocate memory for a const pointer 2d array
void deallocate_memory_for_2D_array(double** const twoD_array, int number_of_rows)
{
	if (twoD_array != nullptr)
	{
		for (int t = 0; t < number_of_rows; t++)
			delete[] twoD_array[t];

		delete[] twoD_array;
	}
}

// deallocate memory for a nonconst pointer 2d array
void deallocate_memory_for_2D_array(double** const & twoD_array, int number_of_rows)
{
	if (twoD_array != nullptr)
	{
		for (int t = 0; t < number_of_rows; t++)
			delete[] twoD_array[t];

		delete[] twoD_array;
	}
}

// deallocate memory for a nonconst pointer 2d array
void deallocate_memory_for_2D_array(double**& twoD_array, int number_of_rows)
{
	if (twoD_array != nullptr)
	{
		for (int t = 0; t < number_of_rows; t++)
			delete[] twoD_array[t];

		delete[] twoD_array;
	}

	twoD_array = nullptr;
}