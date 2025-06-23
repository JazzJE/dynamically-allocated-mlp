#include "MenuFunctions.h"
void generate_border_line()
{
	std::cout << '\n' << std::setw(32) << std::right << "----------------------\n";
}

void input_parameter_rates(double& learning_rate, double& regularization_rate)
{
	// validate input for the learning rate
	while (true)
	{
		std::cout << "\n\tPlease enter a double value for the new ***learning rate***: ";
		std::cin >> learning_rate;

		// if bad input detected, then warn user
		if (std::cin.fail())
		{
			std::cin.clear(); // clear error flags
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input
			std::cout << "\t[ERROR] Invalid input. Please do not enter characters or other anomalous characters for the new ***learning rate***.\n";
		}

		// if valid, then continue
		else break;
	}

	generate_border_line();

	// validate input for the regularization rate
	while (true)
	{
		std::cout << "\n\tPlease enter a double value for the new ***regularization rate***: ";
		std::cin >> regularization_rate;

		if (std::cin.fail())
		{
			std::cin.clear(); // clear error flags
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input
			std::cout << "\t[ERROR] Invalid input. Please do not enter characters or other anomalous characters for the new ***regularization rate***.\n";
		}

		else break;
	}
}

// update the weights and biases
void update_weights_and_biases_file(std::string weights_and_biases_file_name, double*** weights, double** biases,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	// clear the file
	std::fstream weights_and_biases_file(weights_and_biases_file_name, std::ios::out | std::ios::trunc);

	// first layer
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
	{
		for (int w = 0; w < number_of_features; w++)
			weights_and_biases_file << weights[0][n][w] << ", ";
		weights_and_biases_file << biases[0][n] << "\n";
	}

	// rest of the layers
	for (int l = 1; l < number_of_hidden_layers; l++)
	{
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
		{
			for (int w = 0; w < number_of_neurons_each_hidden_layer[l - 1]; w++)
				weights_and_biases_file << weights[l][n][w] << ", ";
			weights_and_biases_file << biases[l][n] << "\n";
		}
	}
	
	// output layer
	for (int w = 0; w < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; w++)
		weights_and_biases_file << weights[number_of_hidden_layers][0][w] << ", ";
	weights_and_biases_file << *(biases[number_of_hidden_layers]) << "\n";

	// close the file
	weights_and_biases_file.close();
}

// update the means and variances OR the scales and shifts file with the current local configs within the program
void update_mv_or_ss_file(std::string mv_or_ss_file_name, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons_in_hidden_layers)
{
	std::fstream mv_or_ss_file(mv_or_ss_file_name, std::ios::out | std::ios::trunc);

	// means and variances & scales and shifts will always only have two features, lest the program doesn't work
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
		mv_or_ss_file << means_or_scales[n] << "," << variances_or_shifts[n] << "\n";

	mv_or_ss_file.close();
}