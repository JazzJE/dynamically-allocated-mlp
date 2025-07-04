#include "MenuFunctions.h"
void generate_border_line()
{
	std::cout << '\n' << std::setw(32) << std::right << "----------------------\n";
}

double* input_new_features(std::string* feature_names, bool* not_normalize, int number_of_features)
{
	double* input_features = new double[number_of_features];

	for (int f = 0; f < number_of_features; f++)
	{
		if (not_normalize[f])
		{
			std::cout << "\n\tEnter the value of the " << feature_names[f] << " as only a 0 or 1 (0 for false, 1 for true): ";

			while (!(std::cin >> input_features[f]) || (input_features[f] != 0 && input_features[f] != 1))
			{
				std::cin.clear(); // clear error flags
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input
				std::cout << "\t[ERROR] Please enter only a boolean value for " << feature_names[f] << "(0 for false, 1 for true): ";
			}

			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore any extra bad key buffer input

			continue;
		}

		std::cout << "\n\tEnter the value of the " << feature_names[f] << ": ";
		while (!(std::cin >> input_features[f]))
		{
			std::cin.clear(); // clear error flags
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input
			std::cout << "\t[ERROR] Please enter only a double value for " << feature_names[f] << ": ";
		}

		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore any extra bad key buffer input
	}

	return input_features;
}

// regenerate the neural network with a new batch size, as the batch size cannot be dynamically changed due to const qualifiers
void update_batch_size_and_regen_new_neural_network(NeuralNetwork*& neural_network, const int* number_of_neurons_each_hidden_layer,
	int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers, int number_of_features, 
	std::string weights_and_biases_file_path, std::string means_and_vars_file_path, std::string scales_and_shifts_file_path, 
	int number_of_samples)
{
	double learning_rate = 0, regularization_rate = 0;
	int patience = 0, prompt_epoch = 0;

	// this is for if the pointer isn't pointing to nullptr, meaning that the nn has already been initialized
	if (neural_network)
	{
		learning_rate = neural_network->get_learning_rate();
		regularization_rate = neural_network->get_regularization_rate();
		patience = neural_network->get_patience();
		prompt_epoch = neural_network->get_prompt_epoch();
	}

	int new_batch_size;

	std::cout << "\n\tPlease enter an integer value for the new ***batch size***; it must be less than or equal to " 
		<< number_of_samples / 5 * 4 << ": ";
	// the last condition is considering when the batch size is too large for a current training fold 
	// (i.e., if there are 5001 training samples, then you can only use 4000 for when there are 1001 in the last cv training fold)
	while (true)
	{
		if (!(std::cin >> new_batch_size))
		{
			std::cin.clear(); // clear error flags
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input
			std::cout << "\t[ERROR] Please do not enter characters for the new ***batch size***: ";
			continue;
		}

		// check for remaining characters in the input buffer
		if (std::cin.peek() != '\n')
		{
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "\t[ERROR] Please do not enter double values for the new ***batch size***: ";
			continue;
		}

		if (new_batch_size <= 0 || new_batch_size > number_of_samples / 5 * 4)
		{
			std::cout << "\t[ERROR] Please do not enter  numbers less than or equal to 0, or numbers greater than " 
				<< number_of_samples / 5 * 4 << " for the new ***batch size***: ";
			continue;
		}

		break;
	}

	delete neural_network;
	neural_network = new NeuralNetwork(number_of_neurons_each_hidden_layer, net_number_of_neurons_in_hidden_layers, number_of_hidden_layers,
		number_of_features, new_batch_size, learning_rate, regularization_rate, patience, prompt_epoch,
		weights_and_biases_file_path, means_and_vars_file_path, scales_and_shifts_file_path);
}

// input customizable parameters within the nn
void input_parameter_rates(NeuralNetwork* neural_network)
{
	// lambda for getting an integer value from the user
	auto get_a_positive_integer = [](std::string prompt_message, std::string error_message) -> int
		{
			int temp_int;
			std::cout << prompt_message;
			while (true)
			{
				if (!(std::cin >> temp_int))
				{
					std::cin.clear();
					std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get rid of bad input
					std::cout << error_message;
					continue;
				}

				// check for remaining characters in the input buffer
				if (std::cin.peek() != '\n') 
				{
					std::cin.clear();
					std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
					std::cout << error_message;
					continue;
				}

				// Validate value range
				if (temp_int <= 0) 
				{
					std::cout << error_message;
					continue;
				}

				break;
			}
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get rid of the new line
			return temp_int;
		};

	auto get_a_positive_double = [](std::string prompt_message, std::string error_message) -> double
		{
			double temp_double;
			std::cout << prompt_message;
			while (true)
			{
				if (!(std::cin >> temp_double))
				{
					std::cin.clear();
					std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get rid of bad input
					std::cout << error_message;
					continue;
				}

				// Validate value range
				if (temp_double < 0) {
					std::cout << error_message;
					continue;
				}

				break;
			}
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get rid of the new line
			return temp_double;
		};

	double new_learning_rate = get_a_positive_double("\n\tPlease enter a double value for the new ***learning rate***: ",
		"\t[ERROR] Please do not enter characters or negative numbers for the new*** learning rate***: ");
	neural_network->set_learning_rate(new_learning_rate);

	generate_border_line();

	double new_regularization_rate = get_a_positive_double("\n\tPlease enter a double value for the new ***regularization rate***: ",
		"\t[ERROR] Please do not enter characters or negative numbers for the new*** regularization rate***: ");
	neural_network->set_regularization_rate(new_regularization_rate);

	generate_border_line();

	int new_patience = get_a_positive_integer("\n\tPlease enter an integer value for the ***patience*** of five-fold training: ",
		"\t[ERROR] Please do not enter characters, double values, or numbers less than or equal to 0 for the new ***patience***: ");
	neural_network->set_patience(new_patience);

	generate_border_line();

	int new_prompt_epoch = get_a_positive_integer("\n\tPlease enter an integer value for the ***number of epochs before prompting to stop training for a fold***: ",
		"\t[ERROR] Please do not enter characters, double values, or numbers less than or equal to 0 for the new ***prompt epoch***: ");
	neural_network->set_prompt_epoch(new_prompt_epoch);

}

// update the weights and biases
void update_weights_and_biases_file(std::string weights_and_biases_file_path, double*** weights, double** biases,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	// clear the file
	std::fstream weights_and_biases_file(weights_and_biases_file_path, std::ios::out | std::ios::trunc);

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
void update_mv_or_ss_file(std::string mv_or_ss_file_path, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons_in_hidden_layers)
{
	std::fstream mv_or_ss_file(mv_or_ss_file_path, std::ios::out | std::ios::trunc);

	// means and variances & scales and shifts will always only have two features, lest the program doesn't work
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
		mv_or_ss_file << means_or_scales[n] << "," << variances_or_shifts[n] << "\n";

	mv_or_ss_file.close();
}