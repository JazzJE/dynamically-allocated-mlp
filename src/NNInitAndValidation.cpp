#include "NeuralNetwork.h"

// validating the neural network files
void NeuralNetwork::validate_neural_network_files(std::string weights_and_biases_file_path, std::string means_and_vars_file_path,
	std::string scales_and_shifts_file_path)
{
	std::fstream weights_and_biases_file(weights_and_biases_file_path, std::ios::in);

	// if the weight file was not opened (therefore doesn't exist), initialize a new one using He initialization
	if (!weights_and_biases_file)
	{
		std::cout << "\n\t" << weights_and_biases_file_path << " not found; creating new one...\n";
		weights_and_biases_file.close();

		generate_weights_and_biases_file(weights_and_biases_file_path);

		weights_and_biases_file.open(weights_and_biases_file_path, std::ios::in);
	}

	// ensure that the weights and biases file has the appropriate weights and biases for each layer, else prompt the user 
	// to generate
	validate_weights_and_biases_file(weights_and_biases_file, weights_and_biases_file_path);

	weights_and_biases_file.close();


	// file will store the running means and running variances of each neuron
	std::fstream means_and_vars_file(means_and_vars_file_path, std::ios::in);

	// if the running means and running variances file doesn't exist, generate a new one
	if (!means_and_vars_file)
	{
		std::cout << "\n\t" << means_and_vars_file_path << " not found; creating new one...\n";
		means_and_vars_file.close();

		generate_means_and_vars_file(means_and_vars_file_path);

		means_and_vars_file.open(means_and_vars_file_path, std::ios::in);
	}

	// validate each line has only 2 fields and no strings
	validate_mv_or_ss_file(means_and_vars_file, means_and_vars_file_path, &NeuralNetwork::generate_means_and_vars_file);

	means_and_vars_file.close();


	// file will store affine transformation parameters for each neuron
	std::fstream scales_and_shifts_file(scales_and_shifts_file_path, std::ios::in);

	// if the running means and running variances file doesn't exist, generate a new one
	if (!scales_and_shifts_file)
	{
		std::cout << "\n\t" << scales_and_shifts_file_path << " not found; creating new one...\n\n";
		scales_and_shifts_file.close();

		generate_scales_and_shifts_file(scales_and_shifts_file_path);

		scales_and_shifts_file.open(scales_and_shifts_file_path, std::ios::in);
	}

	// validate each line has only 2 fields and no strings
	validate_mv_or_ss_file(scales_and_shifts_file, scales_and_shifts_file_path, &NeuralNetwork::generate_scales_and_shifts_file);

	scales_and_shifts_file.close();
}

// generating weights and biases file methods

	// generate weight file if not already made
void NeuralNetwork::generate_weights_and_biases_file(std::string weights_and_biases_file_path)
{
	std::fstream weights_and_biases_file(weights_and_biases_file_path, std::ios::out | std::ios::trunc);

	generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_features, number_of_neurons_each_hidden_layer[0]);

	for (int l = 1; l < number_of_hidden_layers; l++)
		generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1],
			number_of_neurons_each_hidden_layer[l]);

	generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1);

	weights_and_biases_file.close();
}

// helper function to generate the weights and biases for a given layer with a set number of neurons
void NeuralNetwork::generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons)
{
	// generate a random seed for numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// use He initialization for the weights provided the number of input features into the first layer
	double stddev = sqrt(2.0 / number_of_features);
	std::normal_distribution<double> dist(0.0, stddev);
	// for each neuron
	for (int n = 0; n < number_of_neurons; n++)
	{
		// insert a number of weights equal to the number of features coming in from the previous layer
		for (int w = 0; w < number_of_features; w++)
			weights_and_biases_file << dist(gen) << ",";

		// insert an initial bias value of 0 and then end the current line for this neuron within this layer
		weights_and_biases_file << 0 << '\n';
	}
}

// methods to validate the weights and biases file

	// validate the weights and biases file is valid, and if not, prompt user to make a new one
void NeuralNetwork::validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_path)
{
	// the find_error_weights_and_biases_file function will return an integer value of the line in which the error was found
	// if no error is found, the error returned is -1
	int line_error = find_error_weights_and_biases_file(weights_and_biases_file);
	if (line_error != -1)
	{
		char option;

		// ask user if they would like to reset their neural network, and if not, then end the program
		// this is so they can update the configuration to their weights and biases folder if they accidentally interacted with it,
		// as the files should not be changed
		std::cerr << "\n[ERROR] " << weights_and_biases_file_path << " contains one of the following errors"
			<< "\n\n\t1. String value was detected"
			<< "\n\t2. There are not enough weights in the below-provided line"
			<< "\n\t3. There are too many or too few lines within the file"
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << weights_and_biases_file_path << " ***"
			<< "\n\nWould you like to generate a new " << weights_and_biases_file_path << "?"
			<< "\n\nPlease select yes or no (Y / N): ";
		std::cin >> option;

		while (option != 'Y' && option != 'N')
		{
			std::cerr << "[ERROR] Invalid input. Please enter only yes or no (Y/N): ";
			std::cin >> option;
		}

		if (option == 'Y')
		{
			weights_and_biases_file.close();

			std::cout << "\nGenerating new " << weights_and_biases_file_path << " file...\n\n";
			generate_weights_and_biases_file(weights_and_biases_file_path);

			weights_and_biases_file.open(weights_and_biases_file_path, std::ios::in);

		}
		else
		{
			std::cerr << "\nEnding program; please fix the error before continue with this program...\n";
			exit(0);
		}
	}
}

// for each neuron line n in a given l layer, see if there is an equivalent number of features/weights for the number of features 
// plus one bias value
// the function will return the line the error was found so it can be altered easily; if not found, return -1
int NeuralNetwork::find_error_weights_and_biases_file(std::fstream& weights_and_biases_file)
{
	// this will store the line number which the error was found so user may alter it accordingly
	int line_error = 0;

	double temp_double;
	std::string line, value;
	std::stringstream ss, converter;

	// first layer
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
	{
		line_error++;
		getline(weights_and_biases_file, line);

		ss.clear();
		ss.str(line);

		int field_count = 0;

		while (getline(ss, value, ','))
		{
			// try turning each value parsed into a double, and if it fails, that means it's a string value
			converter.clear();
			converter.str(value);
			converter >> temp_double;
			if (converter.fail() || !converter.eof())
				return line_error;

			field_count++;
		}

		if (field_count != number_of_features + 1) return line_error;
	}

	// each subsequent layer
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
		{
			line_error++;
			getline(weights_and_biases_file, line);

			ss.clear();
			ss.str(line);

			int field_count = 0;

			while (getline(ss, value, ','))
			{
				converter.clear();
				converter.str(value);
				converter >> temp_double;
				if (converter.fail() || !converter.eof())
					return line_error;

				field_count++;
			}

			if (field_count != number_of_neurons_each_hidden_layer[l - 1] + 1) return line_error;
		}

	// last output layer
	line_error++;
	getline(weights_and_biases_file, line);

	ss.clear();
	ss.str(line);

	int field_count = 0;

	while (getline(ss, value, ','))
	{
		converter.clear();
		converter.str(value);
		converter >> temp_double;
		if (converter.fail() || !converter.eof())
			return line_error;

		field_count++;
	}

	if (field_count != number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1] + 1) return line_error;

	// check if there are too many lines
	// the last line of the file is be an empty line due to how the weight and biases are generated
	// if the end of the file is not reached or the empty line is not empty, then the file is invalid as there are extra lines
	line_error++;
	getline(weights_and_biases_file, line);
	if (!weights_and_biases_file.eof() || line != "")
		return line_error;

	// reset to start of the file, and ne error was found
	weights_and_biases_file.clear();
	weights_and_biases_file.seekg(0);
	return -1;
}

// running means and running variances for each neuron in normalization

	// generate a file that will store the running means and running variances of each neuron
	// each mean will be initialized to 0, and each variance will be initialize to 1
void NeuralNetwork::generate_means_and_vars_file(std::string means_and_vars_file_path)
{
	std::fstream means_and_vars_file(means_and_vars_file_path, std::ios::out | std::ios::trunc);

	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
		means_and_vars_file << 0 << "," << 1 << "\n";

	means_and_vars_file.close();
}

	// generate a file that will store the shifts and scales of each neuron
	// each mean will be initialized to 0, and each variance will be initialize to 1
void NeuralNetwork::generate_scales_and_shifts_file(std::string scales_and_shifts_file_path)
{
	std::fstream scales_and_shifts_file(scales_and_shifts_file_path, std::ios::out | std::ios::trunc);

	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
		scales_and_shifts_file << 1 << "," << 0 << "\n";

	scales_and_shifts_file.close();
}

// verify the means and var file, and return -1 if no error was detected
void NeuralNetwork::validate_mv_or_ss_file(std::fstream& mv_or_ss_file, std::string mv_or_ss_file_path, 
	void (NeuralNetwork::*generate_mv_or_ss_file)(std::string))
{
	int line_error = find_error_mv_or_ss_file(mv_or_ss_file);
	if (line_error != -1)
	{
		char option;

		std::cout << "\n[ERROR] One of the following errors is within " << mv_or_ss_file_path
			<< "\n\n\t1. String value was detected"
			<< "\n\t2. There are greater than two fields within the below-provided line"
			<< "\n\t3. There are too many or too few lines within the file"
			<< "\n\n\t***The error was found on line " << line_error << " in " << mv_or_ss_file_path << " ***"
			<< "\n\nWould you like to generate a new " << mv_or_ss_file_path << " file?: ";
		std::cin >> option;

		while (option != 'Y' && option != 'N')
		{
			std::cout << "\n[ERROR] Please enter only enter (Y/N): ";
			std::cin >> option;
		}

		if (option == 'Y')
		{
			mv_or_ss_file.close();

			std::cout << "\nGenerating new " << mv_or_ss_file_path << "...\n\n";
			(this->*generate_mv_or_ss_file)(mv_or_ss_file_path);

			mv_or_ss_file.open(mv_or_ss_file_path, std::ios::in);
		}
		else
		{
			std::cout << "\nExiting; please fix the error before interacting with this program...\n";
			exit(0);
		}
	}
}

// find if there is an error in ema file, where if there isn't then return -1
int NeuralNetwork::find_error_mv_or_ss_file(std::fstream& mv_or_ss_file)
{
	int field_count, line_error = 0;
	double temp_double;
	std::string line, value;
	std::stringstream ss, converter;

	// for every neuron in the nn
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		line_error++;

		field_count = 0;
		getline(mv_or_ss_file, line);

		ss.clear();
		ss.str(line);

		while (getline(ss, value, ','))
		{
			// check if the value is a string
			converter.clear();
			converter.str(value);
			converter >> temp_double;
			if (converter.fail() || !converter.eof())
				return line_error;

			// count the number of columns there are
			field_count++;
		}

		// there should only be two values in a line: the mean/scale first, the variance/shift second
		if (field_count != 2) return line_error;
	}


	// the last line of the file will be a empty line due to how means/scales and variances/shifts are generated, so make sure to take it in
	// if the empty line is not empty or the file hasn't reached its end, then that means there is still extra lines and thus the
	// file is invalid
	line_error++;
	getline(mv_or_ss_file, line);
	if (!mv_or_ss_file.eof() || line != "")
		return line_error;

	// return -1 if there was no error
	mv_or_ss_file.clear();
	mv_or_ss_file.seekg(0);
	return -1;
}

// parsing methods

	// parse the weights_and_biases file into an array
void NeuralNetwork::parse_weights_and_biases_file(std::string weights_and_biases_file_path)
{
	std::fstream weights_and_biases_file(weights_and_biases_file_path, std::ios::in);

	parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_features,
		number_of_neurons_each_hidden_layer[0], 0);

	for (int l = 1; l < number_of_hidden_layers; l++)
		parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1],
			number_of_neurons_each_hidden_layer[l], l);

	// parse last layer into the weights and biases array
	// remember that the number of hidden layers is equal to the index of the last layer
	parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1],
		1, number_of_hidden_layers);

	weights_and_biases_file.close();
}

// helper function to parse each neuron of a given layer
void NeuralNetwork::parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons,
	int layer_index)
{
	std::string line, value;
	std::stringstream ss;

	for (int n = 0; n < number_of_neurons; n++)
	{
		getline(weights_and_biases_file, line);

		ss.clear();
		ss.str(line);

		for (int w = 0; w < number_of_features; w++)
		{
			getline(ss, value, ',');
			network_weights[layer_index][n][w] = std::stod(value);
		}

		// last value will be bias value
		getline(ss, value, '\n');
		network_biases[layer_index][n] = std::stod(value);
	}
}

// parse the running means and variances OR shifts and scales file
void NeuralNetwork::parse_mv_or_ss_file(std::string mv_or_ss_file_path, double* means_or_scales, double* variances_or_shifts)
{
	std::fstream mv_or_ss_file(mv_or_ss_file_path, std::ios::in);

	std::string line, value;
	std::stringstream ss;
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
	{
		getline(mv_or_ss_file, line);

		ss.clear();
		ss.str(line);

		getline(ss, value, ',');
		means_or_scales[n] = std::stod(value);

		getline(ss, value, '\n');
		variances_or_shifts[n] = std::stod(value);
	}

	mv_or_ss_file.close();
}