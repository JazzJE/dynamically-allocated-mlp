#include "MenuFunctions.h"
#include "TrainingLogAndList.h"
#include "NeuralNetwork.h"
#include "StatisticsFunctions.h"
#include "Constants.h"

MenuOptions get_option()
{
	char option;
	std::cout << "Option Menu:"
		<< "\n\t1. Train neural network on entire data set"
		<< "\n\t2. Test different parameters using k-fold training"
		<< "\n\t3. Save the best neural network state found from entire data set"
		<< "\n\t4. Randomize order of training samples"
		<< "\n\t5. Predict a value using provided sample features"
		<< "\n\t6. Predict a value using a random sample"
		<< "\n\t7. Print logs of all training sessions performed in this current program"
		<< "\n\t8. Save the training logs within this current program"
		<< "\n\t9. Exit program (will not save the neural network state)"
		<< "\nPlease select an option: ";
	std::cin >> option;

	while (static_cast<MenuOptions>(option) < FIRST_OPTION || static_cast<MenuOptions>(option) > LAST_OPTION || std::cin.peek() != '\n')
	{
		std::cin.clear(); // clear error flags
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input

		std::cout << "[ERROR] Please enter a valid input (" << static_cast<char>(MenuOptions::FIRST_OPTION) << " - " << 
			static_cast<char>(MenuOptions::LAST_OPTION) << "): ";
		std::cin >> option;
	}

	return static_cast<MenuOptions>(option);
}

void generate_border_line()
{ std::cout << '\n' << std::setw(Constants::width) << std::right << "----------------------\n"; }


void save_neural_network_state(NeuralNetwork& neural_network, std::filesystem::path weights_and_biases_file_path, 
	std::filesystem::path means_and_vars_file_path, std::filesystem::path scales_and_shifts_file_path, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features, 
	int net_number_of_neurons_in_hidden_layers)
{
	std::cout << "\n\n\tUpdating the " << weights_and_biases_file_path << " file...";
	update_weights_and_biases_file(weights_and_biases_file_path, neural_network.get_network_weights(),
		neural_network.get_network_biases(), number_of_neurons_each_hidden_layer, number_of_hidden_layers,
		number_of_features);

	std::cout << "\n\tUpdating the " << means_and_vars_file_path << " file...";
	update_mv_or_ss_file(means_and_vars_file_path, neural_network.get_network_running_means(),
		neural_network.get_network_running_variances(), net_number_of_neurons_in_hidden_layers);

	std::cout << "\n\tUpdating the " << scales_and_shifts_file_path << " file...";
	update_mv_or_ss_file(scales_and_shifts_file_path, neural_network.get_network_scales(),
		neural_network.get_network_shifts(), net_number_of_neurons_in_hidden_layers);

	std::cout << "\n\n\tDone!\n";
}

void all_sample_train_network_option(NeuralNetwork& neural_network, TrainingLogList& log_list, double** all_features_normalized, double* log_transformed_target_values,
	int number_of_samples)
{
	// this boolean limits the max number of samples per batch to the all possible samples
	bool using_all_samples = true;

	// input the rates of the nn
	double new_learning_rate, new_regularization_rate;
	int new_patience, new_number_of_epochs;
	input_rates(new_learning_rate, new_regularization_rate, new_patience, new_number_of_epochs);

	generate_border_line();

	int new_batch_size;
	input_batch_size(new_batch_size, number_of_samples, using_all_samples);

	generate_border_line();

	neural_network.set_learning_rate(new_learning_rate);
	neural_network.set_regularization_rate(new_regularization_rate);
	neural_network.set_patience(new_patience);
	neural_network.set_number_of_epochs(new_number_of_epochs);
	neural_network.set_batch_size(new_batch_size);

	std::cout << "\n\tTraining your network on all samples...";
	neural_network.all_sample_train(log_list, all_features_normalized, log_transformed_target_values, number_of_samples);
	std::cout << "\n\n\tDone!\n";
}

void k_fold_train_network_option(NeuralNetwork& neural_network, TrainingLogList& log_list, double** training_features, bool* not_normalize,
	double* log_transformed_target_values, int number_of_samples)
{
	// this boolean limits the max number of samples per batch to the number of samples minus the amount in the cross-validation fold
	bool using_all_samples = false;

	// input rates
	double new_learning_rate, new_regularization_rate;
	int new_patience, new_number_of_epochs;
	input_rates(new_learning_rate, new_regularization_rate, new_patience, new_number_of_epochs);

	generate_border_line();

	int number_of_folds;
	input_number_of_folds(number_of_folds);

	generate_border_line();

	int new_batch_size;
	input_batch_size(new_batch_size, number_of_samples, using_all_samples, number_of_folds);

	generate_border_line();
	
	neural_network.set_learning_rate(new_learning_rate);
	neural_network.set_regularization_rate(new_regularization_rate);
	neural_network.set_patience(new_patience);
	neural_network.set_number_of_epochs(new_number_of_epochs);
	neural_network.set_batch_size(new_batch_size);

	std::cout << "\n\tTraining your network using k-fold training...";
	neural_network.k_fold_train(log_list, training_features, not_normalize, log_transformed_target_values, number_of_samples, 
		number_of_folds);
	std::cout << "\n\n\tDone!\n";
}

// have user input values for each feature, normalize those features, then output a result
void predict_with_provided_features_option(NeuralNetwork& neural_network, std::string* feature_names, std::string target_name,
	double* all_features_means, double* all_features_variances, bool* not_normalize, int number_of_features)
{
	double* new_features = input_new_features(feature_names, not_normalize, number_of_features);

	std::cout << "\n\tProvided these features: ";
	for (int f = 0; f < number_of_features; f++)
		std::cout << "\n\t\t" << feature_names[f] << " - " << new_features[f];

	double* normalized_features = calculate_normalized_features(new_features, not_normalize, number_of_features,
		all_features_means, all_features_variances);

	std::cout << "\n\n\tNormalized features: ";
	for (int f = 0; f < number_of_features; f++)
		std::cout << "\n\t\t" << feature_names[f] << " - " << normalized_features[f];

	double prediction = neural_network.calculate_prediction(normalized_features);
	std::cout << "\n\n\tPredicted log-transformed value of " << target_name << ": " << prediction
		<< "\n\tPredicted actual value of " << target_name << ": " << pow(Constants::euler_number, prediction) - 1 << "\n";

	delete[] new_features;
	delete[] normalized_features;
}

// pick a random sample's features and get its predicted values, as well as original actual values for comparison
void predict_with_random_features_option(NeuralNetwork& neural_network, std::string* feature_names, std::string target_name, 
	double** training_features, double** all_features_normalized, double* target_values, double* log_transformed_target_values,
	int* sample_numbers, int number_of_samples, int number_of_features)
{	
	int random_index = std::rand() % number_of_samples;
	std::cout << "\n\tProvided these features for sample #" << sample_numbers[random_index] << " within your dataset: ";
	for (int f = 0; f < number_of_features; f++)
		std::cout << "\n\t\t" << feature_names[f] << " - " << training_features[random_index][f];

	std::cout << "\n\n\tNormalized features: ";
	for (int f = 0; f < number_of_features; f++)
		std::cout << "\n\t\t" << feature_names[f] << " - " << all_features_normalized[random_index][f];

	std::cout << "\n\n\tActual log-transformed value of " << target_name << ": " << log_transformed_target_values[random_index]
		<< "\n\tActual value of " << target_name << ": " << target_values[random_index];

	double prediction = neural_network.calculate_prediction(all_features_normalized[random_index]);
	std::cout << "\n\n\tPredicted log-transformed value of " << target_name << ": " << prediction
		<< "\n\tPredicted actual value of " << target_name << ": " << pow(Constants::euler_number, prediction) - 1 << "\n";
}

// create dynamically allocated array of all the features that use will input
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

// have the user input a new batch size, which will regenerate the neural network's structure based on the new value
void input_batch_size(int& new_batch_size, int number_of_samples, bool using_all_samples, int number_of_folds)
{
	// if using all samples in the next training, then max batch size is all of the samples
	// else, then all samples except the size of the cross-validation fold
	int max_batch_size = using_all_samples ? number_of_samples : number_of_samples / number_of_folds * (number_of_folds - 1);

	std::cout << "\n\tPlease enter an integer value for the new ***batch size***; it must be less than or equal to " 
		<< max_batch_size << ": ";

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

		if (new_batch_size <= 0 || new_batch_size > max_batch_size)
		{
			std::cout << "\t[ERROR] Please do not enter  numbers less than or equal to 0, or numbers greater than " 
				<< max_batch_size << " for the new ***batch size***: ";
			continue;
		}
		
		break;
	}
}

// helper method to input new parameters
void input_rates(double& new_learning_rate, double& new_regularization_rate, int& new_patience, int& new_number_of_epochs)
{
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

	new_learning_rate = get_a_positive_double("\n\tPlease enter a double value for the new ***learning rate***: ",
		"\t[ERROR] Please do not enter characters or negative numbers for the new*** learning rate***: ");

	generate_border_line();

	new_regularization_rate = get_a_positive_double("\n\tPlease enter a double value for the new ***regularization rate***: ",
		"\t[ERROR] Please do not enter characters or negative numbers for the new*** regularization rate***: ");

	generate_border_line();

	new_patience = get_a_positive_integer("\n\tPlease enter an integer value for the ***patience*** of five-fold training: ",
		"\t[ERROR] Please do not enter characters, double values, or numbers less than or equal to 0 for the new ***patience***: ");

	generate_border_line();

	new_number_of_epochs = get_a_positive_integer("\n\tPlease enter an integer value for the ***number of epochs before stopping training***: ",
		"\t[ERROR] Please do not enter characters, double values, or numbers less than or equal to 0 for the new ***number of epochs***: ");
}

// there should be a limit to the number of folds here
void input_number_of_folds(int& number_of_folds)
{
	std::string prompt_message = "\n\tPlease enter an integer value for the ***number of folds*** from " 
		+ std::to_string(Constants::min_number_of_folds) + " to " + std::to_string(Constants::max_number_of_folds) + ": ";
	std::string error_message = "\t[ERROR] Please do not enter characters, double values, or numbers less than " 
		+ std::to_string(Constants::min_number_of_folds) + " or greater than " + std::to_string(Constants::max_number_of_folds) + ": ";

	std::cout << prompt_message;
	while (true)
	{
		if (!(std::cin >> number_of_folds))
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

		// Validate value range and make sure it's a positive value
		if (number_of_folds < Constants::min_number_of_folds || number_of_folds > Constants::max_number_of_folds)
		{
			std::cout << error_message;
			continue;
		}

		break;
	}

	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get rid of the new line
}

void input_session_name(std::string& new_session_name)
{
	// ignore stuff if there's something in the buffer
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	std::unordered_set<char> invalid_file_characters = { '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', 
		';', '\'', '\"', ',', '<', '>', '?', '|', '~' };

	char invalid_char;
	std::cout << "\n\tPlease enter a name for this session (do not use invalid file name characters): ";
	getline(std::cin, new_session_name);

	// validation to check if the inputted file name is valid
	for (int i = 0; i < new_session_name.length(); i++)
		if (invalid_file_characters.find(new_session_name[i]) != invalid_file_characters.end())
		{
			std::cout << "\t[ERROR] Please enter a valid name for this session (no invalid file name characters): ";
			getline(std::cin, new_session_name);
			i = -1;
		}
}

// randomize the order of the training samples
void randomize_training_samples(double** training_features, double* target_values, double* log_transformed_target_values,
	int* sample_numbers, int number_of_samples)
{
	std::cout << "\n\tRandomizing samples' order " << Constants::number_of_random_shuffles << " times...";

	int random_index;
	double temp_double;
	double* temp_ptr;
	int temp_int;

	for (int s = 0; s < Constants::number_of_random_shuffles; s++)
		for (int current_index = number_of_samples - 1; current_index > 0; current_index--)
		{
			random_index = std::rand() % current_index;

			// swap where pointers are directed
			temp_ptr = training_features[random_index];
			training_features[random_index] = training_features[current_index];
			training_features[current_index] = temp_ptr;

			// swap the target values
			temp_double = target_values[random_index];
			target_values[random_index] = target_values[current_index];
			target_values[current_index] = temp_double;

			// swap the log transformed target values
			temp_double = log_transformed_target_values[random_index];
			log_transformed_target_values[random_index] = log_transformed_target_values[current_index];
			log_transformed_target_values[current_index] = temp_double;

			// swap the sample numbers
			temp_int = sample_numbers[random_index];
			sample_numbers[random_index] = sample_numbers[current_index];
			sample_numbers[current_index] = temp_int;
		}
	
	std::cout << "\n\n\tDone!\n";
}

// update the weights and biases
void update_weights_and_biases_file(std::filesystem::path weights_and_biases_file_path, double*** weights, double** biases,
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
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
		{
			for (int w = 0; w < number_of_neurons_each_hidden_layer[l - 1]; w++)
				weights_and_biases_file << weights[l][n][w] << ", ";
			weights_and_biases_file << biases[l][n] << "\n";
		}
	
	// output layer
	for (int w = 0; w < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; w++)
		weights_and_biases_file << weights[number_of_hidden_layers][0][w] << ", ";
	weights_and_biases_file << *(biases[number_of_hidden_layers]) << "\n";

	// close the file
	weights_and_biases_file.close();
}

// update the means and variances OR the scales and shifts file with the current local configs within the program
void update_mv_or_ss_file(std::filesystem::path mv_or_ss_file_path, double* means_or_scales, double* variances_or_shifts, int net_number_of_neurons_in_hidden_layers)
{
	std::fstream mv_or_ss_file(mv_or_ss_file_path, std::ios::out | std::ios::trunc);

	// means and variances & scales and shifts will always only have two features, lest the program doesn't work
	for (int n = 0; n < net_number_of_neurons_in_hidden_layers; n++)
		mv_or_ss_file << means_or_scales[n] << "," << variances_or_shifts[n] << "\n";

	mv_or_ss_file.close();
}