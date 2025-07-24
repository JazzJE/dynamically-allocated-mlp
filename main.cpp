/*
	Programmer name: Jedrick Espiritu
	Program name: vectorless_neural_network.cpp
	Date: May 4, 2025

	**KEY THINGS TO NOTE WHEN INTERACTING WITH THIS PROGRAM**

	- This program is hard-written with a single neuron for output (meaning that you should only be predicting one value), and is
	more specifically meant to be used as a linear regression rather than logistic regression model (predict a value like prices rather
	than classify something like if a dog is in the image)

	- In the csv dataset file that you will parse, make sure...
		
		1. _____The first row consists of the names of each of the features and target values_____
			
			a. ***** IF YOU WOULD LIKE A COLUMN TO NOT BE NORMALIZED DURING TRAINING (I.E. ONE HOT FEATURE ENCODING) ***** 
			then before the name of the column, identify the column to not be normalized by placing a tilde (~) 
			before the name (i.e., let's say you have a column of feature values with a feature name of ">1 HOUR AWAY FROM OCEAN," where 
			this feature can only be 0 or 1 to represent if the provided sample is indeed greater than one hour away from the ocean.
			To ensure that the program is does not normalize these discrete values into continuous values, rename this feature's column
			name to "~>1 HOUR AWAY FROM OCEAN"; examine the provided dataset for further clarification). 
			
			***** TL:DR If this is not done, the neural network will normalize the columns' discrete values during
			training as if the values were continuous, causing irregular/poor performance *****

		2. _____The last column consists of the target values that you want to predict_____
		
			a. If you want to predict house prices, then every column except the last will be the house features, 
			and then the last column will be the actual house prices
		
		3. _____Every field only consists of number values_____
			
			a. Do NOT have strings/words/letters in the dataset past the FIRST row (as aforementioned, the first row will be used to
			give the features columns names; this includes binary features of "True" and "False" for consistency's sake in only
			using number values as features for this program)

*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <limits>
#include "InitializationFunctions.h"
#include "MemoryFunctions.h"
#include "TrainingLogAndList.h"
#include "MenuFunctions.h"
#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "StatisticsFunctions.h"

int main()
{
	



	// !!! REMINDER that the last column is the column of values you want to predict !!!
		// for example, if you have features of a house and want to predict price, then make the house prices the last column of the csv
	// original data set used: https://www.kaggle.com/datasets/camnugent/california-housing-prices


	// !!! NOTE AGAIN that this program is hard-written with a single neuron for output !!!
	// this is the order and number of neurons you want in each hidden layer
	// in the below example...
		// the first hidden layer will have 128 neurons
		// the second hidden layer will have 64 neurons
		// the third hidden layer will have 32 neurons
		// the fourth/output layer (automatically/implicitly created) will have 1 neuron, predicting the value
	
	const int number_of_neurons_each_hidden_layer[] = { 128, 64, 32 };




	// PAST THIS POINT IS ALL THE HARD CODE; REFER TO THE ABOVE number_of_neurons_each_hidden_layer ARRAY TO CHANGE THE STRUCTURE OF THE MLP

	// all the names of the corresponding files we will parse into this program
	const std::string nn_saved_state_directory_name = "nn_saved_state";
	const std::string training_logs_directory_name = "training_logs";
	const std::string dataset_file_name = "dataset.csv";
	const std::string weights_and_biases_file_name = "weights_and_biases.csv";
	const std::string means_and_vars_file_name = "means_and_vars.csv";
	const std::string scales_and_shifts_file_name = "scales_and_shifts.csv";

	// all the system file paths from root directory
	const std::filesystem::path project_root = PROJECT_ROOT;
	const std::filesystem::path nn_saved_state_file_path = project_root / nn_saved_state_directory_name;
	const std::filesystem::path training_logs_file_path = project_root / training_logs_directory_name;
	const std::filesystem::path dataset_file_path = project_root / dataset_file_name;
	const std::filesystem::path weights_and_biases_file_path = nn_saved_state_file_path / weights_and_biases_file_name;
	const std::filesystem::path means_and_vars_file_path = nn_saved_state_file_path / means_and_vars_file_name;
	const std::filesystem::path scales_and_shifts_file_path = nn_saved_state_file_path / scales_and_shifts_file_name;

	clear_screen();
	std::cout << "The program may take a couple of seconds to load...\n" << std::fixed;

	int number_of_hidden_layers = sizeof(number_of_neurons_each_hidden_layer) / sizeof(int);
	
	// validate the number_of_neurons_each_hidden_layer array
	for (int i = 0; i < number_of_hidden_layers; i++)
		if (number_of_neurons_each_hidden_layer[i] <= 0)
		{
			std::cerr << "\n[ERROR] Before using this program, please ensure that there are no zero or negative values in the "
				<< "\"number_of_neurons_each_hidden_layer\" array";
			exit(0);
		}

	// when choosing batch samples and generating a new weights/biases folder, we will use the rand method
	srand(time(0));

	// EXCLUDING the singular neuron in the "output layer"; this will be used for assigning means and variances & scales and shifts
	int net_number_of_neurons_in_hidden_layers = 0;
	for (int l = 0; l < number_of_hidden_layers; l++)
		net_number_of_neurons_in_hidden_layers += number_of_neurons_each_hidden_layer[l];

	std::cout << "\n\tParsing data set file...\n";
	std::fstream dataset_file(dataset_file_path, std::ios::in);
	if (!dataset_file) 
	{
		std::cerr << "\n[ERROR] The dataset could not be found within the project; please edit the \"dataset_file_name\" variable "
			<< "to the dataset's name, or otherwise include the dataset within the project." << std::endl;
		exit(0);
	}

	// make the program a little more autonomous so user doesn't need to be confused entering these values
	const int number_of_samples = count_number_of_samples(dataset_file_path);
	const int number_of_features = count_number_of_features(dataset_file_path);

	// given the number of features, ensure that each row has the same number of columns
	validate_dataset_file(dataset_file, dataset_file_name, number_of_features);

	double** training_features = allocate_memory_for_2D_array(number_of_samples, number_of_features);
	double* target_values = new double[number_of_samples];
	std::string* feature_names = new std::string[number_of_features]; // get the names of the columns during parsing
	std::string target_name;
	parse_dataset_file(dataset_file, training_features, target_values, feature_names, target_name, number_of_features, number_of_samples);

	// calculate the log transformed values of the target values
	double* log_transformed_target_values = calculate_log_transformed_target_values(target_values, number_of_samples);

	// from the feature names, identify which features should not be normalized if their column names start with a ~ sign
	// the calculating means, variances, and features helper methods will ignore these features during their calculations
	bool* ignore_normalization = identify_ignore_normalization_feature_columns(feature_names, number_of_features);

	// store the sample numbers so that they can correspond to the correct sample within the dataset when shuffled
	int* sample_numbers = new int[number_of_samples];
	for (int i = 0; i < number_of_samples; i++)
		sample_numbers[i] = i + 1;

	// randomize locally
	randomize_training_samples(training_features, target_values, log_transformed_target_values, sample_numbers, number_of_samples);

	// calculate normalized features of all the samples; will be used for testing predictions
	double* all_features_means = calculate_features_means(training_features, ignore_normalization, number_of_features, number_of_samples);
	double* all_features_variances = calculate_features_variances(training_features, ignore_normalization, all_features_means, number_of_features, 
		number_of_samples);
	double** all_features_normalized = calculate_normalized_features(training_features, ignore_normalization, number_of_samples,
		number_of_features, all_features_means, all_features_variances);

	dataset_file.close();

	// create a directory to store the neural network files if not already made
	if (std::filesystem::create_directory(nn_saved_state_file_path))
		std::cout << "\n\tCreating " << nn_saved_state_directory_name << " directory to store the state of your neural network...\n";

	// create a directory to store the training logs if not already made
	if (std::filesystem::create_directory(training_logs_file_path))
		std::cout << "\n\tCreating " << training_logs_directory_name << " directory to store the training logs...\n";

	char option;
	NeuralNetwork neural_network(number_of_neurons_each_hidden_layer, net_number_of_neurons_in_hidden_layers, number_of_hidden_layers, 
		number_of_features, weights_and_biases_file_path, means_and_vars_file_path, scales_and_shifts_file_path, 
		weights_and_biases_file_name, means_and_vars_file_name, scales_and_shifts_file_name);
	TrainingLogList log_list(training_logs_file_path);

	std::cout << "\nHello! Welcome to my hard-coded neural network program.\n";

	while (true)
	{
		MenuOptions selected_option = get_option();

		// end the program if selected
		if (selected_option == MenuOptions::EXIT_PROGRAM_OPTION) break;

		generate_border_line();

		switch (selected_option)
		{

		case MenuOptions::ALL_SAMPLE_TRAIN_OPTION:

			all_sample_train_network_option(neural_network, log_list, all_features_normalized, log_transformed_target_values, number_of_samples);
			break;

		case MenuOptions::FIVE_FOLD_TRAIN_OPTION:

			k_fold_train_network_option(neural_network, log_list, training_features, ignore_normalization, log_transformed_target_values, 
				number_of_samples);
			break;

		case MenuOptions::SAVE_NETWORK_STATE_OPTION: // save current neural network

			save_neural_network_state(neural_network, weights_and_biases_file_path, means_and_vars_file_path, scales_and_shifts_file_path,
				number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features, net_number_of_neurons_in_hidden_layers);
			break; 

		case MenuOptions::RANDOMIZE_SAMPLES_OPTION: // change learning and regularization parameters

			randomize_training_samples(training_features, target_values, log_transformed_target_values, sample_numbers, number_of_samples);
			break;

		case MenuOptions::PREDICT_PROVIDED_FEATURES_OPTION: // change the batch size of the nn and regen it due to const qualifiers
			
			predict_with_provided_features_option(neural_network, feature_names, target_name, all_features_means, all_features_variances,
				ignore_normalization, number_of_features);
			break;

		case MenuOptions::PREDICT_RANDOM_FEATURES_OPTION:

			predict_with_random_features_option(neural_network, feature_names, target_name, training_features, all_features_normalized, 
				target_values, log_transformed_target_values, sample_numbers, number_of_samples, number_of_features);
			break;

		case MenuOptions::PRINT_TRAINING_LOGS_OPTION:

			log_list.print_all_training_logs();
			break;

		case MenuOptions::SAVE_TRAINING_LOGS_OPTION:

			log_list.save_training_logs();

		}

		generate_border_line();
		std::cout << "\n";
	}

	deallocate_memory_for_2D_array(training_features, number_of_samples);
	delete[] target_values;
	delete[] log_transformed_target_values;
	delete[] feature_names;
	delete[] ignore_normalization;

	std::cout << "\nEnding program...\n";
	return 0;
}
