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
#include "InitializationFunctions.h"
#include "MemoryFunctions.h"
#include "MenuFunctions.h"
#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "StatisticsFunctions.h"

int main()
{
	// !!! REMINDER that the last column is the column of values you want to predict !!!
		// for example, if you have features of a house and want to predict price, then make the house prices the last column of the csv
	// data set used: https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download

	// !!! NOTE AGAIN that this program is hard-written with a single neuron for output !!!
	// this is the order and number of neurons you want in each hidden layer
	// in the below example...
		// the first hidden layer will have 256 neurons
		// the second hidden layer will have 128 neurons
		// the third hidden layer will have 32 neurons
		// the fourth/output layer (automatically/implicitly created) will have 1 neuron, predicting the value
	const int number_of_neurons_each_hidden_layer[] = { 256, 128, 32 };

	// this changes how many samples you want whenever you train your network; you can increase it to whatever number you want,
	// so long as it isn't greater than the max number of samples in a "fold," which will equal to (n 
	const int batch_size = 2048;


	// PAST THIS POINT IS ALL THE HARD CODE; REFER TO ABOVE PARTS FOR EDITABLE COMPONENTS

	std::cout << "The program may take a couple of seconds to load...\n" << std::fixed;

	std::string dataset_file_name = "dataset.csv";

	// check if there is a negative number or zero in the number of neurons each hidden layer array
	int number_of_hidden_layers = sizeof(number_of_neurons_each_hidden_layer) / sizeof(int);
	
	if (number_of_hidden_layers == 0)
	{
		std::cerr << "[ERROR] Before using this program, please ensure that the \'number_of_neurons_each_hidden_layer\' array "
			<< "has at least 1 integer.";
		exit(0);
	}

	for (int i = 0; i < number_of_hidden_layers; i++)
		if (number_of_neurons_each_hidden_layer[i] <= 0)
		{
			std::cerr << "Before using this program, please ensure that there are no zero or negative values in the "
				<< "\"number_of_neurons_each_hidden_layer\" array";
			exit(0);
		}

	// when choosing batch samples and generating a new weights/biases folder, we will use the rand method
	srand(time(0));

	// EXCLUDING the singular neuron in the "output layer"; this will be used for assigning means and variances & scales and shifts
	int net_number_of_neurons_in_hidden_layers = 0;
	for (int l = 0; l < number_of_hidden_layers; l++)
		net_number_of_neurons_in_hidden_layers += number_of_neurons_each_hidden_layer[l];

	std::fstream dataset_file(dataset_file_name, std::ios::in);
	if (!dataset_file) 
	{
		std::cerr << "[ERROR] The dataset could not be found within the project; please edit the \"dataset_file_name\" variable "
			<< "to the dataset's name, or otherwise include the dataset within the project." << std::endl;
		exit(0);
	}

	// make the program a little more autonomous so user doesn't need to be confused entering these values
	int number_of_samples = count_number_of_samples(dataset_file_name);
	int number_of_features = count_number_of_features(dataset_file_name);

	// given the number of features, ensure that each row has the same number of columns
	validate_dataset_file(dataset_file, dataset_file_name, number_of_features);

	double** training_features = allocate_memory_for_2D_array(number_of_samples, number_of_features);
	double* target_values = new double[number_of_samples];

	// get the names of the columns during parsing
	std::string* const feature_names = new std::string[number_of_features];
	std::string target_name;
	parse_dataset_file(dataset_file, training_features, target_values, feature_names, target_name, number_of_features, number_of_samples);

	// from the feature names, identify which features should not be normalized if their column names start with a ~ sign
	// the calculating means, variances, and features helper methods will ignore these features during their calculations
	bool* const not_normalize = new bool[number_of_features]();
	identify_not_normalize_feature_columns(feature_names, not_normalize, number_of_features);

	// randomize locally
	int number_of_shuffles = 20;
	for (int s = 0; s < number_of_shuffles; s++)
		randomize_training_samples(training_features, target_values, number_of_samples);

	// calculate normalized features of all the samples; will be used for testing predictions
	double* all_samples_means = calculate_features_means(training_features, not_normalize, number_of_features, number_of_samples);
	double* all_samples_variances = calculate_features_variances(training_features, not_normalize, all_samples_means, number_of_features, 
		number_of_samples);
	double** all_samples_normalized_features = calculate_normalized_features(training_features, not_normalize, number_of_samples,
		number_of_features, all_samples_means, all_samples_variances);

	dataset_file.close();


	// validate all the neural network files, and also check if they exist, otherwise generate new ones
	std::string weights_and_biases_file_name = "weights_and_biases.csv";
	std::string means_and_vars_file_name = "means_and_vars.csv";
	std::string scales_and_shifts_file_name = "scales_and_shifts.csv";

	validate_neural_network_files(number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features, 
		net_number_of_neurons_in_hidden_layers, weights_and_biases_file_name, means_and_vars_file_name, scales_and_shifts_file_name);

	char option;
	double learning_rate, regularization_rate;

	std::cout << "Hello! Welcome to my hard-coded neural network program.\n";
	std::cout << "\nBefore beginning, please give initial values for the following parameters.\n\n";

	// ask user for an initial value of the learning rate and the regularization values
	generate_border_line();
	input_parameter_rates(learning_rate, regularization_rate);
	generate_border_line();

	// neural network will parse its files into itself
	NeuralNetwork neural_network(number_of_neurons_each_hidden_layer, net_number_of_neurons_in_hidden_layers, number_of_hidden_layers, 
		number_of_features, batch_size, learning_rate, regularization_rate, weights_and_biases_file_name, 
		means_and_vars_file_name, scales_and_shifts_file_name);

	while (true)
	{
		std::cout << "\nOption Menu:"
			<< "\n\t1. Train neural network (five-fold, mini-batch gradient descent)"
			<< "\n\t2. Randomize order of training samples"
			<< "\n\t3. Predict a value using provided sample features"
			<< "\n\t4. Predict a value using a random sample"
			<< "\n\t5. Save your current best neural network configs (update all files to latest best version in this program)"
			<< "\n\t6. Change learning and regularization parameters"
			<< "\n\t7. Exit program (exiting will not save the network)"
			<< "\nPlease select an option: ";
		std::cin >> option;

		while (option < '1' || option > '7')
		{
			std::cout << "[ERROR] Please enter a valid input (1-6): ";
			std::cin >> option;
		}

		// end the program if selected
		if (option == '7') break;

		generate_border_line();

		switch (option)
		{
		case '1': // train neural network

			std::cout << "\n\tTraining your network...";
			neural_network.five_fold_train(training_features, not_normalize, target_values, number_of_samples);
			std::cout << "\n\n\tDone!\n";

			break;

		case '2': // randomize order of training samples again

			std::cout << "\n\tRandomizing samples' order...";
			for (int s = 0; s < number_of_shuffles; s++)
				randomize_training_samples(training_features, target_values, number_of_samples);
			std::cout << "\n\n\tDone!\n";

			break;

		case '3': // predict a value provided user features

		{

			double* new_features = input_new_features(feature_names, not_normalize, number_of_features);

			std::cout << "\n\tProvided these features: ";
			for (int f = 0; f < number_of_features; f++)
				std::cout << "\n\t\t" << feature_names[f] << " - " << new_features[f];

			double* normalized_features = calculate_normalized_features(new_features, not_normalize, number_of_features,
				all_samples_means, all_samples_variances);

			std::cout << "\n\n\tNormalized features: ";
			for (int f = 0; f < number_of_features; f++)
				std::cout << normalized_features[f] << " ";
			std::cout << "\n\n\tPrediction of " << target_name << ": " << neural_network.calculate_prediction(normalized_features) << "\n";

			delete[] new_features;
			delete[] normalized_features;

			break;
		}

		case '4': // predict a value with a random sample

		{
			int random_index = std::rand() % number_of_samples;
			std::cout << "\n\tProvided these features for sample #" << random_index << " : ";
			for (int f = 0; f < number_of_features; f++)
				std::cout << "\n\t\t" << feature_names[f] << " - " << training_features[random_index][f];

			std::cout << "\n\n\tActual value of " << target_name << ": " << target_values[random_index];

			double* normalized_features = calculate_normalized_features(training_features[random_index], not_normalize, number_of_features,
				all_samples_means, all_samples_variances);

			std::cout << "\n\n\tNormalized features: ";
			for (int f = 0; f < number_of_features; f++)
				std::cout << normalized_features[f] << " ";
			std::cout << "\n\n\tPrediction of " << target_name << ": " << neural_network.calculate_prediction(normalized_features) << "\n";

			delete[] normalized_features;

			break;
		}

		case '5': // save current neural network

			std::cout << "\n\n\tUpdating the " << weights_and_biases_file_name << " file...";
			update_weights_and_biases_file(weights_and_biases_file_name, neural_network.get_network_weights(), 
				neural_network.get_network_biases(), number_of_neurons_each_hidden_layer, number_of_hidden_layers, 
				number_of_features);

			std::cout << "\n\tUpdating the " << means_and_vars_file_name << " file...";
			update_mv_or_ss_file(means_and_vars_file_name, neural_network.get_network_running_means(), 
				neural_network.get_network_running_variances(), net_number_of_neurons_in_hidden_layers);

			std::cout << "\n\tUpdating the " << scales_and_shifts_file_name << " file...";
			update_mv_or_ss_file(scales_and_shifts_file_name, neural_network.get_network_scales(),
				neural_network.get_network_shifts(), net_number_of_neurons_in_hidden_layers);
			
			std::cout << "\n\n\tDone!\n";

			break; 

		case '6': // change learning and regularization parameters

			input_parameter_rates(learning_rate, regularization_rate);
			neural_network.set_learning_rate(learning_rate);
			neural_network.set_regularization_rate(regularization_rate);

		}

		generate_border_line();
	}
	std::cout << "\nEnding program...\n";
	return 0;
}
