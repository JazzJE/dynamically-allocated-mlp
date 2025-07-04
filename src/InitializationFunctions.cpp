#include "InitializationFunctions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>

// dataset file methods

	// parse in the csv of data
void parse_dataset_file(std::fstream& dataset_file, double** training_samples, double* target_values, std::string* feature_names,
	std::string& target_name, int number_of_features, int number_of_samples)
{
	std::string line, value;
	std::stringstream ss;
	
	// get the first line with titles and parse them into the column names array
	getline(dataset_file, line);

	ss.clear();
	ss.str(line);

	for (int f = 0; f < number_of_features; f++)
	{
		getline(ss, value, ',');
		feature_names[f] = value;
	}

	getline(ss, value, '\n');
	target_name = value;

	// for each training sample t
	for (int t = 0; t < number_of_samples; t++)
	{
		// get the line of features
		getline(dataset_file, line);

		// clear stringstream object
		ss.clear();
		ss.str(line);

		// get each feature and input them into the training samples
		for (int f = 0; f < number_of_features; f++)
		{
			getline(ss, value, ',');
			training_samples[t][f] = std::stod(value);
		}

		// get the target value, which is the last column
		getline(ss, value, '\n');
		target_values[t] = std::stod(value);
	}
}

	// output an error and end program if the samples are not consistent/do not all have the same number of columns/features
void validate_dataset_file(std::fstream& dataset_file, std::string dataset_file_name, int number_of_features)
{
	// the find_error_dataset_file function will return an integer value of the line in which the error was found
	// if no error is found, the error returned is -1
	int line_error = find_error_dataset_file(dataset_file, number_of_features);
	if (line_error != -1)
	{
		std::cerr << "[ERROR] The " << dataset_file_name << " is inconsistent(aka some rows have more features / columns than others) "
			<< "OR there is a string value in the dataset (ONLY THE FIRST LANE CAN HAVE FEATURE NAMES; the rest of the dataset must only accept double values)."
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << dataset_file_name << " ***"
			<< "\n\nPlease update your dataset file accordingly."
			<< "\n\nEnding program...\n";
		exit(0);
	}
}

// validate that the samples are valid (all of them have the same amount of features)
// the function will return the line in which the error was found so it can be altered easily; if not found, return -1
int find_error_dataset_file(std::fstream& dataset_file, int number_of_features)
{
	int line_error = 0;
	int field_count;
	double temp_double;

	std::string line, value;
	std::stringstream ss, converter;

	// ignore the first line with column titles
	getline(dataset_file, line);

	while (getline(dataset_file, line))
	{
		line_error++;
		field_count = 0;

		ss.clear();
		ss.str(line);

		while (getline(ss, value, ','))
		{
			// validate that the value being parsed is not a string or anamolous — there should only be double values
			converter.clear();
			converter.str(value);

			converter >> temp_double;
			if (converter.fail() || !converter.eof())
				return line_error;

			field_count++;
		}

		// the field count also counts the last column, but must ignore it as it's not an input feature but the target value field
		if (field_count - 1 != number_of_features) return line_error;
	}

	// reset to start as no error was found
	dataset_file.clear();
	dataset_file.seekg(0);
	return -1;
}

// miscellaneous methods

	// randomize the order of the training samples
void randomize_training_samples(double** training_features, double* target_values, double* log_transformed_target_values, 
	int* sample_numbers, int number_of_samples)
{
	int random_index;
	double temp_double;
	double* temp_ptr;
	int temp_int;

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
}

	// count the number of samples in the file
int count_number_of_samples(std::string dataset_file_name)
{
	std::fstream dataset_file(dataset_file_name);

	int counter = 0;
	std::string line;

	// ignore first line with titles
	getline(dataset_file, line);

	while (getline(dataset_file, line))
		counter++;

	dataset_file.close();

	return counter;
}

	// count number of column titles, which is equal to number of features
	// note that the number of features is equal to the number of fields taken in minus 1, given the last column are target values
int count_number_of_features(std::string dataset_file_name)
{
	std::fstream dataset_file(dataset_file_name);

	int counter = 0;
	std::string line, value;

	getline(dataset_file, line);
	std::stringstream ss(line);

	while (getline(ss, value, ','))
		counter++;

	dataset_file.close();

	return (counter - 1);
}

	// identify which column features are to not be normalized given by if the string begins with a ~
bool* identify_not_normalize_feature_columns(std::string* feature_names, int number_of_features)
{
	bool* not_normalize = new bool[number_of_features]();

	for (int f = 0; f < number_of_features; f++)
		if (feature_names[f][0] == '~')
		{
			not_normalize[f] = true;

			// this is to update the name of the column to just whitespace if the column name is (somehow) just the tilde by itself
			if (feature_names[f].length() == 1)
			{
				feature_names[f] = " ";
				continue;
			}

			// get rid of the tilde from the feature name
			feature_names[f] = feature_names[f].substr(1, feature_names[f].length());
		}

	return not_normalize;
}

	// calculate the log transformed values of the target values due to their variance
double* calculate_log_transformed_target_values(double* target_values, int number_of_samples)
{
	double* log_transformed_target_values = new double[number_of_samples];

	for (int s = 0; s < number_of_samples; s++)
		log_transformed_target_values[s] = log(target_values[s] + 1);

	return log_transformed_target_values;
}

void clear_screen()
{ std::cout << "\x1B[2J\x1B[H"; }