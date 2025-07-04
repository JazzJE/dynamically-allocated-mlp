#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <cmath>

// dataset file methods
void parse_dataset_file(std::fstream& dataset_file, double** training_features, double* target_values, std::string* feature_names,
	std::string& target_name, int number_of_features, int number_of_samples);
void validate_dataset_file(std::fstream& dataset_file, std::string dataset_file_name, int number_of_features);
int find_error_dataset_file(std::fstream& dataset_file, int number_of_features);

// miscallaneous methods
void randomize_training_samples(double** training_features, double* target_values, double* log_transformed_target_values, 
	int* sample_numbers, int number_of_samples);
int count_number_of_samples(std::string dataset_file_name);
int count_number_of_features(std::string dataset_file_name);
bool* identify_not_normalize_feature_columns(std::string* feature_names, int number_of_features);
double* calculate_log_transformed_target_values(double* target_values, int number_of_samples);
void clear_screen();