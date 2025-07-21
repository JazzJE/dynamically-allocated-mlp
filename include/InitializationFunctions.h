#pragma once
#include <iostream>
#include <filesystem>
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
int count_number_of_samples(std::filesystem::path dataset_file_path);
int count_number_of_features(std::filesystem::path dataset_file_path);
bool* identify_ignore_normalization_feature_columns(std::string* feature_names, int number_of_features);
double* calculate_log_transformed_target_values(double* target_values, int number_of_samples);
void clear_screen();