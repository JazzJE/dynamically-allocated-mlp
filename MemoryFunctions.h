#pragma once

// memory allocation methods
double*** allocate_memory_for_weights(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
double** allocate_memory_for_biases(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers);
double** allocate_memory_for_training_features(int number_of_samples, int number_of_features);
double** allocate_memory_for_mv_or_ss(int net_number_of_neurons_in_hidden_layers);
double* allocate_memory_for_target_values(int number_of_samples);

// memory deallocation methods
void deallocate_memory_for_weights(double*** weights, const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers);
void deallocate_memory_for_biases(double** biases, int number_of_hidden_layers);
void deallocate_memory_for_training_features(double** training_features, int number_of_samples);
void deallocate_memory_for_mv_or_ss(double** mv_or_ss, int net_number_of_neurons_in_hidden_layers);
void deallocate_memory_for_target_values(double* target_values);