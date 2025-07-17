#pragma once

// memory allocation methods
double*** allocate_memory_for_weights(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
double** allocate_memory_for_biases(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers);
double** allocate_memory_for_2D_array(int number_of_rows, int number_of_columns);

// memory deallocation methods
void deallocate_memory_for_weights(double*** weights, const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers);
void deallocate_memory_for_biases(double** biases, int number_of_hidden_layers);
void deallocate_memory_for_2D_array(double** const & twoD_array, int number_of_rows);
void deallocate_memory_for_2D_array(double**& twoD_array, int number_of_rows);