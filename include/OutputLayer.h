#pragma once
#include "DenseLayer.h"

// the output layer will functionally have the same attributes as the normal layer; the only difference will be in the methods, 
// which are optimized for just one (output) neuron
class OutputLayer : public DenseLayer
{
public:
	
	OutputLayer(double** layer_weights, double* layer_biases, double* layer_activation_array, int number_of_features, int number_of_neurons,
		double* learning_rate, double* regularization_rate);

	// delete the output neuron array
	~OutputLayer();

	// no for loops; just a single value
	void compute_activation_array();
	void training_compute_activation_arrays();
	void training_linear_transform();
	void update_parameters();

	// getter/accessor methods only for output layer to access results
	double** get_training_activation_arrays() const;
	double* get_activation_array() const;

};

