#include "NeuralNetwork.h"

// initialize each hidden layer with their...
		// weights,
		// biases,
		// scales and shifts,
		// running means and variances,
		// the number of weights they will have (which is the number of neurons in the previous layer but number of features for first layer),
		// the input arrays of the next layer which will function as the layer's output arrays
		// the network rates,
		// batch size,
		// the number of weights/features,
		// and the number of neurons they will have

NeuralNetwork::NeuralNetwork(const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers,
	int number_of_features, int batch_size, double learning_rate, double regularization_rate, int patience, int prompt_epoch,
	std::string weights_and_biases_file_path, std::string means_and_vars_file_path, std::string scales_and_shifts_file_path) :
	
	network_number_of_features(number_of_features), number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer),
	number_of_hidden_layers(number_of_hidden_layers), network_learning_rate(new double(learning_rate)), 
	network_regularization_rate(new double(regularization_rate)), patience(patience), prompt_epoch(prompt_epoch),

	// dynamic memory allocation of key components
	network_weights(allocate_memory_for_weights(number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features)), 
	network_biases(allocate_memory_for_biases(number_of_neurons_each_hidden_layer, number_of_hidden_layers)),
	network_running_means(new double[net_number_of_neurons_in_hidden_layers]),
	network_running_variances(new double[net_number_of_neurons_in_hidden_layers]),
	network_scales(new double[net_number_of_neurons_in_hidden_layers]),
	network_shifts(new double[net_number_of_neurons_in_hidden_layers]),
	
	batch_size(batch_size),
	hidden_layers(new DenseLayer*[number_of_hidden_layers]), net_number_of_neurons_in_hidden_layers(net_number_of_neurons_in_hidden_layers),
	
	// output layer
	output_layer(new OutputLayer(network_weights[number_of_hidden_layers], network_biases[number_of_hidden_layers],

		// the output for the activation array of training and actual predictions will be "one" activation for any sample
		allocate_memory_for_2D_array(batch_size, 1), new double[1], batch_size, 
		number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1, network_learning_rate, network_regularization_rate))

{
	// ensure that all nn files match up with the array provided by the user, else prompt them to regenerate the nn components
	validate_neural_network_files(weights_and_biases_file_path, means_and_vars_file_path, scales_and_shifts_file_path);

	parse_weights_and_biases_file(weights_and_biases_file_path);
	parse_mv_or_ss_file(means_and_vars_file_path, network_running_means, network_running_variances);
	parse_mv_or_ss_file(scales_and_shifts_file_path, network_scales, network_shifts);

	// "hooking" refers to connecting any given nth layer's input features as the (n - 1)th layer's output activation values via pointers

	// if there is only hidden layer, then hook the output and input layer to this layer
	if (number_of_hidden_layers == 1)

		hidden_layers[0] = new DenseLayer(network_weights[0], network_biases[0], network_running_means, network_running_variances, network_scales,
			network_shifts, output_layer->get_training_input_features(), output_layer->get_input_features(), batch_size, number_of_features, 
			number_of_neurons_each_hidden_layer[0], network_learning_rate, network_regularization_rate);
	
	// else, if there are n layers
	else
	{
		// this will refer to the index of the means and variances & scales and shifts that the layer is allotted to for its neurons
		int current_index = net_number_of_neurons_in_hidden_layers;

		current_index -= number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1];
		hidden_layers[number_of_hidden_layers - 1] = new DenseLayer(network_weights[number_of_hidden_layers - 1], 
			network_biases[number_of_hidden_layers - 1], network_running_means + current_index, network_running_variances + current_index, 
			network_scales + current_index, 
			network_shifts + current_index, output_layer->get_training_input_features(), output_layer->get_input_features(), batch_size, 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 2], 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], network_learning_rate, network_regularization_rate);

		for (int l = number_of_hidden_layers - 1; l > 1; l--)
		{
			current_index -= number_of_neurons_each_hidden_layer[l - 1];
			hidden_layers[l - 1] = new DenseLayer(network_weights[l - 1], network_biases[l - 1], network_running_means + current_index, 
				network_running_variances + current_index, network_scales + current_index, network_shifts + current_index,
				hidden_layers[l]->get_training_input_features(), hidden_layers[l]->get_input_features(), batch_size, 
				number_of_neurons_each_hidden_layer[l - 2], number_of_neurons_each_hidden_layer[l - 1], network_learning_rate, 
				network_regularization_rate);
		}

		// note that the current_index will always equal to 0 at this point
		hidden_layers[0] = new DenseLayer(network_weights[0], network_biases[0], network_running_means, network_running_variances, network_scales,
			network_shifts, hidden_layers[1]->get_training_input_features(), hidden_layers[1]->get_input_features(), batch_size, 
			network_number_of_features, number_of_neurons_each_hidden_layer[0], network_learning_rate, network_regularization_rate);
	}

}

NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < number_of_hidden_layers; i++)
		delete hidden_layers[i];
	delete[] hidden_layers;

	delete output_layer;

	delete network_learning_rate;
	delete network_regularization_rate;

	deallocate_memory_for_weights(network_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(network_biases, number_of_hidden_layers);
	delete[] network_running_means;
	delete[] network_running_variances;
	delete[] network_scales;
	delete[] network_shifts;
}

// train the neural network on five different folds of the training set
void NeuralNetwork::five_fold_train(double** training_features, bool* not_normalize, double* log_transformed_target_values, int number_of_samples)
{
	// best state will save the best state of the nn when the mse is lower than the best mse
	BestStateLoader bs_loader(network_weights, network_biases, network_running_means, network_running_variances, network_scales, network_shifts,
		get_number_of_neurons_each_hidden_layer(), number_of_hidden_layers, net_number_of_neurons_in_hidden_layers, network_number_of_features);

	// lower refers to the index of the lower range of the cross-validation fold, while higher refers to the index of the higher range
	// of the cv fold
	int lower_cross_validation_index, higher_cross_validation_index;
	int samples_per_fold = number_of_samples / 5;

	for (int i = 0; i < 4; i++)
	{
		std::cout << "\n\n\tFold #" << i + 1 << ": ";

		lower_cross_validation_index = i * samples_per_fold;
		higher_cross_validation_index = (i + 1) * samples_per_fold - 1;

		double* training_means = calculate_features_means(training_features, not_normalize, network_number_of_features, 
			number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
		double* training_variances = calculate_features_variances(training_features, not_normalize, training_means,
			network_number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);

		double** training_features_normalized = calculate_normalized_features(training_features, not_normalize, number_of_samples,
			network_number_of_features, training_means, training_variances);

		early_stop_training(bs_loader, training_features_normalized, 
			log_transformed_target_values, lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

		delete[] training_means;
		delete[] training_variances;
		deallocate_memory_for_2D_array(training_features_normalized, number_of_samples);
	}

	std::cout << "\n\n\tFold #5: ";

	// use all the remaining training samples for the kast cross validation set
	lower_cross_validation_index = 4 * samples_per_fold;
	higher_cross_validation_index = number_of_samples - 1;

	double* training_means = calculate_features_means(training_features, not_normalize, network_number_of_features, number_of_samples, 
		lower_cross_validation_index, higher_cross_validation_index);
	double* training_variances = calculate_features_variances(training_features, not_normalize, training_means,
		network_number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
	
	double** training_features_normalized = calculate_normalized_features(training_features, not_normalize, number_of_samples, network_number_of_features,
		training_means, training_variances);

	early_stop_training(bs_loader, training_features_normalized,
		log_transformed_target_values, lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

	// deallocate all unneeded memory
	delete[] training_means;
	delete[] training_variances;
	deallocate_memory_for_2D_array(training_features_normalized, number_of_samples);
}

// run mini-batch gradient descent on the provided fold
void NeuralNetwork::early_stop_training(BestStateLoader& bs_loader, double** training_features_normalized, double* log_transformed_target_values, 
	int lower_validation_index, int higher_validation_index, int number_of_samples)
{
	int epoch_number = 0;
	int fail_decay_epoch = patience / 4 * 3;
	double best_mse, current_mse;

	// store the initial value of the learning rate as we will update it during this function
	double initial_learning_rate = *network_learning_rate;
	
	// each pointer will point to the randomized normalized features within the batch
	double** selected_normalized_features = new double* [batch_size];
	double* selected_log_transformed_target_values = new double[batch_size];

	// count the number of times the network has failed to product a smaller mse value, 
	// and end training with this fold when it meets the patience value
	int failed_epochs = 0;

	// compute initial mse of the fold
	best_mse = 0;
	for (int i = lower_validation_index; i <= higher_validation_index; i++)
	{
		double pred = calculate_prediction(training_features_normalized[i]);
		best_mse += (pred - log_transformed_target_values[i]) * (pred - log_transformed_target_values[i]);
	}
	best_mse /= (higher_validation_index + 1 - lower_validation_index);

	// initialize best state as current state
	bs_loader.save_current_state();

	std::cout << "\n\n\t\tInitial cross-validation MSE is " << best_mse << ".";

	while (failed_epochs < patience)
	{
		epoch_number++;

		// select random sample indices for training
		int* random_sample_indices = select_random_batch_indices(number_of_samples, lower_validation_index, higher_validation_index);

		// assign the random indiced samples to the selected normalized features and target values to be passed into the nn
		for (int i = 0; i < batch_size; i++)
		{
			selected_normalized_features[i] = training_features_normalized[random_sample_indices[i]];
			selected_log_transformed_target_values[i] = log_transformed_target_values[random_sample_indices[i]];
		}

		delete[] random_sample_indices;

		// train the nn
		train_network(selected_normalized_features, selected_log_transformed_target_values);

		// calculate the new mse of the validation set
		current_mse = 0;
		for (int i = lower_validation_index; i <= higher_validation_index; i++)
		{
			double pred = calculate_prediction(training_features_normalized[i]);
			current_mse += (pred - log_transformed_target_values[i]) * (pred - log_transformed_target_values[i]);
		}
		current_mse /= (higher_validation_index + 1 - lower_validation_index);

		// end training early for this fold if we get extremely large errors
		if (current_mse > explosion_max)
		{
			std::cout << "\n\n\t\tExplosion in loss detected - ending this fold early...";
			break;
		}
		else if (current_mse > best_mse)
		{
			failed_epochs++;
			std::cout << "\n\n\t\tTraining epoch #" << epoch_number << ": Current MSE - " << current_mse << ", Best MSE - " << best_mse
				<< "\n\t\t\tCurrent cross-validation MSE is greater than best cross-validation MSE. Failed epochs is now " << failed_epochs << ".";
			
			// decay rate for if the number of failed epochs reaches a certain point
			if (failed_epochs % fail_decay_epoch == 0)
			{
				*network_learning_rate *= decay_rate;
				std::cout << "\n\n\t\tDecaying learning rate by a factor of " << decay_rate << "... "
					<< "The new value of the learning rate is " << *network_learning_rate;
			}
		}
		else
		{
			failed_epochs = 0;
			std::cout << "\n\n\t\tTraining epoch #" << epoch_number << ": Current MSE - " << current_mse << ", Best MSE - " << best_mse 
				<< "\n\t\t\tCurrent cross-validation MSE is less than best cross-validation MSE. Failed epochs is now 0. "
				<< "Saving current state...";
			bs_loader.save_current_state();
			best_mse = current_mse;
		}

		// if the training epochs have gone on for really long, ask user if they would like to stop training for this fold
		if (epoch_number % prompt_epoch == 0)
		{
			char option;
			std::cout << "\n\n\tThis is the " << epoch_number << "th epoch in training for this fold. "
				<< "Would you like to continue training off this fold? (Y / N): ";
			std::cin >> option;

			while (option != 'N' && option != 'Y')
			{
				std::cout << "\t[ERROR] Please only enter yes or no (Y / N): ";
				std::cin >> option;
			}

			if (option == 'N') break;
		}
	}

	std::cout << "\n\n\t\tResetting to best state and restoring initial learning rate...";
	*network_learning_rate = initial_learning_rate;
	bs_loader.load_best_state();

	delete[] selected_normalized_features;
	delete[] selected_log_transformed_target_values;
}

// backpropagate the derived values of all the layers and neurons, then update the all the parameters from beginning to end
void NeuralNetwork::train_network(double** normalized_batch_input_features, double* log_transformed_target_values)
{
	calculate_training_predictions(normalized_batch_input_features);
	backpropagate_derived_values(log_transformed_target_values);
	update_parameters();
}

// calculate training predictions for the training samples
void NeuralNetwork::calculate_training_predictions(double** normalized_input_features) const
{
	// copy the normalized input features into the first layer's training input arrays
	for (int s = 0; s < batch_size; s++)
		for (int f = 0; f < network_number_of_features; f++)
			hidden_layers[0]->get_training_input_features()[s][f] = normalized_input_features[s][f];

	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->training_compute_activation_arrays();

	// output layer will calculate the batch size samples predictions, but will not return them since we won't need to for training
	output_layer->training_compute_activation_arrays();
}

// go from layer to layer, computing each neurons' derived values
void NeuralNetwork::backpropagate_derived_values(double* log_transformed_target_values)
{
	for (int s = 0; s < batch_size; s++)
		output_layer->get_linear_transform_derived_values()[0][s] =
		output_layer->get_training_activation_arrays()[s][0] - log_transformed_target_values[s];

	hidden_layers[number_of_hidden_layers - 1]->calculate_derived_values(output_layer->get_linear_transform_derived_values(),
		output_layer->get_layer_weights(), 1);

	for (int l = number_of_hidden_layers - 1; l > 0; l--)
		hidden_layers[l - 1]->calculate_derived_values(hidden_layers[l]->get_linear_transform_derived_values(),
			hidden_layers[l]->get_layer_weights(), number_of_neurons_each_hidden_layer[l]);
}

// update all the parameters of the entire nn from beginning to end, primarily running means and variances, 
// scales and shifts, weights, and biases
void NeuralNetwork::update_parameters()
{
	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->update_parameters();
	output_layer->update_parameters();
}

// select random batch indices not in the cross validation set
int* NeuralNetwork::select_random_batch_indices(int number_of_samples, int lower_validation_index, int higher_validation_index)
{
	int* selected_sample_indices = new int[batch_size];
	std::unordered_set<int> already_selected_indices;

	// select random sample indices for the batch
	for (int i = 0; i < batch_size; i++)
	{
		while (true)
		{
			// select a random index
			selected_sample_indices[i] = std::rand() % number_of_samples;

			// if the selected index is within the cross-validation set, then select a new index
			if (selected_sample_indices[i] >= lower_validation_index && selected_sample_indices[i] <= higher_validation_index)
				continue;

			// if the selected index has already been selected, then select a new index
			if (already_selected_indices.find(selected_sample_indices[i]) != already_selected_indices.end())
				continue;
			// insert this index if not already in the set
			else
				already_selected_indices.insert(selected_sample_indices[i]);

			break;
		}
	}

	return selected_sample_indices;
}

// return a value based on the current weights and biases as well as the input features; this is for any prediction other than 
// training predictions
double NeuralNetwork::calculate_prediction(double* normalized_input_features) const
{	
	// copy the normalized input features into the first layer's input array
	for (int f = 0; f < network_number_of_features; f++)
		hidden_layers[0]->get_input_features()[f] = normalized_input_features[f];

	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->compute_activation_array();

	// output layer will calculate a singular value and return that value as the result
	output_layer->compute_activation_array();

	return *(output_layer->get_activation_array());
}

// get a dynamically allocated array that will store the number of neurons each hidden layer so best state loader 
// will know how to save the current parameters
int* NeuralNetwork::get_number_of_neurons_each_hidden_layer() const
{
	int* nnehl = new int[number_of_hidden_layers];

	for (int i = 0; i < number_of_hidden_layers; i++)
		nnehl[i] = number_of_neurons_each_hidden_layer[i];

	return nnehl;
}

// mutator/setter methods for rates
void NeuralNetwork::set_regularization_rate(double r_rate)
{ *network_regularization_rate = r_rate; }
void NeuralNetwork::set_learning_rate(double l_rate)
{ (*network_learning_rate) = l_rate; }
void NeuralNetwork::set_patience(int p)
{ patience = p; }
void NeuralNetwork::set_prompt_epoch(int i)
{ prompt_epoch = i; }

// accessor methods for updating the neural network files
double*** NeuralNetwork::get_network_weights() const
{ return network_weights; }
double** NeuralNetwork::get_network_biases() const
{ return network_biases; }
double* NeuralNetwork::get_network_running_means() const
{ return network_running_means; }
double* NeuralNetwork::get_network_running_variances() const
{ return network_running_variances; }
double* NeuralNetwork::get_network_scales() const
{ return network_scales; }
double* NeuralNetwork::get_network_shifts() const
{ return network_shifts; }
double NeuralNetwork::get_learning_rate() const
{ return *network_learning_rate; }
double NeuralNetwork::get_regularization_rate() const
{ return *network_regularization_rate; }
int NeuralNetwork::get_patience() const
{ return patience; }
int NeuralNetwork::get_prompt_epoch() const
{ return prompt_epoch; }
