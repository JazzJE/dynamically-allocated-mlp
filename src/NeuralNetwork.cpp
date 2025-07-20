#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "OutputLayer.h"
#include "MemoryFunctions.h"
#include "MenuFunctions.h"
#include "StatisticsFunctions.h"
#include "Constants.h"

// initialize each hidden layer with their...
		// weights,
		// biases,
		// scales and shifts,
		// running means and variances,
		// the number of weights they will have (which is the number of neurons in the previous layer but number of features for first layer),
		// the input arrays of the next layer which will function as the layer's output arrays
		// the network rates,
		// the number of weights/features,
		// and the number of neurons they will have

NeuralNetwork::NeuralNetwork(const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons_in_hidden_layers, int number_of_hidden_layers,
	int number_of_features, std::filesystem::path weights_and_biases_file_path, std::filesystem::path means_and_vars_file_path,
	std::filesystem::path scales_and_shifts_file_path, std::string weights_and_biases_file_name, std::string means_and_vars_file_name, 
	std::string scales_and_shifts_file_name) :
	
	number_of_features(number_of_features), 

	number_of_neurons_each_hidden_layer(create_dynamically_allocated_number_of_neurons_each_hidden_layer_array
	(number_of_neurons_each_hidden_layer, number_of_hidden_layers)),

	number_of_hidden_layers(number_of_hidden_layers), learning_rate(new double), regularization_rate(new double),

	ss_loader(network_weights, network_biases, network_running_means, network_running_variances, network_scales, network_shifts,
		number_of_neurons_each_hidden_layer, number_of_hidden_layers, net_number_of_neurons_in_hidden_layers, 
		number_of_features),

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
		new double[1], number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1, learning_rate, regularization_rate))

{
	// ensure that all nn files match up with the array provided by the user, else prompt them to regenerate the nn components
	validate_neural_network_files(weights_and_biases_file_path, means_and_vars_file_path, scales_and_shifts_file_path, 
		weights_and_biases_file_name, means_and_vars_file_name, scales_and_shifts_file_name);

	parse_weights_and_biases_file(weights_and_biases_file_path);
	parse_mv_or_ss_file(means_and_vars_file_path, network_running_means, network_running_variances);
	parse_mv_or_ss_file(scales_and_shifts_file_path, network_scales, network_shifts);

	// "hooking" refers to connecting any given nth layer's input features as the (n - 1)th layer's output activation values via pointers

	// if there is only hidden layer, then hook the output and input layer to this layer
	if (number_of_hidden_layers == 1)

		hidden_layers[0] = new DenseLayer(network_weights[0], network_biases[0], network_running_means, network_running_variances, 
			network_scales, network_shifts, output_layer->get_input_features(), number_of_features, number_of_neurons_each_hidden_layer[0], 
			learning_rate, regularization_rate);
	
	// else, if there are n layers
	else
	{
		// this will refer to the index of the means and variances & scales and shifts that the layer is allotted to for its neurons
		int current_index = net_number_of_neurons_in_hidden_layers;

		current_index -= number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1];
		hidden_layers[number_of_hidden_layers - 1] = new DenseLayer(network_weights[number_of_hidden_layers - 1], 
			network_biases[number_of_hidden_layers - 1], network_running_means + current_index, network_running_variances + current_index, 
			network_scales + current_index, network_shifts + current_index, output_layer->get_input_features(), 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 2], 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], learning_rate, regularization_rate);

		for (int l = number_of_hidden_layers - 1; l > 1; l--)
		{
			current_index -= number_of_neurons_each_hidden_layer[l - 1];
			hidden_layers[l - 1] = new DenseLayer(network_weights[l - 1], network_biases[l - 1], network_running_means + current_index, 
				network_running_variances + current_index, network_scales + current_index, network_shifts + current_index, 
				hidden_layers[l]->get_input_features(), number_of_neurons_each_hidden_layer[l - 2], 
				number_of_neurons_each_hidden_layer[l - 1], learning_rate, regularization_rate);
		}

		// note that the current_index will always equal to 0 at this point
		hidden_layers[0] = new DenseLayer(network_weights[0], network_biases[0], network_running_means, network_running_variances, network_scales,
			network_shifts, hidden_layers[1]->get_input_features(), number_of_features, number_of_neurons_each_hidden_layer[0], 
			learning_rate, regularization_rate);
	}

}

NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < number_of_hidden_layers; i++)
		delete hidden_layers[i];
	delete[] hidden_layers;

	delete output_layer;

	delete learning_rate;
	delete regularization_rate;

	deallocate_memory_for_weights(network_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(network_biases, number_of_hidden_layers);
	delete[] network_running_means;
	delete[] network_running_variances;
	delete[] network_scales;
	delete[] network_shifts;
}

// helper function to create a dynamically allocated version of the number of neurons each hidden layer array for easy access for
// the components of the neural network
const int* NeuralNetwork::create_dynamically_allocated_number_of_neurons_each_hidden_layer_array(const int* number_of_neurons_each_hidden_layer, 
	int number_of_hidden_layers)
{
	int* temp = new int[number_of_hidden_layers];

	for (int l = 0; l < number_of_hidden_layers; l++)
		temp[l] = number_of_neurons_each_hidden_layer[l];

	return temp;
}

// update batch size
void NeuralNetwork::update_arrays_using_batch_size()
{
	// deallocate all arrays using batch size and set them to nullptr to ensure that the deallocate_2d_array methods function properly
	output_layer->deallocate_arrays_using_batch_size();
	for (int l = number_of_hidden_layers - 1; l >= 0; l--)
		hidden_layers[l]->deallocate_arrays_using_batch_size();

	// allocate memory and connect all the layers to each other
	// output layer will have batch size x 1 2d array of activations for output as there is only one output neuron
	output_layer->allocate_arrays_using_batch_size(batch_size, allocate_memory_for_2D_array(batch_size, 1));

	if (number_of_hidden_layers == 1)
		hidden_layers[0]->allocate_arrays_using_batch_size(batch_size, output_layer->get_training_input_features());
	else
	{
		hidden_layers[number_of_hidden_layers - 1]->allocate_arrays_using_batch_size(batch_size, output_layer->get_training_input_features());
		for (int l = number_of_hidden_layers - 1; l > 1; l--)
			hidden_layers[l - 1]->allocate_arrays_using_batch_size(batch_size, hidden_layers[l]->get_training_input_features());

		// note that the current_index will always equal to 0 at this point
		hidden_layers[0]->allocate_arrays_using_batch_size(batch_size, hidden_layers[1]->get_training_input_features());
	}
}

// train the neural network on five different folds of the training set
void NeuralNetwork::k_fold_train(TrainingLogList& log_list, double** training_features, bool* not_normalize, double* log_transformed_target_values, 
	int number_of_samples, int number_of_folds)
{	
	// save the initial state of the neural network, which all folds will reset to when patience is reached
	ss_loader.save_current_state();

	// lower refers to the index of the lower range of the cross-validation fold, while higher refers to the index of the higher range
	// of the cv fold
	int lower_cross_validation_index, higher_cross_validation_index;
	int samples_per_fold = number_of_samples / number_of_folds;

	// get the best mse for each fold to add create a new transaction object
	double* best_mse_for_each_fold = new double[number_of_folds];

	for (int i = 0; i < number_of_folds - 1; i++)
	{
		std::cout << "\n\n\tFold #" << i + 1 << ": ";

		lower_cross_validation_index = i * samples_per_fold;
		higher_cross_validation_index = (i + 1) * samples_per_fold - 1;

		double* training_means = calculate_features_means(training_features, not_normalize, number_of_features, 
			number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
		double* training_variances = calculate_features_variances(training_features, not_normalize, training_means,
			number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);

		double** training_features_normalized = calculate_normalized_features(training_features, not_normalize, number_of_samples,
			number_of_features, training_means, training_variances);

		best_mse_for_each_fold[i] = early_stop_training(training_features_normalized, log_transformed_target_values, number_of_samples, 
			lower_cross_validation_index, higher_cross_validation_index);

		// reset the nn to the original state for future folds
		std::cout << "\n\n\t\tResetting the neural network to its initial state...";
		ss_loader.load_saved_state();

		delete[] training_means;
		delete[] training_variances;
		deallocate_memory_for_2D_array(training_features_normalized, number_of_samples);
	}

	std::cout << "\n\n\tFold #" << number_of_folds << ": ";

	// use all the remaining training samples for the kast cross validation set
	lower_cross_validation_index = (number_of_folds - 1) * samples_per_fold;
	higher_cross_validation_index = number_of_samples - 1;

	double* training_means = calculate_features_means(training_features, not_normalize, number_of_features, number_of_samples, 
		lower_cross_validation_index, higher_cross_validation_index);
	double* training_variances = calculate_features_variances(training_features, not_normalize, training_means,
		number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
	
	double** training_features_normalized = calculate_normalized_features(training_features, not_normalize, number_of_samples, number_of_features,
		training_means, training_variances);

	best_mse_for_each_fold[number_of_folds - 1] = early_stop_training(training_features_normalized, log_transformed_target_values, 
		number_of_samples, lower_cross_validation_index, higher_cross_validation_index);

	// reset nn to initial state
	std::cout << "\n\n\t\tResetting the neural network to its initial state...";
	std::cout << "\n";
	ss_loader.load_saved_state();

	delete[] training_means;
	delete[] training_variances;
	deallocate_memory_for_2D_array(training_features_normalized, number_of_samples);

	generate_border_line();

	std::string new_session_name;
	input_session_name(new_session_name);

	generate_border_line();

	// print a final log and add it to the list of training logs
	bool using_all_samples = false;
	TrainingLog* new_training_log = new TrainingLog(new_session_name, using_all_samples, *learning_rate, *regularization_rate, patience,
		number_of_epochs, best_mse_for_each_fold, number_of_folds, batch_size);
	new_training_log->print_training_log();
	log_list.add_training_log(new_training_log);

	delete[] best_mse_for_each_fold;
}

// method use to train the network explicitly for all samples and eventually generate a final model
void NeuralNetwork::all_sample_train(TrainingLogList& log_list, double** all_normalized_training_features, double* log_transformed_target_values, int number_of_samples)
{
	// save the initial, current state of the neural network
	ss_loader.save_current_state();

	// these will be used by the training log to display corresponding information of this session
	int number_of_folds = 1;
	double* best_mse_for_all_samples = new double;

	// train the neural network with early stop training on the entire data set as the cross-validation set
	int first_sample_index = 0;
	int last_sample_index = number_of_samples - 1;
	*best_mse_for_all_samples = early_stop_training(all_normalized_training_features, log_transformed_target_values, number_of_samples, first_sample_index, 
		last_sample_index);
	std::cout << "\n";

	generate_border_line();

	std::string new_session_name;
	input_session_name(new_session_name);

	generate_border_line();

	// print the training log
	bool using_all_samples = true;
	TrainingLog* new_training_log = new TrainingLog(new_session_name, using_all_samples, *learning_rate, *regularization_rate, patience, 
		number_of_epochs, best_mse_for_all_samples, number_of_folds, batch_size);
	new_training_log->print_training_log();
	log_list.add_training_log(new_training_log);
	std::cout << "\n";
	
	delete best_mse_for_all_samples;

	generate_border_line();

	// ask user if they would like to keep the state of the neural network locally within the program
	char option;
	std::cout << "\n\tWould you like to use this neural network for the rest of the current program duration?"
		<< "\n\t\t- Note that this will NOT save the network; select menu option " 
		<< static_cast<char>(MenuOptions::SAVE_NETWORK_STATE_OPTION) << " after entering \'Y\' if desired (Y / N) : ";
	std::cin >> option;

	while (option != 'Y' && option != 'N' || std::cin.peek() != '\n')
	{
		std::cin.clear(); // clear error flags
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // ignore the key buffer bad input

		std::cout << "\t\t[ERROR] Please enter a valid input (Y / N): ";
		std::cin >> option;
	}

	if (option == 'Y')
	{
		std::cout << "\n\tUsing current state...";
		ss_loader.save_current_state();
	}
	else
	{
		std::cout << "\n\tLoading initial state...";
		ss_loader.load_saved_state();
	}
}

// run mini-batch gradient descent on the provided fold
double NeuralNetwork::early_stop_training(double** training_features_normalized, double* log_transformed_target_values, 
	int number_of_samples, int lower_validation_index, int higher_validation_index)
{
	int epoch_number = 1;
	double best_mse, current_mse;
	int fail_decay_epoch = patience * 2 / 3;

	// store the initial value of the learning rate as we will update it during this function
	double initial_learning_rate = *learning_rate;
	
	// each pointer will point to the randomized normalized features within the batch
	double** selected_normalized_features = new double* [batch_size];
	double* selected_log_transformed_target_values = new double[batch_size];

	// count the number of times the network has failed to product a smaller mse value, 
	// and end training with this fold when it meets the patience value
	int failed_epochs = 0;

	best_mse = std::numeric_limits<double>::infinity();

	while (failed_epochs < patience && epoch_number % (number_of_epochs + 1) != 0)
	{
		int* random_sample_indices = select_random_batch_indices(number_of_samples, lower_validation_index, higher_validation_index);

		// assign the random indiced samples to the selected normalized features and target values to be passed into the nn
		for (int i = 0; i < batch_size; i++)
		{
			selected_normalized_features[i] = training_features_normalized[random_sample_indices[i]];
			selected_log_transformed_target_values[i] = log_transformed_target_values[random_sample_indices[i]];
		}

		delete[] random_sample_indices;

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
		if (current_mse > Constants::explosion_max)
		{
			std::cout << "\n\n\t\tExplosion in loss detected - ending this fold early...";
			break;
		}
		
		// if the current mse is greater than the best mse, then fail the epoch
		if (current_mse > best_mse)
		{
			failed_epochs++;
			std::cout << "\n\n\t\tTraining epoch #" << epoch_number << ": Current MSE - " << current_mse << ", Best MSE - " << best_mse
				<< "\n\t\t\tCurrent cross-validation MSE is greater than best cross-validation MSE. Failed epochs is now " << failed_epochs << ".";
			
			// decay rate for if the number of failed epochs reaches a certain point
			if (failed_epochs % fail_decay_epoch == 0)
			{
				*learning_rate *= Constants::decay_rate;
				std::cout << "\n\n\t\tDecaying learning rate by a factor of " << Constants::decay_rate << "... "
					<< "The new value of the learning rate is " << *learning_rate;
			}
		}
		else
		{
			failed_epochs = 0;
			std::cout << "\n\n\t\tTraining epoch #" << epoch_number << ": Current MSE - " << current_mse << ", Best MSE - " << best_mse 
				<< "\n\t\t\tCurrent cross-validation MSE is less than best cross-validation MSE. Failed epochs is now 0. "
				<< "Updating best MSE...";
			best_mse = current_mse;
		}

		epoch_number++;
	}

	std::cout << "\n\n\t\tRestoring initial learning rate value...";
	*learning_rate = initial_learning_rate;

	delete[] selected_normalized_features;
	delete[] selected_log_transformed_target_values;

	return best_mse;
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
		for (int f = 0; f < number_of_features; f++)
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

// print final status report of the best mse of each fold, as well as the current learning rates

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

	// this is a condition used such that the second condition for checking if the selected index is in the cross-validation set
	// always fails, as if we are using the entire data set for indices (or from index 0 to n - 1), 
	// any random index will always be between the lower_validation index and higher_validation_index 
	bool using_entire_dataset = lower_validation_index == 0 && higher_validation_index == number_of_samples - 1;

	for (int i = 0; i < batch_size; i++)
	{
		while (true)
		{
			selected_sample_indices[i] = std::rand() % number_of_samples;

			if (already_selected_indices.find(selected_sample_indices[i]) != already_selected_indices.end())
				continue;

			if (!using_entire_dataset && 
				selected_sample_indices[i] >= lower_validation_index && selected_sample_indices[i] <= higher_validation_index)
				continue;

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
	for (int f = 0; f < number_of_features; f++)
		hidden_layers[0]->get_input_features()[f] = normalized_input_features[f];

	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->compute_activation_array();

	// output layer will calculate a singular value and return that value as the result
	output_layer->compute_activation_array();

	return *(output_layer->get_activation_array());
}

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

void NeuralNetwork::set_learning_rate(double new_learning_rate)
{ *learning_rate = new_learning_rate; }
void NeuralNetwork::set_regularization_rate(double new_regularization_rate)
{ *regularization_rate = new_regularization_rate; }
void NeuralNetwork::set_patience(int new_patience)
{ patience = new_patience; }
void NeuralNetwork::set_number_of_epochs(int new_number_of_epochs)
{ number_of_epochs = new_number_of_epochs; }
void NeuralNetwork::set_batch_size(int new_batch_size)
{
	batch_size = new_batch_size;
	update_arrays_using_batch_size();
}