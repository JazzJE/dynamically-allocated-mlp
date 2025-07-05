#include "StatisticsFunctions.h"

constexpr double epsilon = 1e-5;

// calculating means of a provided range

// the lower_validation_index and higher_validation_index parameters refer to the training sample of the lower range of the 
// cross-validation set and the higher range of the cross-validation set; this is so the cross-validation set can be skipped 
// over when calculating the means of the training set

double* calculate_features_means(double** sample_features, bool* not_normalize, int number_of_features, int number_of_training_samples,
	int lower_validation_index, int higher_validation_index)
{
	double* means_array = new double[number_of_features]();

	for (int t = 0; t < number_of_training_samples; t++)
	{
		// skip over the cross-validation set when calculating the features of the network
		if (t == lower_validation_index)
		{
			t = higher_validation_index;
			continue;
		}
		
		for (int f = 0; f < number_of_features; f++)
		{
			// skip over calculating the mean of this feature if the program identified the value to not be normalized
			if (not_normalize[f]) continue;

			means_array[f] += sample_features[t][f];
		}
	}

	// the number of test samples is equal to the number of samples minus 1,
	// minus the difference between the high and low cross validation indices
	int number_of_test_samples = number_of_training_samples - 1 - (higher_validation_index - lower_validation_index);
	for (int f = 0; f < number_of_features; f++)
		means_array[f] /= number_of_test_samples;

	return means_array;
}

// calculating variances of a provided range

// the lower_validation_index and higher_validation_index parameters refer to the training sample of that is the 
// lower boundary of the cross-validation set and the higher boundary of the cross-validation set, such that the cross-validation
// set can be skipped over when calculating the variances of the training set

double* calculate_features_variances(double** sample_features, bool* not_normalize, double* features_means, int number_of_features,
	int number_of_training_samples, int lower_validation_index, int higher_validation_index)
{
	double* variances_array = new double[number_of_features]();

	for (int t = 0; t < number_of_training_samples; t++)
	{
		// skip over the cross-validation set when calculating the variances of the network
		if (t == lower_validation_index)
		{
			t = higher_validation_index;
			continue;
		}

		for (int f = 0; f < number_of_features; f++)
		{
			// skip calculating the variance for this feature if the program identified it to not be normalized
			if (not_normalize[f]) continue;

			variances_array[f] += pow(sample_features[t][f] - features_means[f], 2.0);
		}
	}

	// the number of test samples is equal to the number of samples minus 1,
	// minus the difference between the high and low cross validation indices
	int number_of_test_samples = number_of_training_samples - 1 - (higher_validation_index - lower_validation_index);
	for (int f = 0; f < number_of_features; f++)
		variances_array[f] /= number_of_test_samples;

	return variances_array;
}

// normalizing features in a new dynamically allocated array
// this is used for when given multiple samples of features
double** calculate_normalized_features(double** sample_features, bool* not_normalize, int number_of_samples, int number_of_features, double* means_array,
	double* variances_array)
{
	double** normalized_features = allocate_memory_for_2D_array(number_of_samples, number_of_features);

	for (int t = 0; t < number_of_samples; t++)
		for (int f = 0; f < number_of_features; f++)
		{
			// skip over normalizing this feature if identified as a feature of a column of values to not be normalized
			if (not_normalize[f])
			{
				normalized_features[t][f] = sample_features[t][f];
				continue;
			}

			normalized_features[t][f] = ((sample_features[t][f] - means_array[f]) / (sqrt(variances_array[f] + epsilon)));
		}

	return normalized_features;
}

// this is used for calculating normalized features for a single sample of features
double* calculate_normalized_features(double* sample_features, bool* not_normalize, int number_of_features, double* means_array, double* variances_array)
{
	double* normalized_features = new double[number_of_features];

	for (int f = 0; f < number_of_features; f++)
	{
		// skip over normalizing this feature if identified as column of binary values
		if (not_normalize[f])
		{
			normalized_features[f] = sample_features[f];
			continue;
		}

		normalized_features[f] = ((sample_features[f] - means_array[f]) / (sqrt(variances_array[f] + epsilon)));
	}

	return normalized_features;
}