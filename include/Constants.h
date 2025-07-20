#pragma once

namespace Constants
{
	// used in bn formulas
	constexpr double epsilon = 1e-5;
	
	// used in reversing log-transformed values
	constexpr double euler_number = 2.718281828459045235360287471352;
	
	// limits the number of possible folds the user can enter
	constexpr int max_number_of_folds = 20;
	constexpr int min_number_of_folds = 5;
	
	// controls how much space the border line takes
	constexpr int width = 50;
	
	// number of times the randomize samples function will randomize
	constexpr int number_of_random_shuffles = 50;

	// constant components for training
	const double decay_rate = 0.10;
	const double explosion_max = 1e30;
}