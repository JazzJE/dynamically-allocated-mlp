#pragma once
#include <string>
#include <filesystem>

struct TrainingLog
{
	std::string session_name;
	bool using_all_samples;
	double learning_rate;
	double regularization_rate;
	int patience;
	int number_of_epochs;

	int number_of_folds;
	double* const best_mse_for_each_fold;

	TrainingLog* next_log;

	TrainingLog(std::string session_name, bool using_all_samples, double learning_rate, double regularization_rate, int patience, int number_of_epochs, 
		double* best_mse_for_each_fold, int number_of_folds);
	~TrainingLog();

	void print_training_log();

};

// linked list of all the training logs
class TrainingLogList
{
private:

	const std::filesystem::path training_logs_file_path;
	TrainingLog* head;

public:

	TrainingLogList(std::filesystem::path training_logs_file_path);
	~TrainingLogList();

	void add_training_log(TrainingLog* new_training_log);
	void print_all_training_logs();
	void save_training_logs();

};