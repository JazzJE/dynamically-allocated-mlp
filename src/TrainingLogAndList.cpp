#include "TrainingLogAndList.h"
#include "Constants.h"
#include "MenuFunctions.h"

TrainingLog::TrainingLog(std::string session_name, bool using_all_samples, double learning_rate, double regularization_rate, 
	int patience, int number_of_epochs, double* best_mse_for_each_fold, int number_of_folds, int batch_size) :
	using_all_samples(using_all_samples), learning_rate(learning_rate), regularization_rate(regularization_rate), patience(patience), 
	number_of_epochs(number_of_epochs), best_mse_for_each_fold(new double[number_of_folds]), number_of_folds(number_of_folds), 
	next_log(nullptr), session_name(session_name), batch_size(batch_size)
{ 
	for (int f = 0; f < number_of_folds; f++)
		this->best_mse_for_each_fold[f] = best_mse_for_each_fold[f];
}

TrainingLog::~TrainingLog()
{ delete[] best_mse_for_each_fold; }

void TrainingLog::print_training_log()
{
	std::cout << "\n\tTraining configs for " << session_name << ":"

		<< "\n\n\t\tTraining option: ";
	using_all_samples ? std::cout << "all samples" : std::cout << "k-folded samples";
	std::cout << "\n\t\tInitial learning rate - " << learning_rate
		<< "\n\t\tRegularization rate - " << regularization_rate
		<< "\n\t\tPatience - " << patience
		<< "\n\t\tNumber of epochs - " << number_of_epochs
		<< "\n\t\tNumber of folds - " << number_of_folds
		<< "\n\t\tBatch size - " << batch_size;
	std::cout << "\n";

	std::cout << "\n\tBest MSEs for each fold:";
	std::cout << "\n";

	for (int i = 0; i < number_of_folds; i++)
		std::cout << "\n\t\tFold #" << i + 1 << " - " << best_mse_for_each_fold[i];
}

TrainingLogList::TrainingLogList(std::filesystem::path training_logs_file_path) : head(nullptr), 
training_logs_file_path(training_logs_file_path)
{ }

// delete the linked list
TrainingLogList::~TrainingLogList()
{
	TrainingLog* curr = head;
	TrainingLog* next;

	while (curr != nullptr)
	{
		next = curr->next_log;
		delete curr;
		curr = next;
	}
}

// add a new TrainingLog object to the LinkedList
void TrainingLogList::add_training_log(TrainingLog* new_training_log)
{
	if (head == nullptr)
		head = new_training_log;
	else
	{
		TrainingLog* curr = head;

		while (curr->next_log != nullptr)
			curr = curr->next_log;

		curr->next_log = new_training_log;
	}
}

// print each training log to the terminal
void TrainingLogList::print_all_training_logs()
{
	if (head == nullptr)
		std::cout << "\n\tNo logs currently available!\n";

	else
	{
		int current_log_number = 1;
		TrainingLog* curr = head;
		while (curr != nullptr)
		{
			generate_border_line();
			std::cout << "\n\tLog for training iteration #" << current_log_number << " (" << curr->session_name << "):";
			std::cout << "\n";
			curr->print_training_log();
			std::cout << "\n";
			generate_border_line();

			current_log_number++;
			curr = curr->next_log;
		}
	}
}

// save the training logs to different text files within the training logs file directory
void TrainingLogList::save_training_logs()
{
	if (head == nullptr)
		std::cout << "\n\tNo logs to save!\n";

	else
	{
		std::cout << "\n\tSaved training logs!\n";

		TrainingLog* curr = head;
		while (curr != nullptr)
		{
			std::fstream session_file(training_logs_file_path / (curr->session_name + ".txt"), std::ios::out | std::ios::trunc);

			session_file << "\n\tTraining configs for " << curr->session_name << ":"

				<< "\n\n\t\tTraining option: ";
			curr->using_all_samples ? session_file << "all samples" : session_file << "k-folded samples";
			session_file << "\n\t\tInitial learning rate - " << curr->learning_rate
				<< "\n\t\tRegularization rate - " << curr->regularization_rate
				<< "\n\t\tPatience - " << curr->patience
				<< "\n\t\tNumber of epochs - " << curr->number_of_epochs
				<< "\n\t\tNumber of folds - " << curr->number_of_folds
				<< "\n\t\tBatch size - " << curr->batch_size;
			session_file << "\n";

			session_file << "\n\tBest MSEs for each fold:";
			session_file << "\n";

			for (int i = 0; i < curr->number_of_folds; i++)
				session_file << "\n\t\tFold #" << i + 1 << " - " << curr->best_mse_for_each_fold[i];

			curr = curr->next_log;
		}
	}
}