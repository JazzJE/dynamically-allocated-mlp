# Dynamically Allocated Multi-layer Perceptron

Simple multi-layer perceptron made from scratch that does not use C++'s vector library, and instead opts to function via dynamic memory allocation w/ raw pointers. This project is more of a demonstration of my personal interest in ML, C++, and math in general than something to be used seriously in production (I love pointers a lot).

## Table of Contents
- [Original Dataset](#original-dataset)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Architecture Details](#architecture-details)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Original Dataset
This project uses a transformed version of the [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) dataset from Kaggle:
Nugent, C. (2017). California Housing Prices. Kaggle.

License: CC0 1.0 (Public Domain).

## Features
The program does all of the following automatically within the program w/o editing the original data set file prior to menu prompts:

- Normalizes all training data's features
- Log-transforms target values
- Generates files for the neural network that allow it to be saved
- Validates all associated files that will be used within the program (data set and neural network files)
- Randomizes the order of all samples

Other runtime features of the program:
- Testing hyperparameters via k-fold training and early-stopping
- Predicting values using random sample features or user-inputted sample features
- Creation of simple log files which describe hyperparameters used in a given training session

## Setup

### Requirements
- C++ 17 or higher
- OpenMP support (required for parallel processing)
- CMake 3.15 or higher
- A compatible C++ compiler (GCC, Clang, or MSVC)

### Setup Guide
1. Clone this repo
   ```bash
   git clone https://github.com/JazzJE/dynamically-allocated-mlp
   cd dynamically-allocated-mlp
   ```

2. Build the project
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## Usage

### Generating a new MLP

**Option 1: Retrain with existing structure**
- If you would like to retrain a neural network from scratch and still use the given `number_of_neurons_each_hidden_layer` array, simply delete the `nn_current_state` directory in the root folder

**Option 2: Change MLP structure**
1. Go to `main.cpp` and edit the `number_of_neurons_each_hidden_layer` array
   - This array should only have positive integers
   - There should be at least one integer within the array
   - Example: `{ 256, 128, 32 }` creates:
     - 256 neurons in the first layer
     - 128 neurons in the second layer
     - 32 neurons in the third layer
     - 1 neuron in the output layer (implicitly created when program is run)

2. Recompile the program:
   ```bash
   cd build
   cmake --build .
   ```

3. When you next run the program, select 'Y' for all prompts to generate new save files

### CSV Data Set Requirements

1. **File naming**: Rename your dataset to `dataset.csv` and place it in the root directory
2. **Structure requirements**:
   - Last column must contain target values
        - All target values cannot be negative or equal to zero due to log transformation of target values
   - First line must contain feature names
   - All values (except feature names) must be integers or doubles
        - Boolean values should be 0s or 1s, not "True"/"False"
        - No strings or characters allowed
3. **Feature normalization**: To skip normalization for certain features, prefix the feature name in the data set with "\~" (e.g., "\~is_coastal_luxury"; refer to initial `dataset.csv` feature column names)
4. **MLP Regeneration**: Delete the `nn_current_state` directory to retrain the MLP on the new data set from scratch and get rid of any old values

### Training Logs
      
1. All logs, when the save selection is chosen in the program, are saved to the `training_logs` directory found in the root folder
    - The program will automatically generate the directory if it does not exist at its start
    - It will not parse existent logs inside of the directory to the program for printing

### Running the Program

Execute from `build` directory:
```bash
./neural_network.exe
```

### Runtime Parameters

- **Number of folds**: How many different sections of samples you want, where each section will be used as a cross-validation set
- **Batch size**: Number of samples loaded per training epoch
- **Learning rate**: Controls how fast the MLP changes with respect to loss
- **Regularization rate**: Adjustment factor for better generalization to new samples
- **Patience**: Number of failed training epochs before ending training on a fold
- **Number of epochs**: Number of epochs before stopping training

## Architecture Details

- **Activation Functions**: ReLU for hidden layers, linear for output
- **Optimization**: Mini-batch gradient descent
- **Normalization**: Z-score feature normalization, BatchNorm between layers
- **Regularization**: L2 regularization
- **Loss Function**: Mean Squared Error
- **Cross-Validation**: k-fold with early stopping
- **Learning Rate Scheduling**: ReudceLROnPlateau

## Performance

### Expected Results
The MLP typically achieves decent performance on the California Housing dataset. Training time depends on:
- Network architecture (number of layers/neurons)
- Batch size
- Learning rate
- Dataset size

### Memory Usage
This implementation uses dynamic memory allocation with raw pointers, which provides:
- Fine-grained memory control
- Educational insight into memory management
- Potential performance benefits for experienced users

**Note**: For production use, consider using `std::vector` or other modern C++ containers for better safety and maintainability.

## Troubleshooting

### Common Issues

**Program exits with file validation errors**
- Ensure `dataset.csv` is in the root directory
- Check that the CSVs follows the format requirements; ensure to check the line in which the error was found
- Verify neural network files aren't corrupted or erroneously generated (delete `nn_current_state` folder to regenerate)

**Compilation errors**
- Ensure you have C++17 or higher
- Verify OpenMP is available on your system
- Check that CMake version is 3.15 or higher

**Memory-related crashes**
- This is a learning project using raw pointers - such issues are expected
- Reduce batch size if experiencing memory pressure
- Consider the network architecture size relative to available RAM

**Training convergence issues**
- Adjust learning rate (try values between 0.0001 and 0.1)
- Modify regularization rate
- Increase patience for more training epochs

### Getting Help

If you encounter issues:
1. Check that your dataset meets all requirements
2. Verify the build process completed successfully
3. Try regenerating neural network files by deleting the `nn_current_state` directory

## Contributing

This is primarily an educational project demonstrating C++ memory management and neural network implementation. While not intended for production use, contributions that improve the educational value or fix critical issues are welcome.

### Areas for potential improvement:
- More robust error handling
- Performance optimizations
- Better documentation, review, or fixes for the mathematical operations and backpropagation formulas 