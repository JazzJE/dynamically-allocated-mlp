# Dynamically Allocated Multi-layer Perceptron

Simple multi-layer perceptron made from scratch that does not use C++'s vector library, and instead opts to function via dynamic memory allocation w/ raw pointers. This project is more of a demonstration of my interest in ML, C++, and math in general than something to be used seriously in production; I just love pointers a lot.

## Features

* The program does all of the following automatically within the program w/o editing the original data set file prior to menu prompts
    - Normalizes all training data's features
    - Log-transforms target values
    - Randomizes the order of all samples
    - Generates weights, biases, running means, running variances, scales, and shifts files that allow you to save the state of the neural network if the files do not exist, if the data set file has changed, or if the structure of the MLP has changed
    - Validates all associated files that will be used within the program (data set and neural network files)

* Other runtime features of the program
    - Saving the current state of the neural network for future use via the aforementioned neural network files
    - Training the neural network via five-fold training and early-stopping with customizable options
    - Predicting values using random sample features or user-inputted sample features

## Setup

### Requirements

- C++ 17 or higher
- OpenMP support (required for parallel processing)

### Setup Guide

1. Clone this repo
    ```
    git clone https://github.com/JazzJE/dynamically-allocated-mlp
    ```
2. Go into the root directory (where main.cpp is) and run these commands
    ```
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ```

## Usage

* Generating a new MLP
    - If you would like to retrain a nerual network from scratch and still use the given number_of_neurons_each_hidden_layer array, then simply delete the "nn_current_state" directory in the root folder  
    - Otherwise, _whenever you to change the number of neurons for each layer or the structure of the MLP_, follow these steps
        1. Go to main.cpp and edit the number_of_neurons_each_hidden_layer array
            1. This array should only have positive integers, and there should be at least one integer within the array
            2. If an array of { 256, 128, 32 } is provided for number_of_neurons_each_hidden_layer, then there will be...
                * 256 neurons in the first layer
                * 128 neurons in the second layer
                * 32 neurons in the third layer
                *  1 neuron in the output layer, which is implicitly created whenever the program is run, and which will output the result
        2. Then, recompile the program with the following commands from the root directory; **you must do this every time the array is edited, else the program will not be updated next execution**
            ```
            cd build
            cmake --build .
            ```
        3. When you next run the program, it will prompt you that the weights and biases, the scales and shifts, and the means and vars files are all erroneous, but simply select 'Y' for all options to generate new save files

* _Rules for the CSV data set_ that will be imported into this MLP
    1. **Rename the data set to "dataset.csv"** and drop it into the root directory (where main.cpp automatically is)
    2. **The last column of the CSV should be the target values** you would like to predict
    3. **The first line must contain the names of each feature**
        1. **If you would like certain features to be ignored during the normalization of features, please make sure to place a "\~" before the feature name** (i.e., if I have a one-hot encoded feature name of "is_coastal_luxury," then I would rename this feature within my data set to "\~is_coastal_luxury"; refer to first line of the CSV data set initially within this program for reference)
    4. After the first line of column feature name titles, **every value within the data set must be an integer or double value**
        1. This includes boolean values (represent them as 0s or 1s, not "True" or "False")
        2. No characters or strings are allowed
    5. If the data set changes in its number of features, then the MLP will ask you to regenerate all neural network files next execution; do so if you would like to retrain the neural network from scratch, else remove the new data

* Run the program from the root directory, _NOT_ the build directory
    ```
    ./neural_network.exe
    ```
    - If the neural network files or the data set files have errors in them, the program will end and ask you to fix them before interacting further
        - However, for the neural network files, the program will ask if they would also like to generate new ones instead; please do this if you would like to generate a new MLP that has a different structure of neurons or the data set has changed in number of features
    - Description of every runtime parameter
        - **Batch size**: choose how many samples will be loaded for each training epoch
            - Updating this will regenerate the entire MLP due to const qualifiers within the network's layers' inputs and outputs, but it will maintain the same training options (learning rate, regularization rate, patience, and iteration prompt)
        - **Learning rate**: how fast the MLP should change in respect to the loss
        - **Regularization rate**: how much the MLP should adjust itself to better predict new samples
        - **Patience**: how many failed training epochs before ending training on a fold 
        - **Prompt epoch**: how many epochs should pass before prompting you to stop training on a given fold; this is for if the training has gone on for really long on a given fold due to slow covergence or large patience values so that the neural network can still be saved

## Architecture Details

- **Activation Functions**: ReLU for hidden layers, linear for output
- **Optimization**: Mini-batch gradient descent
- **Normalization**: Z-score feature normalization, BatchNorm between layers 
- **Regularization**: L2 regularization
- **Loss Function**: Mean Squared Error
