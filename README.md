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

* A list of the ML configs I hard-coded this MLP with
    - Mini-batch gradient descent
    - Batch normalization
    - ReLU activation functions

## Installation and Setup



## Usage

* _Rules for the CSV data set that will be imported into this MLP_
    - Rename the data set to "dataset.csv" and drop it into the root directory (where main.cpp automatically is)
    - **The last column of the CSV are the target values** you would like to predict
    - The first line must contain the names of each feature
        - **If you would like certain features to be ignored during the normalization of features, please make sure to place a "\~" before the feature name** (i.e., if I have a one-hot encoded feature name of "is_coastal_luxury," then I would rename this feature within my data set to "\~is_coastal_luxury"; refer to first line of the CSV data set initially within this program for reference)
    - After the first line of column feature name titles, every value within the data set must be an integer or double value
        - This includes boolean values (represent them as 0s or 1s, not "True" or "False")
        - No characters or strings are allowed
 
* Before running the program, whenever you to change the number of neurons for each layer or the structure of the mlp, go to main.cpp and edit the number_of_neurons_each_hidden_layer array
    - This array should only have positive integers and there should be at least one integer within the array
    - If an array of { 256, 128, 32 } is provided, then there will be...
        - 256 neurons in the first layer
        - 128 neurons in the second layer
        - 32 neurons in the third layer
        - 1 neuron in the output layer, which is implicitly created whenever the program is run, and which will output the result
    - Then, recompile the program with the following commands
        ```
        mkdir build         // make this directory if not made already
        cd build
        cmake ..            // only run this command once after cloning the project
        cmake --build .
        ```

* Run the program
    ```
    ./neural_network.exe    // from the root directory
    ```
    - If the neural network files or the data set files have errors in them, the program will end and ask you to fix them before interacting further
        - However, for the neural network files, the program will ask if they would also like to generate new ones instead; please do this if you would like to generate a new MLP that has a different structure of neurons or the data set has changed
    - This is what every runtime parameter does
        - **Batch size**: choose how many samples will be loaded for each training epoch
            - Updating this will regenerate the entire MLP due to const qualifiers within the network's layers' inputs and outputs, but it will maintain the same training options (learning rate, regularization rate, patience, and iteration prompt)
        - **Learning rate**: how fast the MLP should change in respect to the loss
        - **Regularization rate**: how much the MLP should adjust itself to better predict new samples
        - **Patience**: how many failed training epochs before ending training on a fold 
        - **Prompt epoch**: how many epochs should pass before prompting you to stop training on a given fold; this is for if the training has gone on for really long on one due to slow covergence or large patience values
