# Single Perceptron Inference

This repository provides a simple implementation of a single-layer perceptron for inference. The perceptron can use either a step or sigmoid activation function to compute outputs based on the given weights, bias, and input vector.

## Features

- **Activation Functions**: Supports `step` and `sigmoid` activation functions.
- **Core Functions**:
  - **`perceptron_inference`**: Computes the perceptron output using weights, bias, input vector, and activation function.
  - **`activation_function`**: Applies the specified activation function to the computed value.
  - **`step_function`**: Step activation function.
  - **`sigmoid_function`**: Sigmoid activation function.

## Usage

1. **Inputs**:
   - `b`: Bias weight (float).
   - `w`: Column vector of weights (2D numpy array with a single column).
   - `activation`: String specifying the activation function (`"step"` or `"sigmoid"`).
   - `input_vector`: Column vector of inputs (2D numpy array with a single column).

2. **Output**:
   - A tuple `(a, z)` where:
     - `a` is the linear combination of weights, input, and bias.
     - `z` is the final output after applying the activation function.
