# Cong Tran 
# 1002046419

import numpy as np
import os
import sys
from nn_load import *
from perceptron_inference_solution import perceptron_inference

# Specify parameters for a test case.
weights_file = "weights2.txt"
input_file = "input2a.txt"

#activation_string = "step"
activation_string = "sigmoid"

# Read the weights of the perceptron.
weights = read_matrix(weights_file)
b = weights[0,0]
w = weights[1:, :]

# Read the input vector.
input_vector = read_matrix(input_file)

# The next line is where your function is called.
(a, z) = perceptron_inference(b, w, activation_string, input_vector)

# Print the results.
print("a = %.4f\nz = %.4f" % (a, z))


# MY CODE

#array of tuples with the the cases
# test_case = [
#     ("weights1.txt", "input1_00.txt", "step"), 
#     ("weights1.txt", "input1_11.txt", "step"), 
#     ("weights1.txt", "input1_11.txt", "sigmoid"), 
#     ("weights2.txt", "input2a.txt", "sigmoid"), 
# ]
# print()

# for w_file, i_file, act_function in test_case: 
#     weights = read_matrix(w_file)
#     b = weights[0,0]
#     w = weights[1:, :]
#     input_vector = read_matrix(i_file)
#     (a, z) = perceptron_inference(b, w, act_function, input_vector)
#     print("a = %.4f\nz = %.4f" % (a, z))
#     print()
