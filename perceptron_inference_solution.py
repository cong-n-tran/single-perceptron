# Cong Tran 
# 1002046419

import numpy as np

# b: a real number (float) specifying the bias weight for the perceptron.
# w: a column vector specifying the weights w of the perceptron. It is a 2D numpy array with a single column.
# activation: it is a string that specifies the activation function. The value is either "step" or "sigmoid".
# input_vector: a column vector specifying the input to the perceptron. It is a 2D numpy array with a single column.

def perceptron_inference(b: float, w: any, activation: str, input_vector: any) -> tuple:
    # transpose the weight
    w_T = np.transpose(w)
    
    # perform linear combination
    a = np.dot(w_T, input_vector) + b

    # apply the activation function
    z = activation_function(a, activation=activation)

    #return the tuple
    return (a, z)
def activation_function(a: float, activation: str)-> float: 
    # either is one or zero 
    if activation == 'step': 
        return step_function(a)
    # 1 / (1 + np.exp(- a))
    elif activation == 'sigmoid': 
        return sigmoid_function(a)
    else: 
        print('activation string is incorrect')
    return None

def step_function(a: float): 
    return 1 if a >= 0 else 0
    
def sigmoid_function(a: float): 
    return 1 / (1 + np.exp(- a))
    

# a: a real number (float), equal to the result of step 1 (dot product weights and input, plus bias).
# z: a real number (float), equal to the final output of the perceptron, obtained by applying the activation function to the result of step 1.