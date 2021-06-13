
# imports
import numpy as np
import math

def activation(activationName, matrix):

    if (activationName == "elu"):

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] < 0):
                    matrix[i][j] = 0.2*(np.exp(matrix[i][j])-1)
        return matrix


    elif (activationName == "relu"):

        if (matrix.min() >= 0):
            return matrix
        else:
            return np.maximum(0, matrix)


    elif (activationName == "sigmoid"):

        return 1.0 / (1 + np.exp(-matrix))


    elif (activationName == "tanh"):

        matrix = matrix - np.max(matrix, axis=1, keepdims=True)
        return (np.exp(matrix) - np.exp(-matrix)) / (np.exp(matrix) + np.exp(-matrix))


    elif (activationName == "softmax"):

        matrix = matrix.T
        exp_value = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        return prob.T




def derivative_activation(activationName, matrix):

    if (activationName == "elu"):

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] <= 0):
                    matrix[i][j] = 0.2 * np.exp(matrix[i][j])
                else:
                    matrix[i][j] = 1

        return matrix


    elif (activationName == "relu"):
        matrix[matrix <= 0] = 0
        matrix[matrix > 0] = 1

        return matrix


    elif (activationName == "sigmoid"):

        return activation(activationName, matrix) * (1-activation(activationName, matrix))


    elif (activationName == "tanh"):

        return 1 - np.power(activation("tanh",matrix), 2)


    elif (activationName == "softmax"):

        matrix = matrix.T
        exp_value = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        prob = exp_value / np.sum(exp_value, axis=1, keepdims=True)

        return prob.T