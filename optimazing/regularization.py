import numpy as np


def regularization_L1(w,regularizationRate):

    weights = w.copy()
    weights[w < 0] = regularizationRate
    weights[w > 0 ] = - regularizationRate

    return weights


def regularization_L2(w,regularizationRate):

    return -2*regularizationRate*w


def regularize(w,regularization):

    if(regularization[0]== "None"):
        return 0
    elif(regularization[0]== "L1") or (regularization[0]== "Regularization_L1"):
        return regularization_L1(w, regularization[1])
    elif (regularization[0] == "L2") or (regularization[0] == "Regularization_L2"):
        return regularization_L2(w, regularization[1])

