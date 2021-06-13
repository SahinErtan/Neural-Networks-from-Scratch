import math
import numpy as np

class Optimizer():

    def __init__(self, optimizer, learningRate):

        self.optimizer = optimizer.lower()
        self.learningRate = learningRate

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon1 = 1e10-6
        self.epsilon2 = 1e10-8
        self.M1, self.M2, self.M3 = 0,0,0
        self.V1, self.V2, self.V3 = 0, 0, 0
        self.M = {1:self.M1, 2:self.M2, 3:self.M3}
        self.V = {1:self.V1, 2:self.V2, 3:self.V3}

    def optimizer_momentum(self):

        self.M[self.num] = self.beta1 * self.M[self.num] + (1-self.beta1) * self.grad

        self.w = self.w - self.learningRate * self.M[self.num]

        return self.w

    def optimizer_RMSProb(self):

        self.V[self.num] = (self.beta1 * self.V[self.num]) + (1 - self.beta1) * self.grad^2

        self.w = self.w - (self.learningRate / (math.sqrt(self.V[self.num] + self.epsilon1))) * self.grad

        return self.w

    def optimizer_Adam(self):

        self.M[self.num] = self.beta1 * self.M[self.num] + (1 - self.beta1) * self.grad

        self.V[self.num] = (self.beta1 * self.V[self.num]) + (1 - self.beta1) * self.grad ^ 2

        self.M[self.num] = self.M[self.num] / (1 - self.epsilon2)

        self.V[self.num] = self.V[self.num] / (1 - self.beta2)

        self.w = self.w - ( self.learningRate / ((math.sqrt(self.V[self.num]) + self.epsilon2) * self.M[self.num] ) )


    def optimize(self, num, w, grad):

        self.num = num
        self.w = w
        self.grad = grad

        if (self.optimizer == "momentum"):
            return self.optimizer_momentum()
        elif(self.optimizer == "rmsprob"):
            return self.optimizer_momentum()
        elif (self.optimizer == "adam"):
            return self.optimizer_momentum()

