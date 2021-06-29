import os
import time
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from preprocess.prepareData import prepare_Data
from training import activationFuncs
from optimazing.regularization import regularize
from optimazing.optimizer import Optimizer




class Net():

    def __init__(self, layers, load_mode=False):

        workingPath = os.getcwd()
        self.workingPath = os.path.dirname(workingPath)

        self.input_layer = layers[0]
        self.hidden_layer1 = layers[1]
        self.hidden_layer2 = layers[2]
        self.output_layer = layers[3]
        self.load_mode = load_mode

        self.train_loss_values = []
        self.val_loss_values = []

        self.train_accuracy_values = []
        self.val_accuracy_values = []

        if(load_mode==False):
            self.initialize_wb()
        elif(load_mode==True):
            
            self.w1 = np.load(self.workingPath + "/data/parameters/w1.npy")
            self.w2 = np.load(self.workingPath + "/data/parameters/w2.npy")
            self.w3 = np.load(self.workingPath + "/data/parameters/w3.npy")
            self.b1 = np.load(self.workingPath + "/data/parameters/b1.npy")
            self.b2 = np.load(self.workingPath + "/data/parameters/b2.npy")
            self.b3 = np.load(self.workingPath + "/data/parameters/b3.npy")

        print(layers)


    def initialize_wb(self):

        if (self.load_mode == True):
            # LOAD from params weights and bias
            pass
        else:
            # Normal Distribution
            # self.w1 = np.random.randn(self.hidden_layer1[0], self.input_layer[0])
            # self.w2 = np.random.randn(self.hidden_layer2[0], self.hidden_layer1[0])
            # self.w3 = np.random.randn(self.output_layer[0], self.hidden_layer2[0])
            # self.b1 = np.random.randn(self.hidden_layer1[0], 1)
            # self.b2 = np.random.randn(self.hidden_layer2[0], 1)
            # self.b3 = np.random.randn(self.output_layer[0], 1)
            
            # Uniorm Distribution
            self.w1 = np.random.uniform(0,1,(self.hidden_layer1[0], self.input_layer[0]))
            self.w2 = np.random.uniform(0,1,(self.hidden_layer2[0], self.hidden_layer1[0]))
            self.w3 = np.random.uniform(0,1,(self.output_layer[0], self.hidden_layer2[0]))
            self.b1 = np.random.uniform(0,1,(self.hidden_layer1[0], 1))
            self.b2 = np.random.uniform(0, 1, (self.hidden_layer2[0], 1))
            self.b3 = np.random.uniform(0, 1, (self.output_layer[0], 1))




    def feedforward(self, x_data):      # matrix carpimi
        print("ileri besleme")

        self.a1 = x_data.T
        self.z2 = self.w1.dot(self.a1) + self.b1
        self.a2 = activationFuncs.activation(self.hidden_layer1[1], self.z2)
        self.z3 = self.w2.dot(self.a2) + self.b2
        self.a3 = activationFuncs.activation(self.hidden_layer2[1], self.z3)
        self.z4 = self.w3.dot(self.a3) + self.b3
        self.a4 = activationFuncs.activation(self.output_layer[1], self.z4)


    def calculate_accuracy(self,y_data):

        success = 0
        pred_i = 0
        total = len(y_data)
        hatalar = []
        for id, arr in enumerate(self.a4.T):
            memoryArr = 0
            for i in range(len(arr)):
                if(arr[i]>memoryArr):
                    memoryArr = arr[i]
                    pred_i = i
            if(y_data[id][pred_i] == 1):
                success+=1

        print(f"{success} basarili tahmin, {total} deger icinden")

        return (success * 100) / total

    def calculate_loss(self,y_data):

        self.calculate_accuracy(y_data)
        sum_score = 0.0
        self.actual = y_data
        self.predicted = self.a4.T
        for i in range(len(self.actual)):
            for j in range(len(self.actual[i])):
                sum_score += self.actual[i][j] * np.log(1e-15 + self.predicted[i][j])
        mean_sum_score = 1.0 / len(self.actual) * sum_score
        print(f"loss = {-mean_sum_score}")
        return -mean_sum_score


    def backpropagation(self, y_data):  # element wise carpım
        print("geri besleme")

        self.calculate_loss(y_data)

        if(self.output_layer[1] == "softmax"):
            self.dz4 = self.a4 - y_data.T
        else:
            self.dz4 = (self.a4 - y_data.T) * activationFuncs.derivative_activation(self.output_layer[1],self.z4)


        self.da3 = activationFuncs.derivative_activation(self.hidden_layer2[1], self.z3)
        self.dz3 = self.w3.T.dot(self.dz4) * activationFuncs.derivative_activation(self.hidden_layer1[1], self.z3)
        self.da2 = activationFuncs.derivative_activation(self.hidden_layer1[1], self.z2)
        self.dz2 = self.w2.T.dot(self.dz3) * self.da2

        self.grad4 = self.dz4.dot(self.a3.T)
        self.grad3 = self.dz3.dot(self.a2.T)
        self.grad2 = self.dz2.dot(self.a1.T)


        if(self.optimizer == "None"):

            self.w3 = self.w3 - (self.learningRate * (self.grad4 + regularize(self.w3, self.regularization)))
            self.w2 = self.w2 - (self.learningRate * (self.grad3 + regularize(self.w2, self.regularization)))
            self.w1 = self.w1 - (self.learningRate * (self.grad2 + regularize(self.w1, self.regularization)))

        else:

            self.w3 = self.opt.optimize(3, self.w3, self.grad4)
            self.w2 = self.opt.optimize(2, self.w2, self.grad3)
            self.w1 = self.opt.optimize(1, self.w1, self.grad2)


        self.b3 = (self.b3.T - (self.learningRate * np.sum(self.dz4, axis=1))).T
        self.b2 = (self.b2.T - (self.learningRate * np.sum(self.dz3, axis=1))).T
        self.b1 = (self.b1.T - (self.learningRate * np.sum(self.dz2, axis=1))).T


    def shuffle_dataset(self, input_data, output_data):
        shuffler = np.random.permutation(len(input_data))
        shuffled_input = input_data[shuffler]
        shuffled_output = output_data[shuffler]
        return input_data, output_data


    def draw(self):

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(range(len(self.train_loss_values)), self.train_loss_values)
        axs[0, 0].set_title('Train Loss')
        axs[0, 1].plot(range(len(self.train_accuracy_values)), self.train_accuracy_values, 'tab:orange')
        axs[0, 1].set_title('Train Accuracy')
        axs[1, 0].plot(range(len(self.val_loss_values)), self.val_loss_values, 'tab:green')
        axs[1, 0].set_title('Validation Loss')
        axs[1, 1].plot(range(len(self.val_accuracy_values)), self.val_accuracy_values, 'tab:red')
        axs[1, 1].set_title('Validation Accuracy')

        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')


        plt.waitforbuttonpress()


    def predict(self,input_data):
        self.feedforward(input_data)
        return self.a4

    def test(self,input_data, output_data):
        self.feedforward(input_data)
        loss = self.calculate_loss(output_data)
        accuracy = self.calculate_accuracy(output_data)
        print(f"test hatasi = {loss}")
        return loss, accuracy

    def train(self, input_data, output_data,val_input_data,val_output_data, epoch, minibatch, learningRate = 0.001, regularization = ("None",0),
              optimizer = "None", shuffledMode = False): # lr = 0.0001

        sTime = time.time()

        self.learningRate = learningRate
        self.regularization = regularization
        self.optimizer = optimizer

        if (optimizer != "None"):
            self.opt = Optimizer(self.optimizer, self.learningRate)

        self.initialize_wb()



        for e in range(epoch):
            for mb in range(int(len(input_data)/minibatch)):
                print(f"{e} / {epoch} epoch , {mb} / {minibatch} minibatch")
                low_index = mb*minibatch
                high_index = (mb+1)*minibatch


                x_data = input_data[low_index:high_index,:]
                y_data = output_data[low_index:high_index,:]
                self.feedforward(x_data)
                self.backpropagation(y_data)

            loss, accuracy = self.test(input_data, output_data)
            self.train_loss_values.append(loss)
            self.train_accuracy_values.append(accuracy)

            loss, accuracy = self.test(val_input_data,val_output_data)
            self.val_loss_values.append(loss)
            self.val_accuracy_values.append(accuracy)

            if(shuffledMode == True):
                input_data, output_data = self.shuffle_dataset(input_data,output_data)

        print(self.val_loss_values)
        fTime = time.time()
        print(f"Egitim {fTime-sTime} saniyede bitti")
        print(f"Training Dataseti Icin Basari = % {self.train_accuracy_values[-1]}")
        print(f"Validation Dataseti Icin Basari = % {self.val_accuracy_values[-1]}")

        self.draw()



        np.save(self.workingPath+"/data/parameters/w1",self.w1)
        np.save(self.workingPath+"/data/parameters/w2",self.w2)
        np.save(self.workingPath+"/data/parameters/w3", self.w3)
        np.save(self.workingPath+"/data/parameters/b1", self.b1)
        np.save(self.workingPath+"/data/parameters/b2", self.b2)
        np.save(self.workingPath+"/data/parameters/b3", self.b3)


def main():


    print("Train basliyor")

    layers = [[42], [1200, "relu"], [600, "relu"], [5, "softmax"]] # relu relu softmax 42-120-60-5
    net = Net(layers)


    data =pandas.read_csv("C:\\Users\\ASUS\\PycharmProjects\\tensorflowEnvironment\\Deneme1\\CollectingPose\\poz_train.csv")
    input_data = data.iloc[:,3:45].to_numpy()
    output_data = data.iloc[:,45:50].to_numpy()

    data = pandas.read_csv("C:\\Users\\ASUS\\PycharmProjects\\tensorflowEnvironment\\Deneme1\\CollectingPose\\poz_val.csv")
    validation_data_input = data.iloc[:,3:45].to_numpy()
    validation_data_output = data.iloc[:,45:50].to_numpy()

    net.train(input_data, output_data, validation_data_input, validation_data_output, 50, 32, learningRate=0.0003,regularization=("None",0),
              shuffledMode=True, optimizer="None")  # 50 epoch 32 minibatch

    print("Test Datasi Hesaplaniyor")
    data = pandas.read_csv("C:\\Users\\ASUS\\PycharmProjects\\tensorflowEnvironment\\Deneme1\\CollectingPose\\poz_test.csv")
    test_data_input = data.iloc[:, 3:45].to_numpy()
    test_data_output = data.iloc[:, 45:50].to_numpy()
    net.test(test_data_input,test_data_output)




if __name__ == "__main__":
    main()
