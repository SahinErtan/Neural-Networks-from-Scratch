import os
import pandas


def prepare_Data():

    workingPath = os.getcwd()

    csvPath = os.path.dirname(workingPath) + "/data/splitDataset/"

    train_data = pandas.read_csv(csvPath + "trainData.csv")

    validation_data = pandas.read_csv(csvPath + "validationData.csv")

    test_data = pandas.read_csv(csvPath + "testData.csv")

    train_data_input = train_data.loc[:," timedelta":" shares"].to_numpy()
    train_data_output = train_data.loc[:,"label1":"label7"].to_numpy()

    validation_data_input = validation_data.loc[:, " timedelta":" shares"].to_numpy()
    validation_data_output = validation_data.loc[:, "label1":"label7"].to_numpy()

    test_data_input = test_data.loc[:, " timedelta":" shares"].to_numpy()
    test_data_output = test_data.loc[:, "label1":"label7"].to_numpy()

    return train_data_input, train_data_output, validation_data_input, validation_data_output, test_data_input,test_data_output
