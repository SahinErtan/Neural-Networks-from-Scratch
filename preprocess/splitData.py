import os
import pandas
import numpy as np



data1 = pandas.read_csv("5/pandasPoz.csv")

data2 = pandas.read_csv("6/pandasPoz.csv")

data3 = pandas.read_csv("7/pandasPoz.csv")

data4 = pandas.read_csv("8/pandasPoz.csv")

data5 = pandas.read_csv("9/pandasPoz.csv")

data1_input = data1.loc[:, "0":"41"].to_numpy()

data2_input = data1.loc[:, "0":"41"].to_numpy()

data3_input = data1.loc[:, "0":"41"].to_numpy()

data4_input = data1.loc[:, "0":"41"].to_numpy()

data5_input = data1.loc[:, "0":"41"].to_numpy()

data = np.vstack((data1,data2,data3,data4,data5))

encode_label = np.zeros((2194,5))

row = 0
memory = 0
for i in range(196):
    encode_label[i+row] = [1,0,0,0,0]
    memory = i
row = memory +1
for i in range(477):
    encode_label[i+row] = [0,1,0,0,0]
    memory = i + row
row = memory+1
for i in range(356):
    encode_label[i+row] = [0,0,1,0,0]
    memory = i + row
row = memory+1
for i in range(627):
    encode_label[i+row] = [0,0,0,1,0]
    memory = i + row
row = memory+1
for i in range(538):
    encode_label[i+row] = [0,0,0,0,1]
    memory = i + row
row = memory

labeled_data = np.hstack((data,encode_label))
print("bitti")



df = pandas.DataFrame(labeled_data)

df = df.sample(frac = 1).reset_index(drop=True)

df.to_csv("poz.csv")

df_train = df.loc[0:1424,:].to_csv("poz_train.csv")
df_val = df.loc[1424:1680,:].to_csv("poz_val.csv")
df_test = df.loc[1680:2192,:].to_csv("poz_test.csv")