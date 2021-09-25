# importing the numpy library
import numpy as np

# importing the pandas library
import pandas as pd

# importing the datasets and concatenating them into one dataframe

data1 = pd.read_csv('jester-data-1.csv', header = None)
data2 = pd.read_csv('jester-data-2.csv', header = None)
data3 = pd.read_csv('jester-data-3.csv', header = None)


data = pd.concat([data1, data2, data3])

# selecting the user-ratings of the fixed 100 jokes
X = np.array(data)[:,1:]

# 80-20 Train Test Split of the dataset
training_set = X[0:int(0.8*73421)]
test_set = X[int(0.8*73421):]

# importing the PyTorch libraries for RBM Instantiation and Training
import torch
import torch.nn as nn 
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# converting the training dataframe into torch tensor
training_set = torch.FloatTensor(training_set)

# converting the test dataframe into torch tensor
test_set = torch.FloatTensor(test_set)

# data-preprocessing of the training set
## 1. Ratings in the range [7, 10] is set to 1
## 2. Ratings in the range [-10, 7) is set to 0
## 3. Missing Ratings described by 99 is set to -1
for i in range(training_set.shape[0]):
    for j in range(training_set.shape[1]):
        if training_set[i,j] >= 7 and training_set[i,j] <= 10:
            training_set[i,j] = 1
        elif training_set[i,j] < 7:
            training_set[i,j] = 0
training_set[training_set == 99] = -1

# data-preprocessing of the test set
## 1. Ratings in the range [7, 10] is set to 1
## 2. Ratings in the range [-10, 7) is set to 0
## 3. Missing Ratings described by 99 is set to -1
for i in range(test_set.shape[0]):
    for j in range(test_set.shape[1]):
        if test_set[i,j] >= 7 and test_set[i,j] <= 10:
            test_set[i,j] = 1
        elif test_set[i,j] < 7:
            test_set[i,j] = 0
test_set[test_set == 99] = -1

# loading the dataset, D2 i.e., the recommended ratings
answer = np.loadtxt("answer.csv", delimiter = ",")

# preparing the datset, D1
x = np.concatenate((training_set, test_set), axis = 0)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i,j] == -1:
            x[i,j] = answer[i,j]
            
# applying k-Means Clustering using 3 clusters on D1
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(answer)

# getting the predicted clusters on D2
predictions = kmeans.predict(x)

# preparaing the Preference Function Vectors for all the 3 clusters
n_joke_1 = np.zeros(100)
n_joke_2 = np.zeros(100)
n_joke_3 = np.zeros(100)

for i in range(predictions.shape[0]):
    if predictions[i] == 0:
        n_joke_1 += x[i]
    elif predictions[i] == 1:
        n_joke_2 += x[i]
    elif predictions[i] == 2:
        n_joke_3 += x[i]
        
unique, counts = np.unique(predictions, return_counts = True)

p_1 = n_joke_1/dict(zip(unique, counts))[0]
p_2 = n_joke_2/dict(zip(unique, counts))[1]
p_3 = n_joke_3/dict(zip(unique, counts))[2]


# saving each of the preference vectors in separate csv files
np.savetxt("cluster_0_mod.csv", p_1, delimiter = ",")
np.savetxt("cluster_1_mod.csv", p_2, delimiter = ",")
np.savetxt("cluster_2_mod.csv", p_3, delimiter = ",")


# saving list of joke names with serial numbers in a csv file
l = []
for i in range(1,101):
    label = "Joke " + str(i)
    l.append(label)

l = pd.Series(l)
l.to_csv("l.csv")

# The Line-Chart Visualization is done  in Excel Sheets by placing the Joke Names and preference columns together for each cluster at a time.