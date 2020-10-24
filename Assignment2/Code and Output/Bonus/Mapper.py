#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import sys
import operator
from sklearn import preprocessing


train_dataset = pd.read_csv('Train.csv')
x_train = train_dataset.iloc[:, 0:48]
sc_x = preprocessing.MinMaxScaler()
x_train = sc_x.fit_transform(x_train)
x_train = np.asarray(x_train)
f = sys.stdin
x_test_data = pd.read_csv(f)
x_test = x_test_data.iloc[:, 0:48]
x_test = sc_x.transform(x_test)
x_test = np.asarray(x_test)
y_train = train_dataset.iloc[:, 48]
y_train = np.asarray(y_train)

def euclidean_distance(i):
    knn = {}
    for row in range(40956):
        euclidean_dist = np.linalg.norm(x_test[i] - x_train[row])
        knn[row] = euclidean_dist
    sorted_dict = sorted(knn.items(), key=operator.itemgetter(1))
    sorted_dict = sorted_dict[:7]
    # print(sorted_dict)
    for x in sorted_dict:
        x = str(x)
        x = x.replace("'", '')
        x = x.replace(",", '')
        x = x.replace('(', '')
        x = x.replace(')', '')
        x = x.split()
        x = x[0]
        b = int(x)
        print(i, y_train[b])


euclidean_distance(0)
euclidean_distance(1)
euclidean_distance(2)
euclidean_distance(3)
euclidean_distance(4)
euclidean_distance(5)
euclidean_distance(6)
euclidean_distance(7)
euclidean_distance(8)
euclidean_distance(9)
euclidean_distance(10)
euclidean_distance(11)
euclidean_distance(12)
euclidean_distance(13)
euclidean_distance(14)
