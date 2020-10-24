# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np
import pandas as pd
import sklearn
import random
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
# from helper_functions import calculate_accuracy, train_test_split
dataset = pd.read_csv("./data.csv")
X_train = dataset.iloc[0:32764, 0:48]
X_test = dataset.iloc[32764:, 0:48]
Y_train = dataset.iloc[0:32764, 48]
Y_test = dataset.iloc[32764:, 48]
sc_x = preprocessing.MinMaxScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
# prediction = max(set(pred_values), key=pred_values.count)
# cm = confusion_matrix(Y_test, Y_pred)
# print(cm)
# # print(precision_score(Y_test, Y_pred, average='macro'))
# # print(f1_score(Y_test, Y_pred))
# # print(accuracy_score(Y_test, Y_pred))
# # print(Accuracy(Y_test,Y_pred))

def bootstrapping(X_train, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(X_train), size=n_bootstrap)
    df_bootstrapped = X_train[bootstrap_indices]

    return df_bootstrapped


# print(get_potential_splits(X_train, random_subspace=2))
# print(bootstrapping(X_train, 500))



def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    count = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count += 1

    accuracy = count * 100 / len(y_pred)
    print(f'Accuracy = {accuracy}%')


def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            true_positive += 1
        if y_true[i] == 1 != y_pred[i]:
            false_negative += 1
     recall1 = true_positive/(true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 2:
             true_positive += 1
         if y_true[i] == 2 != y_pred[i]:
             false_negative += 1
     recall2 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 3:
             true_positive += 1
         if y_true[i] == 3 != y_pred[i]:
             false_negative += 1
     recall3 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 4:
             true_positive += 1
         if y_true[i] == 4 != y_pred[i]:
             false_negative += 1
     recall4 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 5:
             true_positive += 1
         if y_true[i] == 5 != y_pred[i]:
             false_negative += 1
     recall5 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 6:
             true_positive += 1
         if y_true[i] == 6 != y_pred[i]:
             false_negative += 1
     recall6 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 7:
             true_positive += 1
         if y_true[i] == 7 != y_pred[i]:
             false_negative += 1
     recall7 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 8:
             true_positive += 1
         if y_true[i] == 8 != y_pred[i]:
             false_negative += 1
     recall8 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 9:
             true_positive += 1
         if y_true[i] == 9 != y_pred[i]:
             false_negative += 1
     recall9 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 10:
             true_positive += 1
         if y_true[i] == 10 != y_pred[i]:
             false_negative += 1
     recall10 = true_positive / (true_positive + false_negative)

     true_positive = 0
     false_negative = 0
     for i in range(len(y_true)):
         if y_true[i] == y_pred[i] == 11:
             true_positive += 1
         if y_true[i] == 11 != y_pred[i]:
             false_negative += 1
     recall11 = true_positive / (true_positive + false_negative)
     recall = ((recall1 + recall2 + recall3 + recall4 + recall5 + recall6 + recall7 + recall8 + recall9 + recall10 + recall11)/11)
     print(recall)

# print(Recall(Y_test, Y_pred))
def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            true_positive += 1
        if y_pred[i] == 1 != y_true[i]:
            false_positive += 1
    precision1 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 2:
            true_positive += 1
        if y_pred[i] == 2 != y_true[i]:
            false_positive += 1
    precision2 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 3:
            true_positive += 1
        if y_pred[i] == 3 != y_true[i]:
            false_positive += 1
    precision3 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 4:
            true_positive += 1
        if y_pred[i] == 4 != y_true[i]:
            false_positive += 1
    precision4 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 5:
            true_positive += 1
        if y_pred[i] == 5 != y_true[i]:
            false_positive += 1
    precision5 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 6:
            true_positive += 1
        if y_pred[i] == 6 != y_true[i]:
            false_positive += 1
    precision6 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 7:
            true_positive += 1
        if y_pred[i] == 7 != y_true[i]:
            false_positive += 1
    precision7 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 8:
            true_positive += 1
        if y_pred[i] == 8 != y_true[i]:
            false_positive += 1
    precision8 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 9:
            true_positive += 1
        if y_pred[i] == 9 != y_true[i]:
            false_positive += 1
    precision9 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 10:
            true_positive += 1
        if y_pred[i] == 10 != y_true[i]:
            false_positive += 1
    precision10 = true_positive / (true_positive + false_positive)

    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 11:
            true_positive += 1
        if y_pred[i] == 11 != y_true[i]:
            false_positive += 1
    precision11 = true_positive / (true_positive + false_positive)

    precision = ((precision1 + precision2 + precision3 + precision4 + precision5 + precision6 + precision7 + precision8 + precision9 + precision10 + precision11)/11)
    print(precision)
# print(Precision(Y_test, Y_pred))

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    print("Number of Clusters are: " + str(len(Clusters)))

    def Euclidean_Distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return np.sqrt(distance)

    centroids = [X_train[100], X_train[500], X_train[1000], X_train[20], X_train[250], X_train[340]]
    for i in X_train:
        a = Euclidean_Distance(i, centroids[0])
        b = Euclidean_Distance(i, centroids[1])
        c = Euclidean_Distance(i, centroids[2])
        d = Euclidean_Distance(i, centroids[3])
        e = Euclidean_Distance(i, centroids[4])
        f = Euclidean_Distance(i, centroids[5])
    distances = [a, b, c, d, e, f]
    wcss = distances[0] + distances[1] + distances[2] + distances[3] + distances[4] + distances[5]
    print("Total is: " + str(wcss))
    return wcss

def ConfusionMatrix(y_true,y_pred):

    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    confusion_mat = np.zeros((11, 11))
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            confusion_mat[0][0] += 1
        if y_true[i] == y_pred[i] == 2:
            confusion_mat[1][1] += 1
        if y_true[i] == y_pred[i] == 3:
            confusion_mat[2][2] += 1
        if y_true[i] == y_pred[i] == 4:
            confusion_mat[3][3] += 1
        if y_true[i] == y_pred[i] == 5:
            confusion_mat[4][4] += 1
        if y_true[i] == y_pred[i] == 6:
            confusion_mat[5][5] += 1
        if y_true[i] == y_pred[i] == 7:
            confusion_mat[6][6] += 1
        if y_true[i] == y_pred[i] == 8:
            confusion_mat[7][7] += 1
        if y_true[i] == y_pred[i] == 9:
            confusion_mat[8][8] += 1
        if y_true[i] == y_pred[i] == 10:
            confusion_mat[9][9] += 1
        if y_true[i] == y_pred[i] == 11:
            confusion_mat[10][10] += 1


    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            confusion_mat[y_true[i]-1][y_pred[i]-1] += 1

    print(confusion_mat)


def KNN(X_train,X_test,Y_train,Y_test):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """


     # Euclidean Distance
     def Euclidean_Distance(row1, row2):
         distance = 0.0
         for i in range(len(row1)):
             distance += (row1[i] - row2[i]) ** 2
         return np.sqrt(distance)

     def get_neighbours(X_train, X_test, number_of_neighbours):
         distances = []
         count = 0
         for row in X_train:
             dist = Euclidean_Distance(row, X_test)
             distances.append((count, dist))
             count += 1
         distances.sort(key=lambda tup: tup[1])
         neighbors = []
         for i in range(number_of_neighbours):
             neighbors.append(distances[i][0])
         return neighbors

     # Predict
     def predict_classification(train, test_row, num_neighbors):
         neighbors = get_neighbours(train, test_row, num_neighbors)
         output_values = []
         for i in neighbors:
             output_values.append(Y_test[i])
         prediction = max(set(output_values), key=output_values.count)
         return prediction


     count = 0
     for i in range(100):
         prediction = predict_classification(X_test, X_test[i], 7)
         if Y_test[i] == prediction:
             count += 1

     accuracy = count * 100 / 100
     print(f'Accuracy = {accuracy}%')


# knnclass = KNN(X_train,X_test,Y_train,Y_test)
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """

def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    rows = np.shape(X_train)[0]
    # print("Number of rows in the dataset are: " + str(m))

    # Number of columns in the dataset
    columns = np.shape(X_train)[1]
    # print("Number of columns in the dataset are: " + str(n))

    # Initializing classes for data points
    classes = np.random.randint(low=0, high=N, size=rows)

    centroids = [X_train[100], X_train[500], X_train[1000], X_train[20], X_train[250], X_train[340]]

    max_iter = 300
    for i in range(max_iter):
        # computing distances between datapoints and centroids
        # refrred from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        distances = np.array(
            [np.linalg.norm(X_train - c, axis=1) for c in centroids])

        # New classes - Centroids with minimal distance
        # Referred standard numpy function from https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
        new_classes = np.argmin(distances, axis=0)

        if (classes == new_classes).all():
            classes = new_classes
            print('K-means is completed')
            break
        else:
            difference = np.mean(classes != new_classes)
            print('%4f%% labels changed' % (difference * 100))
            classes = new_classes
            for c in range(N):
                # computing centroids by taking the mean over associated data points
                centroids[c] = np.mean(X_train[classes == c], axis=0)

    return classes

# Kmeans(X_train, 5)
def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :rtype: List[numpy.ndarray] 
    """
    #KNN
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, Y_train)
    Y_predknn = classifier.predict(X_test)
    print(Accuracy(Y_test, Y_predknn))
    confusionknn = ConfusionMatrix(Y_test, Y_predknn)
    plt.plot(confusionknn)

    # from matplotlib import style
    # style.use('ggplot')

    # Fitting SVM into training set
    classifier = SVC(kernel='linear', random_state=0, C=1.0)
    classifier.fit(X_train, Y_train)
    # Predicting the Test set results
    Y_predsvm = classifier.predict(X_test)
    print(Accuracy(Y_test, Y_predsvm))
    # confusionsvm = ConfusionMatrix(Y_test, Y_predsvm)

    # Making the Confusion Matrix
    # ConfusionMatrix(Y_test, Y_pred)

    # plt.scatter(Y_test, Y_pred)
    # plt.show()


    logisticRegr = LogisticRegression(max_iter=9999)
    logisticRegr.fit(X_train, Y_train)
    ypredlogres = logisticRegr.predict(X_test)
    print(Accuracy(Y_test, ypredlogres))
    confusionlogis = ConfusionMatrix(Y_test, ypredlogres)
    # plt.matshow(confusionlogis)

    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train , Y_train)
    y_preddec = classifier.predict(X_test)
    print(Accuracy(Y_test, y_preddec))
    confusiondec = ConfusionMatrix(Y_test, y_preddec)
    # plt.matshow(confusiondec)

# skKnn = SklearnSupervisedLearning(X_train, Y_train, X_test)

def SklearnVotingClassifier(X_train,Y_train,X_test):

    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, Y_train)
    Y_predknn = classifier.predict(X_test)
    classifier = SVC(kernel='linear', random_state=0, C=1.0)
    classifier.fit(X_train, Y_train)
    Y_predsvm = classifier.predict(X_test)
    logisticRegr = LogisticRegression(max_iter=9999)
    logisticRegr.fit(X_train, Y_train)
    ypredlogres = logisticRegr.predict(X_test)
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, Y_train)
    y_preddec = classifier.predict(X_test)
    ypred = {}
    count = 0
    for i in range(len(X_test)):
        ypred[count] = [Y_predknn[count], Y_predsvm[count], ypredlogres[count], y_preddec[count]]
        count += 1
    pred_values = []
    for key, value in ypred.items():
        pred_values.append(max(set(value)))
    print(Accuracy(Y_test, pred_values))

# SklearnVotingClassifier(X_train,Y_train,X_test)


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""




