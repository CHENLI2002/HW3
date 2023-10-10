import numpy as np
import pandas as pd
import math
from scipy.spatial import distance

data = pd.read_csv("emails.csv")
data.iloc[:, 1:] = data.iloc[:, 1:].astype(np.float64)

def get_fold(data):
    fold_size = 1000
    test = []
    train = []
    for i in range(5):
        test = data.iloc[i * fold_size : (i+1) * fold_size]
        train = pd.concat([data.iloc[: i * fold_size], data.iloc[(i+1) * fold_size :]])
        yield test, train


k_fold = list(get_fold(data))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, theta):
    y_hat = sigmoid(np.dot(X, theta))
    gradient = np.dot(X.T, (y_hat - y)) / y.size
    return gradient

def gradient_descent(X, y):
    theta = np.zeros(X.shape[1])

    for i in range(3000):
        gradient = compute_gradient(X, y, theta)
        theta -= 0.001 * gradient

    return theta

for index, fold in enumerate(k_fold):
    test, train = fold
    X_train = train.iloc[:, 1:].values
    y_train = train['Prediction'].values
    X_test = test.iloc[:, 1:].values
    y_test = test['Prediction'].values
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    theta = gradient_descent(X_train, y_train)

    predictions = [1 if sigmoid(np.dot(theta, x)) > 0.5 else 0 for x in X_test]

    tp = 0
    all_pre_pos = 0
    true_pos = 0
    correct = 0

    for i, _ in enumerate(predictions):
        if predictions[i] == 1:
            all_pre_pos += 1

        if y_test[i] == 1:
            true_pos += 1

        if predictions[i] == 1 and y_test[i] == predictions[i]:
            tp += 1

        if y_test[i] == predictions[i]:
            correct += 1
    print(
        f"{index + 1}fold accuracy: {correct / len(predictions)}, recall: {tp / true_pos}, precision: {tp / all_pre_pos}")
