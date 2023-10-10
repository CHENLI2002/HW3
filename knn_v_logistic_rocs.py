import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt
from scipy.spatial import distance
from gradientdecent import gradient_descent, sigmoid
from concurrent.futures import ProcessPoolExecutor



def calculate_dist(X, test):
    different = X - test
    different = different**2
    different = np.sum(different, axis=1)
    different = np.sqrt(different)

    return different

def knn(X, y, test_sample, k):
    test_sample = test_sample
    distances = calculate_dist(X, test_sample)
    sorted_indices = distances.argsort()[:k]
    closest_labels = y[sorted_indices]
    count_1 = np.sum(closest_labels == 1)
    return count_1/k

def knn_worker(args):
    X, y, test_sample, k = args
    return knn(X, y, test_sample, k)

def parallel_knn(X, y, X_test, k):
    with ProcessPoolExecutor() as executor:
        confidence = list(executor.map(knn_worker, [(X, y, sample, k) for sample in X_test]))
    return confidence

if __name__ == '__main__':

    data = pd.read_csv("emails.csv")
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(np.float64)
    X = data[:, 1:-1]
    y = data["Prediction"]
    X_test = X.iloc[0:1001]
    X_train = X.iloc[1001:]
    y_test = y.iloc[0:1001]
    y_train = y.iloc[1001:]
    predict_grad = []
    predict_knn = []

    theta = gradient_descent(X_train, y_train)
    confidences = sigmoid(np.dot(theta, X_test))
    for index, con in enumerate(confidences):
        predict_grad.append([con, index])

    confidence_knn = parallel_knn(X_train, y_train, X_test, 5)
    for index, con in enumerate(confidence_knn):
        predict_knn.append([con, index])

    num_neg = np.any(y == 0)
    num_pos = np.any(y == 1)
    Tp = 0
    Fp = 0
    last_Tp = 0
    coordinates = []

    for index, data in enumerate(confidences):
        if index > 0 and confidences[index - 1][0] != confidences[index][0] and data[1] == 0 and Tp > last_Tp:
            FPR = Fp / num_neg
            TPR = Tp / num_pos
            coordinates.append([FPR, TPR])
            last_Tp = Tp

        if data[1] == 1:
            Tp += 1
        else:
            Fp += 1

    FPR = Fp / num_neg
    TPR = Tp / num_pos
    coordinates.append([FPR, TPR])
    print(coordinates)

    coordinates_knn = []

    for index, data in enumerate(confidence_knn):
        if index > 0 and confidence_knn[index - 1][0] != confidence_knn[index][0] and data[1] == 0 and Tp > last_Tp:
            FPR = Fp / num_neg
            TPR = Tp / num_pos
            coordinates_knn.append([FPR, TPR])
            last_Tp = Tp

        if data[1] == 1:
            Tp += 1
        else:
            Fp += 1

    FPR = Fp / num_neg
    TPR = Tp / num_pos
    coordinates_knn.append([FPR, TPR])
    print(coordinates_knn)

    plt.figure(figsize=(10, 8))
    plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], color='r')
    plt.plot([x[0] for x in coordinates_knn], [x[1] for x in coordinates_knn], color='b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.savefig("rocCurve_knn_vs_grad")
    plt.show()

