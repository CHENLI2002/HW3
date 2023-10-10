import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from concurrent.futures import ProcessPoolExecutor


data = pd.read_csv("emails.csv")

y = np.array(data['Prediction'].values)

def get_fold(data):
    fold_size = 1000
    test = []
    train = []
    for i in range(5):
        test = data.iloc[i * fold_size : (i+1) * fold_size]
        train = pd.concat([data.iloc[: i * fold_size], data.iloc[(i+1) * fold_size :]])
        yield test, train

k_fold = list(get_fold(data))

def calculate_dist(X, test):
    different = X - test
    different = different**2
    different = np.sum(different, axis=1)
    different = np.sqrt(different)

    return  different

def knn(X, y, test_sample, k):
    test_sample = test_sample
    distances = calculate_dist(X, test_sample)
    sorted_indices = distances.argsort()[:k]
    closest_labels = y[sorted_indices]
    count_1 = np.sum(closest_labels == 1)
    count_0 = k - count_1
    return 0 if count_0 > count_1 else 1

def knn_worker(args):
    X, y, test_sample, k = args
    return knn(X, y, test_sample, k)

def parallel_knn(X, y, X_test, k):
    with ProcessPoolExecutor() as executor:
        predictions = list(executor.map(knn_worker, [(X, y, sample, k) for sample in X_test]))
    return predictions

if __name__ == '__main__':
    ks = [1, 3, 5, 7, 10]
    accuracy = []

    for k in ks:
        k_accuracy = 0
        for index, fold in enumerate(k_fold):
            print(k)
            test, train = fold
            X_train = train.iloc[:, 1:-1].values.astype(float)
            y_train = train['Prediction'].values.astype(float)
            X_test = test.iloc[:, 1:-1].values.astype(float)
            y_test = test['Prediction'].values.astype(float)
            predictions = parallel_knn(X_train, y_train, X_test, k)

            correct = 0
            for i, _ in enumerate(predictions):

                if y_test[i] == predictions[i]:
                    correct += 1

            k_accuracy += correct / len(y_test) / 5

        accuracy.append([k, k_accuracy])

    plt.figure(figsize=(10, 8))
    plt.plot([x[0] for x in accuracy], [x[1] for x in accuracy])
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.grid()
    plt.title("k vs. accuracy")
    plt.savefig("k vs accuracy")
    plt.show()
