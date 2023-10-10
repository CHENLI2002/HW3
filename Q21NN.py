import numpy as np
import pandas as pd
import math
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
    for index, fold in enumerate(k_fold):
        test, train = fold
        X_train = train.iloc[:, 1:-1].values.astype(float)
        y_train = train['Prediction'].values.astype(float)
        X_test = test.iloc[:, 1:-1].values.astype(float)
        y_test = test['Prediction'].values.astype(float)
        predictions = parallel_knn(X_train, y_train, X_test, 1)

        tp = 0
        all_pre_pos = 0
        true_pos = 0
        correct = 0
        print("getshere")

        for i, _ in enumerate(predictions):
            if predictions[i] == 1:
                all_pre_pos += 1

            if y_test[i] == 1:
                true_pos += 1

            if predictions[i] == 1 and y_test[i] == predictions[i]:
                tp += 1

            if y_test[i] == predictions[i]:
                correct += 1
        if all_pre_pos == 0:
            precision = 0
        else:
            precision = tp / all_pre_pos
        print(f"{index + 1}fold accuracy: {correct / len(predictions)}, recall: {tp/ true_pos}, precision: {precision}")


