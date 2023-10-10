import math
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("D2z.txt")
X = data[:, :-1]
y = data[:, -1]

prediction = []

x_test = np.arange(-2, 2.1, 0.1)
x2_test = np.arange(-2, 2.1, 0.1)

X_0 = X[y == 0]
X_1 = X[y == 1]


def find_closest(x, x2):
    prev_dist = math.inf
    prev_label = -1
    for index, data in enumerate(X):
        dist = math.sqrt(math.pow((x - data[0]), 2) + math.pow((x2 - data[1]), 2))
        if dist < prev_dist:
            prev_dist = dist
            prev_label = y[index]

    return prev_label

for x in x_test:
    for x2 in x2_test:
        prediction.append([x, x2, find_closest(x, x2)])


pre_0 = [pre for pre in prediction if pre[2] == 0]
pre_1 = [pre for pre in prediction if pre[2] == 1]

print(pre_0)

plt.figure(figsize=(10, 8))
plt.scatter(X_0[:, 0], X_0[:, 1], c='black', alpha=0.5)
plt.scatter(X_1[:, 0], X_1[:, 1], c='g', alpha=0.5)
plt.scatter([pre[0] for pre in pre_0], [pre[1] for pre in pre_0], c='b')
plt.scatter([pre[0] for pre in pre_1], [pre[1] for pre in pre_1], c='r')
plt.xlabel("feature 1")
plt.ylabel('feature 2')
plt.title("plt for q1 (transparent (black/green) dots are training samples)")
plt.savefig("Q1 1nn")
plt.show()