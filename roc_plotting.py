import matplotlib.pyplot as plt

predictions = [[0.95, 'pos'], [0.95, 'pos'], [0.8, 'neg'], [0.7, 'pos'], [0.55, 'pos'], [0.45, 'neg'],
               [0.4, 'pos'], [0.3, 'pos'], [0.2, 'neg'], [0.1, 'neg']]

num_neg = 4
num_pos = 6
Tp = 0
Fp = 0
last_Tp = 0
coordinates = []

for index, data in enumerate(predictions):
    if index > 0 and predictions[index - 1][0] != predictions[index][0] and data[1] == 'neg' and Tp > last_Tp:
        FPR = Fp/num_neg
        TPR = Tp/num_pos
        coordinates.append([FPR, TPR])
        last_Tp = Tp

    if data[1] == 'pos':
        Tp += 1
    else:
        Fp += 1


FPR = Fp / num_neg
TPR = Tp / num_pos
coordinates.append([FPR, TPR])
print(coordinates)

plt.figure(figsize=(2, 1))
plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates], label='ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.savefig("rocCurve")