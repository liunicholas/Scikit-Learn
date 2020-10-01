# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn.cluster
import sklearn.datasets
import sklearn.preprocessing

digits = sklearn.datasets.load_digits()
digitsX = digits.images
digitsX = digitsX.reshape((len(digitsX), 64))
digitsY = digits.target

estimator = sklearn.cluster.KMeans(
    init="k-means++",
    n_clusters = 10,
    n_init = 10,
)
estimator.fit(sklearn.preprocessing.scale(digitsX))
predY = estimator.predict(digitsX)

finalPreds = [0 for i in range(10)]
for label in range(10):
    counts = [0 for i in range(10)]
    for pred, gt in zip(predY, digitsY):
        if pred == label:
            counts[gt] += 1
    finalPreds[label] = counts.index(max(counts))

correct = 0
incorrect = 0
for i in range(len(predY)):
    predY[i] = finalPreds[predY[i]]
for pred, gt in zip(predY, digitsY):
    if pred == gt: correct += 1
    if pred != gt: incorrect += 1
print(correct, incorrect)
print(f"{correct/len(predY):5.2} correct")
