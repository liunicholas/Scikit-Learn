from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot

digits = load_digits()
digitsX = digits.images.reshape(len(digits.images), 64)
digitsY = digits.target
trainX, testX, trainY, testY = train_test_split(digitsX, digitsY, test_size = 0.3, shuffle = True)

classifier1 = LogisticRegression(max_iter = 10000)
classifier2 = RidgeClassifier(max_iter = 10000)
classifier3 = SGDClassifier(max_iter = 10000, loss='log')
classifier4 = Perceptron()
classifier5

classifier4.fit(trainX, trainY)
preds = classifier4.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier4, testX, testY)
pyplot.show()
