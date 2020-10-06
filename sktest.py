from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
from numpy import *

# digits = load_digits()
# digitsX = digits.images.reshape(len(digits.images), 64)
# digitsY = digits.target
# trainX, testX, trainY, testY = train_test_split(digitsX, digitsY, test_size = 0.3, shuffle = True)

houseData = fetch_california_housing()
# print(houseData)
print(type(houseData))
#median income
dataX = []
# dataX = houseData.data[0]
for item in houseData.data:
    dataX.append(item[0])
dataX = asarray(dataX)
print(type(dataX))
#house value in units of 100,000
dataY = houseData.target
for index, item in enumerate(dataY):
    print(item)
    if item%1 > 0.75:
        dataY[index] = item//1 + 1
        print("here1")
    elif item%1 > 0.5:
        dataY[index] = item//1 + 0.75
        print("here2")
    elif item%1 > 0.25:
        dataY[index] = item//1 + 0.5
        print("here3")
    else:
        dataY[index] = item//1
        print("here4")
    dataY[index] *= 100
    print(dataY[index])

trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.3, shuffle = True)

classifier1 = LogisticRegression(max_iter = 10000)
classifier2 = RidgeClassifier(max_iter = 10000)
classifier3 = SGDClassifier(max_iter = 10000, loss='log')
classifier4 = Perceptron()
classifier5 = SVC()
classifier6 = LinearSVC()

trainX = trainX.reshape(-1,1)
testX = testX.reshape(-1,1)
# classifier6.fit(trainX, trainY)
classifier2.fit(trainX, trainY)
preds = classifier2.predict(testX)

correct = 0
incorrect = 0
for pred, real in zip(preds, testY):
    if pred == real:
        correct += 1
    else:
        incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

fig = pyplot.figure("median income of block vs house price")
fig.tight_layout()
# plt1 = fig.add_subplot(221)
plt2 = fig.add_subplot(212)
plt3 = fig.add_subplot(211)
plt2.title.set_text("all values")
plt3.title.set_text("testing values")
# plt1.plot_confusion_matrix(classifier2, testX, testY)
#put house price on x axis and income on y axis
plt2.scatter(testX, testY, c='green', marker='_', alpha=0.5, label='real values')
plt2.scatter(testX, preds, c='red', marker='|', alpha=0.5, label='predicted values')
plt3.scatter(dataX, dataY, c='black', marker='.', alpha=0.5)
# pyplot.subplots_adjust(top=1.5)
pyplot.subplots_adjust(hspace = 0.4)
plt2.legend(loc='lower right')
pyplot.show()
