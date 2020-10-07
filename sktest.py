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

from multiprocessing import Pool

# python 3.8.1
# certifi               2020.6.20
# cycler                0.10.0
# decorator             4.4.2
# imageio               2.9.0
# joblib                0.16.0
# kiwisolver            1.2.0
# mahotas               1.4.11
# matplotlib            3.3.1
# networkx              2.4
# numpy                 1.19.1
# opencv-contrib-python 4.4.0.42
# Pillow                7.2.0
# pip                   20.2.3
# pyparsing             2.4.7
# python-dateutil       2.8.1
# PyWavelets            1.1.1
# scikit-image          0.17.2
# scikit-learn          0.23.2
# scipy                 1.5.2
# setuptools            49.6.0
# six                   1.15.0
# threadpoolctl         2.1.0
# tifffile              2020.8.13
# wheel                 0.35.1

def getTargetPrices(houseData):
    #house value in units of 100,000
    dataY = []
    dataY = houseData.target
    for index, item in enumerate(dataY):
        if item%1 > 0.75:
            dataY[index] = item//1 + 1
        elif item%1 > 0.5:
            dataY[index] = item//1 + 0.75
        elif item%1 > 0.25:
            dataY[index] = item//1 + 0.5
        else:
            dataY[index] = item//1
        dataY[index] *= 100

    return dataY

def predictWithOneCategory(testParams):
    houseData, dataY, categoryNumber, categoryName = testParams[0], testParams[1], testParams[2], testParams[3]
    #different data each time
    dataX = []
    for item in houseData.data:
        dataX.append(item[categoryNumber])
    dataX = asarray(dataX)

    #reset all categories
    trainX, testX, trainY, testY = [], [], [], []
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.3, shuffle = True)

    #different classifiers to try
    classifier1 = LogisticRegression(max_iter = 100000)
    classifier2 = RidgeClassifier(max_iter = 10000)
    classifier3 = SGDClassifier(max_iter = 10000, loss='log')
    classifier4 = Perceptron()
    classifier5 = SVC()
    classifier6 = LinearSVC()

    trainX = trainX.reshape(-1,1)
    testX = testX.reshape(-1,1)
    # classifier6.fit(trainX, trainY)
    classifier1.fit(trainX, trainY)
    preds = []
    preds = classifier1.predict(testX)

    correct = 0
    incorrect = 0
    for pred, real in zip(preds, testY):
        if pred == real:
            correct += 1
        else:
            incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    fig = pyplot.figure(f"house price vs {categoryName}")
    fig.tight_layout()
    # plt1 = fig.add_subplot(221)
    plt2 = fig.add_subplot(212)
    plt3 = fig.add_subplot(211)
    plt2.title.set_text("testing values")
    plt3.title.set_text("all values")
    # plt1.plot_confusion_matrix(classifier2, testX, testY)
    #put testing category on x axis and house price on y axis
    plt2.scatter(testX, testY, c='green', marker='_', alpha=0.5, label='real values')
    plt2.scatter(testX, preds, c='red', marker='|', alpha=0.5, label='predicted values')
    plt3.scatter(dataX, dataY, c='black', marker='.', alpha=0.5)
    # pyplot.subplots_adjust(top=1.5)
    plt2.legend(loc='lower right')

    return fig

# def getTestFigs(houseData, houseTargetPrices):
#     testParams = [[houseData, houseTargetPrices, 0, "median income in block"],[houseData, houseTargetPrices, 1, "median house age"],[houseData, houseTargetPrices, 2, "average number of rooms"],[houseData, houseTargetPrices, 3, "average number of bedrooms"],[houseData, houseTargetPrices, 4, "block population"],[houseData, houseTargetPrices, 5, "house occupancy"],[houseData, houseTargetPrices, 6, "latitude"],[houseData, houseTargetPrices, 7, "longitude"]]
#
#     with Pool(processes=4, maxtasksperchild = 1) as pool:
#             results = pool.map(predictWithOneCategory, testParams)
#             pool.close()
#             pool.join()
#
#     return results

def main():
    #https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset
    houseData = fetch_california_housing()
    houseTargetPrices = getTargetPrices(houseData)

    testParams = [[houseData, houseTargetPrices, 0, "median income in block"],[houseData, houseTargetPrices, 1, "median house age"],[houseData, houseTargetPrices, 2, "average number of rooms"],[houseData, houseTargetPrices, 3, "average number of bedrooms"],[houseData, houseTargetPrices, 4, "block population"],[houseData, houseTargetPrices, 5, "house occupancy"],[houseData, houseTargetPrices, 6, "latitude"],[houseData, houseTargetPrices, 7, "longitude"]]

    with Pool(processes=4, maxtasksperchild = 1) as pool:
            results = pool.map(predictWithOneCategory, testParams)
            pool.close()
            pool.join()

    # results = getTestFigs(houseData, houseTargetPrices)

    # fig0 = predictWithOneCategory(houseData, houseTargetPrices, 0, "median income in block")
    # fig1 = predictWithOneCategory(houseData, houseTargetPrices, 1, "median house age")
    # fig2 = predictWithOneCategory(houseData, houseTargetPrices, 2, "average number of rooms")
    # fig3 = predictWithOneCategory(houseData, houseTargetPrices, 3, "average number of bedrooms")
    # fig4 = predictWithOneCategory(houseData, houseTargetPrices, 4, "block population")
    # fig5 = predictWithOneCategory(houseData, houseTargetPrices, 5, "house occupancy")
    # fig6 = predictWithOneCategory(houseData, houseTargetPrices, 6, "latitude")
    # fig7 = predictWithOneCategory(houseData, houseTargetPrices, 7, "longitude")

    pyplot.subplots_adjust(hspace = 0.4)
    pyplot.show()

if __name__ == '__main__':
	main()
