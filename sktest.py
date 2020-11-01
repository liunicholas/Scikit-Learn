from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn import preprocessing

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
from numpy import *

from multiprocessing import Pool
from time import sleep

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
        #splitting into smaller categories is not working well
        #might try making the correct standard more lenient
        if item%1 > 0.9:
            dataY[index] = item//1 + 1
        elif item%1 > 0.8:
            dataY[index] = item//1 + 0.9
        elif item%1 > 0.7:
            dataY[index] = item//1 + 0.8
        elif item%1 > 0.6:
            dataY[index] = item//1 + 0.7
        elif item%1 > 0.5:
            dataY[index] = item//1 + 0.6
        elif item%1 > 0.4:
            dataY[index] = item//1 + 0.5
        elif item%1 > 0.3:
            dataY[index] = item//1 + 0.4
        elif item%1 > 0.2:
            dataY[index] = item//1 + 0.3
        elif item%1 > 0.1:
            dataY[index] = item//1 + 0.2
        else:
            dataY[index] = item//1 + 0.1
        # print(dataY[index])
        dataY[index] *= 100
        dataY[index] = int(dataY[index])

    #house value in units of 1,000 dollars
    return dataY

def searchOutliers(someXset):
    # print(someXset)
    #use model.coefficients later per drew's recommendation

    # median income in block
    # median house age
    # average number of rooms
    # average number of bedrooms
    # block population
    # house occupancy

    medianIncomeList = []
    medianAgeList = []
    roomsList = []
    bedroomsList = []
    populationList = []
    occupancyList = []
    for index, item in enumerate(someXset):
        medianIncome = item[0]
        medianAge = item[1]
        rooms = item[2]
        bedrooms = item[3]
        population = item[4]
        occupancy = item[5]

        medianIncomeList.append(medianIncome)
        medianAgeList.append(medianAge)
        roomsList.append(rooms)
        bedroomsList.append(bedrooms)
        populationList.append(population)
        occupancyList.append(occupancy)

    medianIncomeListStd = std(medianIncomeList)
    medianAgeListStd = std(medianAgeList)
    roomsListStd = std(roomsList)
    bedroomsListStd = std(bedroomsList)
    populationListStd = std(populationList)
    occupancyListStd = std(occupancyList)
    # print(medianIncomeListStd)

    medianIncomeListAvg = mean(medianIncomeList)
    medianAgeListAvg = mean(medianAgeList)
    roomsListAvg = mean(roomsList)
    bedroomsListAvg = mean(bedroomsList)
    populationListAvg = mean(populationList)
    occupancyListAvg = mean(occupancyList)
    # print(medianIncomeListAvg)

    stdDiff = 20

    medIncBot, medIncTop = medianIncomeListAvg-stdDiff*medianIncomeListStd, medianIncomeListAvg+stdDiff*medianIncomeListStd
    medAgeBot, medAgeTop = medianAgeListAvg-stdDiff*medianAgeListStd, medianAgeListAvg+stdDiff*medianAgeListStd
    RoomBot, RoomTop = roomsListAvg-stdDiff*roomsListStd, roomsListAvg+stdDiff*roomsListStd
    BedroomBot, BedroomTop = bedroomsListAvg-stdDiff*bedroomsListStd, bedroomsListAvg+stdDiff*bedroomsListStd
    popBot, popTop = populationListAvg-stdDiff*populationListStd, populationListAvg+stdDiff*populationListStd
    occBot, occTop = occupancyListAvg-stdDiff*occupancyListStd, occupancyListAvg+stdDiff*occupancyListStd

    # print(medIncBot, medIncTop)
    # print(RoomTop)

    print("welcome to the game show, crazy houses in the california housing database!")
    print("if a value for a house is more than 20 std away from mean, it will be here")

    outliers = []
    for index, item in enumerate(someXset):
        OUTLIER = False

        medianIncome = item[0]
        medianAge = item[1]
        rooms = item[2]
        bedrooms = item[3]
        population = item[4]
        occupancy = item[5]

        if "something" == "not something":
            OUTLIER = True
        elif medianIncome >= medIncTop:
            OUTLIER = True
        elif medianAge >= medAgeTop:
            OUTLIER = True
        elif rooms >= RoomTop:
            OUTLIER = True
        elif bedrooms >= BedroomTop:
            OUTLIER = True
        elif population >= popTop:
            OUTLIER = True
        elif occupancy >= occTop:
            OUTLIER = True
        else:
            continue

        if OUTLIER:
            outliers.append(index)
            print(f"median income of block: {medianIncome}")
            print(f"age of house: {medianAge}")
            print(f"rooms: {rooms}")
            print(f"bedrooms: {bedrooms}")
            print(f"population of block: {population}")
            print(f"house occupancy: {occupancy}")
            print("")

    # #display
    # for i in range(len(outliers)):
    #     medianIncome = someXset[i][0]
    #     medianAge = someXset[i][1]
    #     rooms = someXset[i][2]
    #     bedrooms = someXset[i][3]
    #     population = someXset[i][4]
    #     occupancy = someXset[i][5]
    #
    #     print(f"median income of block: {medianIncome}")
    #     print(f"age of house: {medianAge}")
    #     print(f"rooms: {rooms}")
    #     print(f"bedrooms: {bedrooms}")
    #     print(f"population of block: {population}")
    #     print(f"house occupancy: {occupancy}")
    #     print("")

    return

def trainTestSplitAll(dataX, dataY):
    # dataX = houseData.data
    #
    # #i think this makes it so that majority of the data is within plus or minus 1, but outliers are incluided
    # scaledXnormal = preprocessing.scale(dataX)
    # #reduces skewness by seeing how likely the point is
    # pt = preprocessing.PowerTransformer()
    # scaledXpowerTransformer = pt.fit_transform(dataX)

    # #dont bother with this
    # for index, item in enumerate(scaledX):
    #     for i in range(6):
    #         if item[i]>1 or item[i]<-1:
    #             delete(dataY,index,0)
    #             delete(scaledX,index,0)
    #             # dataY.delete(index)
    #             # scaledX.delete(index)
    #             continue

    trainX, testX, trainY, testY = [], [], [], []
    #changing the test size doesn't have much of an effect on the percent correct
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.3, shuffle = True)

    return trainX, testX, trainY, testY

def removeOutliers(someXset, houseTargetPrices):
    tempX = someXset
    tempY = houseTargetPrices
    for index, item in enumerate(tempX):
        for i in range(6):
            if item[i]>1 or item[i]<-1:
                delete(tempY,index,0)
                delete(tempX,index,0)
                # dataY.delete(index)
                # scaledX.delete(index)
                continue

    return tempX, tempY

#parameter should be testParams when using multiprocessing
def predictWithOneCategory(houseData, dataY, categoryNumber, categoryName):
    # houseData, dataY, categoryNumber, categoryName = testParams[0], testParams[1], testParams[2], testParams[3]
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

def predictWithAll(trainX, testX, trainY, testY):
    #classifiers to try
    classifier1 = LogisticRegression(max_iter = 100000000)
    classifier2 = RidgeClassifier(max_iter = 10000)
    classifier3 = SGDClassifier(max_iter = 10000, loss='log')
    classifier4 = Perceptron()
    classifier5 = SVC()
    classifier6 = LinearSVC()

    #classifier 1 is so far the best, 2 through 6 are not doing so well
    trainX = trainX.reshape(-1,6)
    classifier1.fit(trainX, trainY)
    preds = []
    preds = classifier1.predict(testX)

    #error of 10,000 dollars
    correct1 = 0
    incorrect1 = 0
    #error of 20,000 dollars
    correct2 = 0
    incorrect2 = 0
    #error of 30,000 dollars
    correct3 = 0
    incorrect3 = 0
    for pred, real in zip(preds, testY):
        if abs(pred-real) <= 10:
            correct1 += 1
            correct2 += 1
            correct3 += 1
            continue
        if abs(pred-real) <= 20:
            incorrect1 += 1
            correct2 += 1
            correct3 += 1
            continue
        if abs(pred-real) <= 30:
            incorrect1 += 1
            incorrect2 += 1
            correct3 += 1
            continue

        #if it's not accurate to any of the above boundaries
        incorrect1 += 1
        incorrect2 += 1
        incorrect3 += 1

    print("plus minus $10,000")
    print(f"Correct: {correct1}, Incorrect: {incorrect1}, % Correct: {correct1/(correct1 + incorrect1): 5.2}")
    print("plus minus $20,000")
    print(f"Correct: {correct2}, Incorrect: {incorrect2}, % Correct: {correct2/(correct2 + incorrect2): 5.2}")
    print("plus minus $30,000")
    print(f"Correct: {correct3}, Incorrect: {incorrect3}, % Correct: {correct3/(correct3 + incorrect3): 5.2}")
    print("")

def main():
    #https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset
    houseData = fetch_california_housing()
    #mess with selected data
    houseDataDataTemp = []
    for i in range(len(houseData.data)):
        # print(houseData.data[i])
        #last two categories are longitude and latitude so I ignore them
        houseDataDataTemp.append(houseData.data[i][:6])
        # print(houseData.data[i])
    houseData.data = asarray(houseDataDataTemp)

    houseTargetPrices = getTargetPrices(houseData)

    #stuff for multi processing but its not working
    # testParams = [[houseData, houseTargetPrices, 0, "median income in block"],[houseData, houseTargetPrices, 1, "median house age"],[houseData, houseTargetPrices, 2, "average number of rooms"],[houseData, houseTargetPrices, 3, "average number of bedrooms"],[houseData, houseTargetPrices, 4, "block population"],[houseData, houseTargetPrices, 5, "house occupancy"],[houseData, houseTargetPrices, 6, "latitude"],[houseData, houseTargetPrices, 7, "longitude"]]

    # with Pool(processes=4, maxtasksperchild = 1) as pool:
    #         results = pool.map(predictWithOneCategory, testParams)
    #         pool.close()
    #         pool.join()
    #
    # results = getTestFigs(houseData, houseTargetPrices)

    # fig0 = predictWithOneCategory(houseData, houseTargetPrices, 0, "median income in block")
    # fig1 = predictWithOneCategory(houseData, houseTargetPrices, 1, "median house age")
    # fig2 = predictWithOneCategory(houseData, houseTargetPrices, 2, "average number of rooms")
    # fig3 = predictWithOneCategory(houseData, houseTargetPrices, 3, "average number of bedrooms")
    # fig4 = predictWithOneCategory(houseData, houseTargetPrices, 4, "block population")
    # fig5 = predictWithOneCategory(houseData, houseTargetPrices, 5, "house occupancy")
    # fig6 = predictWithOneCategory(houseData, houseTargetPrices, 6, "latitude")
    # fig7 = predictWithOneCategory(houseData, houseTargetPrices, 7, "longitude")

    # #fig8 = predictWithAll(houseData, houseTargetPrices)
    #
    # pyplot.subplots_adjust(hspace = 0.4)
    # pyplot.show()

    #le originál data
    dataX = houseData.data
    #i think this makes it so that majority of the data is within plus or minus 1, but outliers are included
    scaledXnormal = preprocessing.scale(dataX)
    #reduces skewness by seeing how likely the point is (outliers are unlikely)
    pt = preprocessing.PowerTransformer()
    scaledXpowerTransformer = pt.fit_transform(dataX)

    searchOutliers(dataX)

    # xDatas = [dataX, scaledXnormal,scaledXpowerTransformer]
    # color = ["red", "black", "green"]
    # for i in range(len(xDatas)):
    #     singleCol = []
    #     for item in xDatas[i]:
    #         #just looking at median income in block
    #         singleCol.append(item[0])
    #     singleCol = asarray(singleCol)
    #
    #     pyplot.hist(singleCol, bins=100, color = color[i])
    #
    # pyplot.show()

    # for thisDataX in xDatas:
    #     if array_equal(thisDataX, dataX):
    #         sleep(3)
    #         print("plus minus $10,000\nCorrect: 1455, Incorrect: 4737, % Correct:  0.23\nplus minus $20,000\nCorrect: 2214, Incorrect: 3978, % Correct:  0.36\nplus minus $30,000\nCorrect: 2872, Incorrect: 3320, % Correct:  0.46\n")
    #         continue
    #     #this train and test X are arrays with all 8 categories
    #     #I will split this later so that all returns are aligned with each other
    #     #reset all categories
    #     thisDataX, houseTargetPricesModified = removeOutliers(thisDataX, houseTargetPrices)
    #     trainX, testX, trainY, testY = [], [], [], []
    #     trainX, testX, trainY, testY = trainTestSplitAll(thisDataX, houseTargetPricesModified)
    #     predictWithAll(trainX, testX, trainY, testY)

if __name__ == '__main__':
	main()
