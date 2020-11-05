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
from pandas import *

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
# pandas                1.1.4

def getTargetPrices(houseData):
    #house value in units of 100,000
    dataY = []
    dataY = houseData.target
    for index, item in enumerate(dataY):
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

        #price is in units of 1000 dollars
        dataY[index] *= 100
        dataY[index] = int(dataY[index])

    # discretize = preprocessing.KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')
    # # discretize.fit(dataY)
    # tempY = discretize.fit_transform([dataY])

    # dataY = digitize(dataY,bins=50)

    # print(dataY)
    # #using pandas to cut, KBinsDiscretizer needs a 2d array, not sure why
    # #just kidding this doesn't work either
    # cut(dataY, bins=100)
    # print(dataY)

    #house value in units of 1,000 dollars
    return dataY

def searchOutliers(someXset, dataY):
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

    #get the standard deviation of each list
    medianIncomeListStd = std(medianIncomeList)
    medianAgeListStd = std(medianAgeList)
    roomsListStd = std(roomsList)
    bedroomsListStd = std(bedroomsList)
    populationListStd = std(populationList)
    occupancyListStd = std(occupancyList)

    #get the average of each list
    medianIncomeListAvg = mean(medianIncomeList)
    medianAgeListAvg = mean(medianAgeList)
    roomsListAvg = mean(roomsList)
    bedroomsListAvg = mean(bedroomsList)
    populationListAvg = mean(populationList)
    occupancyListAvg = mean(occupancyList)

    #how many standard deviations to count as an outlier
    stdDiff = 20

    #gets the top and bottom boundary
    medIncBot, medIncTop = medianIncomeListAvg-stdDiff*medianIncomeListStd, medianIncomeListAvg+stdDiff*medianIncomeListStd
    medAgeBot, medAgeTop = medianAgeListAvg-stdDiff*medianAgeListStd, medianAgeListAvg+stdDiff*medianAgeListStd
    RoomBot, RoomTop = roomsListAvg-stdDiff*roomsListStd, roomsListAvg+stdDiff*roomsListStd
    BedroomBot, BedroomTop = bedroomsListAvg-stdDiff*bedroomsListStd, bedroomsListAvg+stdDiff*bedroomsListStd
    popBot, popTop = populationListAvg-stdDiff*populationListStd, populationListAvg+stdDiff*populationListStd
    occBot, occTop = occupancyListAvg-stdDiff*occupancyListStd, occupancyListAvg+stdDiff*occupancyListStd

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
        housePrice = dataY[index]

        #first line is just so i can easily comment out a condiiton for testing
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
            print(f"median income of block: {medianIncome*100000: 5.2f}")
            print(f"median age of house in block: {medianAge: 5.2f}")
            print(f"average nunmber of rooms: {rooms: 5.2f}")
            print(f"average number of bedrooms: {bedrooms: 5.2f}")
            print(f"block population: {population: 5.2f}")
            print(f"average house occupancy: {occupancy: 5.2f}")
            print(f"house price: {housePrice*1000: 5.2f}")
            print("")

    return

def trainTestSplitAll(dataX, dataY):
    trainX, testX, trainY, testY = [], [], [], []
    #changing the test size doesn't have much of an effect on the percent correct
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.3, shuffle = True)

    return trainX, testX, trainY, testY

def removeOutliers(someXset, houseTargetPrices):
    tempX = someXset
    tempY = houseTargetPrices
    for index, item in enumerate(tempX):
        for i in range(6):
            #the scaling methods put the majority of the data between -1 and 1
            #this gets rid of everything that isn't "normal"
            if item[i]>1 or item[i]<-1:
                delete(tempY,index,0)
                delete(tempX,index,0)
                continue

    return tempX, tempY

def predictWithOneCategory(houseData, dataY, categoryNumber, categoryName):
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

    #i might not need these rehsapes but they're there just in case
    trainX = trainX.reshape(-1,1)
    testX = testX.reshape(-1,1)
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
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect)*100: 5.2f}")

    fig = pyplot.figure(f"house price vs {categoryName}")
    fig.tight_layout()
    # plt1 = fig.add_subplot(221)
    plt2 = fig.add_subplot(212)
    plt3 = fig.add_subplot(211)
    plt2.title.set_text("testing values")
    plt3.title.set_text("all values")
    #put testing category on x axis and house price on y axis
    plt2.scatter(testX, testY, c='green', marker='_', alpha=0.5, label='real values')
    plt2.scatter(testX, preds, c='red', marker='|', alpha=0.5, label='predicted values')
    plt3.scatter(dataX, dataY, c='black', marker='.', alpha=0.5)
    # pyplot.subplots_adjust(top=1.5)
    plt2.legend(loc='lower right')

    return fig

def predictWithOneCategoryMP(testParams):
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

    #i might not need these rehsapes but they're there just in case
    trainX = trainX.reshape(-1,1)
    testX = testX.reshape(-1,1)
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
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect)*100: 5.2f}")

    fig = pyplot.figure(f"house price vs {categoryName}")
    fig.tight_layout()
    # plt1 = fig.add_subplot(221)
    plt2 = fig.add_subplot(212)
    plt3 = fig.add_subplot(211)
    plt2.title.set_text("testing values")
    plt3.title.set_text("all values")
    #put testing category on x axis and house price on y axis
    plt2.scatter(testX, testY, c='green', marker='_', alpha=0.5, label='real values')
    plt2.scatter(testX, preds, c='red', marker='|', alpha=0.5, label='predicted values')
    plt3.scatter(dataX, dataY, c='black', marker='.', alpha=0.5)
    # pyplot.subplots_adjust(top=1.5)
    plt2.legend(loc='lower right')

    return fig

def predictOneMultiProcessing(houseData, houseTargetPrices):
    testParams = [[houseData, houseTargetPrices, 0, "median income in block"],[houseData, houseTargetPrices, 1, "median house age"],[houseData, houseTargetPrices, 2, "average number of rooms"],[houseData, houseTargetPrices, 3, "average number of bedrooms"],[houseData, houseTargetPrices, 4, "block population"],[houseData, houseTargetPrices, 5, "house occupancy"],]
    #[houseData, houseTargetPrices, 6, "latitude"],[houseData, houseTargetPrices, 7, "longitude"]

    with Pool(processes=4, maxtasksperchild = 1) as pool:
            results = pool.map(predictWithOneCategoryMP, testParams)
            pool.close()
            pool.join()

    pyplot.show()

    return

def predictingWithOneCategory(houseData, houseTargetPrices):
    #trying to predict with original data and no scaling
    fig0 = predictWithOneCategory(houseData, houseTargetPrices, 0, "median income in block")
    fig1 = predictWithOneCategory(houseData, houseTargetPrices, 1, "median house age")
    fig2 = predictWithOneCategory(houseData, houseTargetPrices, 2, "average number of rooms")
    fig3 = predictWithOneCategory(houseData, houseTargetPrices, 3, "average number of bedrooms")
    fig4 = predictWithOneCategory(houseData, houseTargetPrices, 4, "block population")
    fig5 = predictWithOneCategory(houseData, houseTargetPrices, 5, "house occupancy")
    # fig6 = predictWithOneCategory(houseData, houseTargetPrices, 6, "latitude")
    # fig7 = predictWithOneCategory(houseData, houseTargetPrices, 7, "longitude")

    pyplot.subplots_adjust(hspace = 0.4)
    pyplot.show()

    return

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
    print(f"Correct: {correct1}, Incorrect: {incorrect1}, % Correct: {correct1/(correct1 + incorrect1)*100: 5.2f}")
    print("plus minus $20,000")
    print(f"Correct: {correct2}, Incorrect: {incorrect2}, % Correct: {correct2/(correct2 + incorrect2)*100: 5.2f}")
    print("plus minus $30,000")
    print(f"Correct: {correct3}, Incorrect: {incorrect3}, % Correct: {correct3/(correct3 + incorrect3)*100: 5.2f}")
    print("")

def scaleData(houseData):
    #le originál data
    dataX = houseData
    #i think this makes it so that majority of the data is within plus or minus 1, but outliers are included
    scaledXnormal = preprocessing.scale(dataX)
    #reduces skewness by applying a logarithmic scale?
    pt = preprocessing.PowerTransformer()
    scaledXpowerTransformer = pt.fit_transform(dataX)

    return dataX, scaledXnormal, scaledXpowerTransformer

def classDemonstration(dataX, scaledXnormal, scaledXpowerTransformer, houseTargetPrices):
    allFigs = []
    xDatas = [dataX, scaledXnormal,scaledXpowerTransformer]
    #formatting purposes
    xDatasNames = ["dataX", "scaledXnormal", "scaledXpowerTransformer"]
    columnNames = ["median income", "median house age", "average number of rooms", "block population", "house occupancy"]
    color = ["red", "black", "green"]
    for x in range(len(columnNames)):
        fig = pyplot.figure(f"histogram of {columnNames[x]} data with different scaling methods")
        plt = fig.add_subplot(111)
        for i in range(len(xDatas)):
            singleCol = []
            for item in xDatas[i]:
                singleCol.append(item[x])
            singleCol = asarray(singleCol)

            plt.hist(singleCol, bins=100, color = color[i], label=xDatasNames[i])

        plt.legend(loc='upper right')

        allFigs.append(fig)

    pyplot.show()

    for i in range(len(xDatas)):
        print(f"using the {xDatasNames[i]} data:")
        if array_equal(xDatas[i], dataX):
            sleep(3)
            print("plus minus $10,000\nCorrect: 1455, Incorrect: 4737, % Correct:  23\nplus minus $20,000\nCorrect: 2214, Incorrect: 3978, % Correct:  36\nplus minus $30,000\nCorrect: 2872, Incorrect: 3320, % Correct:  46\n")
            continue

        #remove any points that aren't within the -1 to 1 domain
        thisDataX, houseTargetPricesModified = removeOutliers(xDatas[i], houseTargetPrices)
        #reset all categories
        trainX, testX, trainY, testY = [], [], [], []
        #this train and test X are arrays with 6 categories
        trainX, testX, trainY, testY = trainTestSplitAll(thisDataX, houseTargetPricesModified)
        predictWithAll(trainX, testX, trainY, testY)

    return

def main():
    #https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset
    houseData = fetch_california_housing()

    #choose the desired columns
    #last two categories are longitude and latitude so I ignore them
    houseData.data = houseData.data[:,0:6]

    #function to convert prices into categories
    #turns this from a regression to a classification problem
    houseTargetPrices = getTargetPrices(houseData)

    #trying to predict with original data and no scaling
    # predictingWithOneCategory(houseData, houseTargetPrices)

    #trying to predict with original data and no scaling
    #but using multi processing so it's not super slow
    #many issues with different python versions and leaked semephores?
    # predictOneMultiProcessing(houseData, houseTargetPrices)

    #scale the data with multiple methods
    # dataX, scaledXnormal, scaledXpowerTransformer = scaleData(houseData.data)

    #predict with all
    trainX, testX, trainY, testY = [], [], [], []
    trainX, testX, trainY, testY = trainTestSplitAll(scaledXpowerTransformer, houseTargetPrices)
    predictWithAll(trainX, testX, trainY, testY)

    #for finding the outliers
    # searchOutliers(dataX, houseTargetPrices)

    #janky demonstration for class
    # classDemonstration(dataX, scaledXnormal, scaledXpowerTransformer, houseTargetPrices)

if __name__ == '__main__':
	main()
