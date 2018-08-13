import csv
import os, datetime, sys
import numpy as np
import matplotlib.pyplot as plt
from data import *
from GMM import GMM

def init():
    trainData = Data()
    with open(os.path.join(input_dir, "train.txt"), newline="") as csvFile:
        trainReader = csv.reader(csvFile, delimiter=" ")
        for row in trainReader:
            trainData.append([eval(row[0]), eval(row[1])], int(row[2]))
    devData = Data()
    with open(os.path.join(input_dir, "dev.txt"), newline="") as csvFile:
        devReader = csv.reader(csvFile, delimiter=" ")
        for row in devReader:
            devData.append([eval(row[0]), eval(row[1])], int(row[2]))
    testData = Data()
    with open(os.path.join(input_dir, "test.csv"), newline="") as csvFile:
        testReader = csv.reader(csvFile, delimiter=',')
        for row in testReader:
            if (row[0] == "id"):
                continue
            testData.append([eval(row[1]), eval(row[2])], 0)
    return (trainData, devData, testData)

def display(data):
    x = data.nx()
    y = data.ny()
    plt.plot(x[y==1, 0],x[y==1, 1],'ro', x[y==2, 0], x[y==2, 1], 'bo')
    plt.show()

def main():
    trainD, devD, testD = init()
    allD = Data(trainD, devD)
    if sys.argv[1] == "display":
        display(allD)
        exit(0)
    if sys.argv[1] == "train":
        # local settings
        x = trainD.nx()
        y = trainD.ny()
        xx = devD.nx()
    else:
        # submit settings
        x = allD.nx()
        y = allD.ny()
        xx = testD.nx()
    gmm1 = GMM(x[y==1], round=500, K=4)
    gmm2 = GMM(x[y==2], round=500, K=4)
    print("GMM1.dist: ",gmm1.pi)
    print("GMM2.dist: ",gmm2.pi)
    r1 = gmm1.predict(xx)*np.sum(y==2)
    r2 = gmm2.predict(xx)*np.sum(y==1)
    result = 1 + (r1<r2)*1
    if sys.argv[1] == "train":
        # local settings
        print("accuracy: ", sum(result==devD.ny())/devD.ny().shape[0])
    else:
        # submit settings
        testD.y = list(result)
        testD.output()
if __name__ == "__main__":
    main()