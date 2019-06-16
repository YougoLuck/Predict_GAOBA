from Preprocessor import  Preprocessor
import random
import numpy as np

class Model(object):

    def __init__(self):
        self.preProcessor = Preprocessor()
        self.preProcessor.loadPreprocessedData(200)
        self.Intdata, self.Label = self.shuffle(self.preProcessor.intData, self.preProcessor.allLabel)
        self.splitTrainData()
        self.lstmSize = 256
        self.lstmLayers = 1
        self.batchSize = 500
        self.learningRate = 0.001


    def splitTrainData(self):
        splitFrac = 0.8
        splitIdx = int(len(self.Intdata) * splitFrac)
        trainX, valX = self.Intdata[:splitIdx], self.Intdata[splitIdx:]
        trainY, valY = self.Label[:splitIdx], self.Label[splitIdx:]

        testIdx = int(len(valX) * 0.5)
        valX, testX = valX[:testIdx], valX[testIdx:]
        valY, testY = valY[:testIdx], valY[testIdx:]

        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.valX = valX
        self.valY = valY
        print('trainX:{}, trainY:{}'.format(self.trainX.shape, self.trainY.shape))
        print('testX:{}, testY:{}'.format(self.testX.shape, self.testY.shape))
        print('valX:{}, valY:{}'.format(self.valX.shape, self.valY.shape))


    def shuffle(self, allData, allLabel):
        allIndex = list(range(len(allData)))
        random.shuffle(allIndex)
        shuffleData = []
        shuffleLabel = []
        for index in allIndex:
            shuffleData.append(allData[index])
            shuffleLabel.append(allLabel[index])
        return np.array(shuffleData), np.array(shuffleLabel)


test = Model()