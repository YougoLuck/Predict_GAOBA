from MALHandler import MALHandler, FileHandler
from collections import Counter
import os
import re
import numpy as np
import random

class Preprocessor(object):

    def __init__(self):
        self.malHandler = MALHandler()
        self.fileHandler = FileHandler()
        self.savePath = '.'
        self.intData = []
        self.allLabel = []
        self.vocabToInt = {}
        self.intToVocab = {}


    def loadAllMetaDataAndLabel(self):
        fileList = os.listdir(self.malHandler.dataPath)
        yearList = []
        allData = []
        allLabel = []
        for filename in fileList:
            yearList.append(filename.split('_')[0])
        for year in yearList:
            for season in self.malHandler.allSeoson:
                data, label = self.malHandler.loadData(year, season)
                allData = allData + data
                allLabel = allLabel + label
        return allData, allLabel

    def savePreprocessedData(self):
        self.fileHandler.saveMetaFileHandler('{}/preprocessed_data.txt'.format(self.savePath),
                                             self.intData)
        self.fileHandler.saveMetaFileHandler('{}/preprocessed_label.txt'.format(self.savePath),
                                             self.allLabel)
        self.fileHandler.saveMetaFileHandler('{}/vocab_to_int.txt'.format(self.savePath),
                                             self.vocabToInt)

    def loadPreprocessedData(self, seqLen):
        intData = self.fileHandler.loadMetaFileHandler('{}/preprocessed_data.txt'.format(self.savePath))
        allLabel = self.fileHandler.loadMetaFileHandler('{}/preprocessed_label.txt'.format(self.savePath))
        print('Data cnt before Clean up: {}'.format(len(allLabel)))
        self.loadVocabToInt()
        if len(intData) != len(allLabel):
            raise RuntimeError('Input and label numbers not match!')

        self.intData = self.converIntDataToFeatures(intData, seqLen)
        self.allLabel = np.array(allLabel, dtype = np.float)
        return self.intData, self.allLabel


    def generateVocabToInt(self, allData):
        data = ' '.join(allData)
        words = data.split()
        counts = Counter(words)
        vocab = sorted(counts, key = counts.get, reverse = True)
        vocabToInt = {word: ii for ii, word in enumerate(vocab, 1)}
        self.vocabToInt = vocabToInt
        self.getIntToVocab()

    def loadVocabToInt(self):
        path = '{}/vocab_to_int.txt'.format(self.savePath)
        vocabToInt = self.fileHandler.loadMetaFileHandler(path)
        self.vocabToInt = vocabToInt
        self.getIntToVocab()

    def getIntToVocab(self):
        intToVocab = {value: key for key, value in self.vocabToInt.items()}
        self.intToVocab = intToVocab

    def converDataToInt(self, allData):
        intData = []
        for data in allData:
            intData.append([self.vocabToInt[word] for word in data.split() if word in self.vocabToInt])
        return intData

    def converIntDataToFeatures(self, intData, seqLen):
        features = np.zeros((len(intData), seqLen), dtype = int)
        for i, row in enumerate(intData):
            features[i, -len(row):] = np.array(row)[:seqLen]
        return features

    def removeSourceTag(self, data):
        index = data.find('(Source')
        if index >= 0:
            data = data[:index]

        index = data.find('[Written')
        if index >= 0:
            data = data[:index]
        return data

    def removeShortData(self, allData, allLabel, threshold):
        newData = []
        newLabel = []
        for i in range(len(allData)):
            data = allData[i]
            dataArr = data.split(' ')
            if len(dataArr) >= threshold and allLabel[i] != 'None':
                newData.append(data)
                newLabel.append(allLabel[i])
        return newData, newLabel

    def cleanUpData(self, allData):
        newData = []
        rule = re.compile(r"[^a-zA-Z ]|( *$)")
        for data in allData:
            data = self.removeSourceTag(data)
            data = rule.sub(' ', data)
            data = data.lower()
            newData.append(data)
        return newData

    def increaseLabelScale(self, allLabel, scale):
        newLabel = [float(label) * scale for label in allLabel]
        return newLabel

    def converLabelTo2Category(self, allLabel, threshold):
        newLabel = [1 if float(label) > threshold else 0 for label in allLabel]
        return newLabel


    def shuffle(self, intData, allLabel):
        allIndex = list(range(len(intData)))
        random.shuffle(allIndex)
        shuffleData = []
        shuffleLabel = []
        for index in allIndex:
            shuffleData.append(intData[index])
            shuffleLabel.append(allLabel[index])
        return shuffleData, shuffleLabel


    def runIntial(self, dataThreshold, labelScale):

        #加载原始数据
        allData, allLabel = self.loadAllMetaDataAndLabel()
        #过滤数据
        allData = self.cleanUpData(allData)
        allData, allLabel = self.removeShortData(allData, allLabel, dataThreshold)
        #生成vocab
        self.generateVocabToInt(allData)
        #转换数据到Int
        intData = self.converDataToInt(allData)
        #加大scale
        allLabel = self.increaseLabelScale(allLabel, labelScale)
        #shuffle
        self.intData, self.allLabel = self.shuffle(intData, allLabel)
        #保存
        self.savePreprocessedData()


    def runIntial2Category(self, dataThreshold, labelThreshold):
        #加载原始数据
        allData, allLabel = self.loadAllMetaDataAndLabel()
        #过滤数据
        allData = self.cleanUpData(allData)
        allData, allLabel = self.removeShortData(allData, allLabel, dataThreshold)
        #生成vocab
        self.generateVocabToInt(allData)
        #转换数据到Int
        intData = self.converDataToInt(allData)
        #加大scale
        allLabel = self.converLabelTo2Category(allLabel, labelThreshold)
        #shuffle
        self.intData, self.allLabel = self.shuffle(intData, allLabel)
        #保存
        self.savePreprocessedData()


test = Preprocessor()
test.runIntial2Category(20, 6.5)

