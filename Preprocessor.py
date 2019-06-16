from MALHandler import MALHandler, FileHandler
from collections import Counter
import os
import re
import numpy as np

class Preprocessor(object):

    def __init__(self):
        self.malHandler = MALHandler()
        self.fileHandler = FileHandler()
        self.savePath = '.'
        self.allData = []
        self.allLabel = []
        self.intData = []
        self.vocabToInt = {}
        self.IntToVocab = {}


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
        self.allData = allData
        self.allLabel = allLabel

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
        vocabToInt = self.fileHandler.loadMetaFileHandler('{}/vocab_to_int.txt'.format(self.savePath))
        if len(intData) != len(allLabel):
            raise RuntimeError('Input and label numbers not match!')
        self.vocabToInt = vocabToInt
        features = np.zeros((len(intData), seqLen), dtype = int)
        for i, row in enumerate(intData):
            features[i, -len(row):] = np.array(row)[:seqLen]
        self.intData = features
        self.allLabel = np.array(allLabel, dtype = np.float)

    def removeSourceTag(self, data):
        index = data.find('(Source')
        if index >= 0:
            data = data[:index]

        index = data.find('[Written')
        if index >= 0:
            data = data[:index]
        return data

    def removeShortData(self, threshold):
        allData = []
        allLabel = []
        for i in range(len(self.allData)):
            data = self.allData[i]
            dataArr = data.split(' ')
            if len(dataArr) >= threshold and self.allLabel[i] != 'None':
                allData.append(data)
                allLabel.append(self.allLabel[i])
        self.allLabel = allLabel
        self.allData = allData

    def cleanUpData(self):
        allData = []
        rule = re.compile(r"[^a-zA-Z ]|( *$)")
        for data in self.allData:
            data = self.removeSourceTag(data)
            data = rule.sub(' ', data)
            data = data.lower()
            allData.append(data)
        self.allData = allData

    def increaseLabelScale(self, scale):
        allLabel = [float(label) * scale for label in self.allLabel]
        self.allLabel = allLabel

    def generateW2V(self):
        data = ' '.join(self.allData)
        words = data.split()
        counts = Counter(words)
        vocab = sorted(counts, key = counts.get, reverse = True)
        vocabToInt = {word: ii for ii, word in enumerate(vocab, 1)}
        self.vocabToInt = vocabToInt
        intData = []
        for data in self.allData:
            intData.append([vocabToInt[word] for word in data.split()])
        self.intData = intData


    def run(self, dataThreshold, labelScale):
        self.loadAllMetaDataAndLabel()
        self.cleanUpData()
        self.removeShortData(dataThreshold)
        self.increaseLabelScale(labelScale)
        self.generateW2V()
        self.savePreprocessedData()


