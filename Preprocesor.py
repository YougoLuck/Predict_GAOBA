from MALHandler import MALHandler, FileHandler
import os
import re


class Preprocessor(object):

    def __init__(self):
        self.malHandler = MALHandler()
        self.fileHandler = FileHandler()
        self.saveDataPath = './data'
        self.saveLabelPath = './label'
        self.allData = []
        self.splitData = []
        self.allLabel = []


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
        self.fileHandler.saveFileHandler('{}/preprocessed_data.txt'.format(self.saveDataPath),
                                        self.allData)
        self.fileHandler.saveFileHandler('{}/preprocessed_label.txt'.format(self.saveLabelPath),
                                        self.allLabel)


    def loadPreprocessedData(self):
        allData = self.fileHandler.loadFileHandler('{}/preprocessed_data.txt'.format(self.saveDataPath))
        allLabel = self.fileHandler.loadFileHandler('{}/preprocessed_label.txt'.format(self.saveLabelPath))
        if len(allData) != len(allLabel):
            raise RuntimeError('Input and label numbers not match!')
        self.allData = allData
        self.allLabel = allLabel

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
            if len(dataArr) >= threshold:
                allData.append(data)
                allLabel.append(self.allLabel[i])
        self.allLabel = allLabel
        self.allData = allData


    def cleanUpData(self):
        allData = []
        rule = re.compile(r"[^a-zA-Z ]|( *$)")
        for data in self.allData:
            data = self.removeSourceTag(data)
            data = rule.sub('', data)
            data = data.lower()
            allData.append(data)
        self.allData = allData

    def getSplitData(self):
        splitData = []
        for data in self.allData:
            splitData.append(data.split(' '))
        self.splitData = splitData


test = Preprocessor()
test.loadPreprocessedData()
test.getSplitData()
print(test.splitData)

