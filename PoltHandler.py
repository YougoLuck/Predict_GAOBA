from matplotlib import pyplot as plt
from MALHandler import FileHandler
from matplotlib.pyplot import MultipleLocator

def generatePltImg(title, xLabel, yLabel, x, y):
    fig = plt.figure(0)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(x, y)

def savePltImg(saveName):
    plt.savefig(saveName)
    plt.close(0)
