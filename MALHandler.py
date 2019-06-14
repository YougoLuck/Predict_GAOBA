from jikanpy import Jikan
import time


class MALHandler(object):
    def __init__(self):
        self.jikan = Jikan()
        self.fileHandler = FileHandler()
        self.dataPath = './data/meta'
        self.labelPath = './label/meta'
        self.allSeoson = ['spring', 'summer', 'fall', 'winter']

    def getDetailSynopsisAndScore(self, id):
        detail_anime = self.jikan.anime(id)
        return detail_anime['score']

    def getAnimeData(self, year, season):
        print('Getting all anime in {} {}'.format(year, season))
        animeData = self.jikan.season(year = year, season = season)
        input = []
        label = []
        for anime in animeData['anime']:
            print('total:{}, processed:{}'.format(len(animeData['anime']), len(input)))
            synopsis = anime['synopsis']
            synopsis = ' '.join(synopsis.splitlines())
            synopsis = synopsis.replace('  ', ' ')
            if anime['score'] is None:
                while True:
                    try:
                        score = self.getDetailSynopsisAndScore(anime['mal_id'])
                    except:
                        time.sleep(10)
                    else:
                        break
            else:
                score = anime['score']

            label.append(str(score))
            input.append(synopsis)
        return input, label




    def saveAnimeData(self, year, season):
        input, label = self.getAnimeData(year, season)
        self.fileHandler.saveFileHandler('{}/{}_{}_data.txt'.format(self.dataPath, year, season), input)
        self.fileHandler.saveFileHandler('{}/{}_{}_label.txt'.format(self.labelPath, year, season), label)


    def loadData(self, year, season):

        data = self.fileHandler.loadFileHandler('{}/{}_{}_data.txt'.format(self.dataPath, year, season))
        label = self.fileHandler.loadFileHandler('{}/{}_{}_label.txt'.format(self.labelPath, year, season))
        if len(data) != len(label):
            raise RuntimeError('Input and label numbers not match!')
        return data, label

    def savaAllSeasonAnimeData(self, year):
        for season in self.allSeason:
            self.saveAnimeData(year, season)


class FileHandler(object):
    def saveFileHandler(self, path, data):
        f = open(path, 'w')
        temData = [tem + '\n' for tem in data]
        f.writelines(temData)
        f.close()

    def loadFileHandler(self, path):
        f = open(path, 'r')
        data = []
        while True:
            line = f.readline()[:-1]
            data.append(line)
            if not line:
                break
        data.pop()
        f.close()
        return data
