from jikanpy import Jikan
import time

class MALHandler(object):
    def __init__(self):
        self.jikan = Jikan()
        self.dataPath = './data/meta'
        self.labelPath = './label/meta'

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
            synopsis = synopsis + '\n'
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

            label.append(str(score) + '\n')
            input.append(synopsis)
        return input, label

    def saveAnimeData(self, year, season):
        input, label = self.getAnimeData(year, season)
        f_input = open('{}/{}_{}_data.txt'.format(self.dataPath, year, season), 'w')
        f_label = open('{}/{}_{}_label.txt'.format(self.labelPath, year, season), 'w')
        f_input.writelines(input)
        f_label.writelines(label)
        f_input.close()
        f_label.close()

    def loadData(self, year, season):
        f_input = open('{}/{}_{}_data.txt'.format(self.dataPath, year, season), 'r')
        f_label = open('{}/{}_{}_label.txt'.format(self.labelPath, year, season), 'r')
        data = []
        label = []
        while True:
            data_line = f_input.readline()
            label_line = f_label.readline()
            if not data_line or not label_line:
                if label_line != data_line:
                    raise RuntimeError('Input and label numbers not match!')
                break
            data.append(data_line)
            label.append(label_line)

    def savaAllSeasonAnimeData(self, year):
        allSeason = ['spring', 'summer', 'fall', 'winter']
        for season in allSeason:
            self.saveAnimeData(year, season)



