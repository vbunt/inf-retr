import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Matrix:

    def __init__(self, corpus):
        vectorizer = CountVectorizer(analyzer='word')
        self.full = vectorizer.fit_transform(corpus)  # 2D document-term matrix
        self.freq = np.asarray(self.full.sum(axis=0)).ravel()  # 1D word frequency matrix
        self.words = vectorizer.get_feature_names()  # list of terms

    def most_frequent_word(self):
        print('самое частое слово')
        print('частота: ',
              self.freq[np.unravel_index(self.freq.argmax(axis=None),
                                         self.freq.shape)[0]],
              '\nслово: ',
              self.words[np.unravel_index(self.freq.argmax(axis=None),
                                          self.freq.shape)[0]])

    def rare(self):
        # in a big enough corpora there will have to be at least one word with frequency 1
        # no need to consider other frequencies
        rares = []
        for el in np.argwhere(self.freq == 1):
            rares.append(self.words[el[0]])
        print('слов частоты 1: ', len(rares))

    def common_words(self):
        print('слова, которые встречаются в каждом документе: ')
        for i in range(self.full.transpose().shape[0]):
            # checking if there are 0 in any columns (term-document matrix)
            if np.all(self.full.transpose()[i].toarray()):
                print(self.words[i])

    def most_frequent_name(self):

        names = [['моника', 'мон'],
                 ['рейчел', 'рейч'],
                 ['чендлер', 'чэндлер', 'чен'],
                 ['фиби', 'фибс'],
                 ['росс'],
                 ['джоуи', 'джо']]
        # deleted джои from the list as it was not in corpus

        d_names = {}
        for name in names:
            d_names[name[0]] = 0
            for var in name:
                ind = self.words.index(var)
                d_names[name[0]] += self.freq[ind]

        print('самый часто упоминающийся герой: ', max(d_names, key=d_names.get))
