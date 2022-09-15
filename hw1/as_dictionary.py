from as_matrix import Matrix


class Dictionary:

    def __init__(self, corpus=None, matrix=None):
        self.data = {}
        # create dictionary from matrix
        if not matrix:
            # if we don't have matrix, create it
            m = Matrix(corpus)
        else:
            # if we already have matrix
            m = matrix

        for i in range(m.full.transpose().shape[0]):
            word = m.words[i]
            self.data[word] = {}
            for j in range(m.full.transpose().shape[1]):
                self.data[word][j] = m.full.transpose()[i, j]

    def most_frequent_word(self):
        max_ = 1
        max_key = ''
        for key, value in self.data.items():
            if sum(value.values()) > max_:
                max_key = key
                max_ = sum(value.values())
        print('самое частое слово')
        print('частота:', max_)
        print('слово', max_key)

    def rare(self):
        rares = []
        # in a big enough corpora there will have to be at least one word with frequency 1
        # no need to consider other frequencies
        for key, value in self.data.items():
            if sum(value.values()) == 1:
                rares.append(key)
        print('слов частоты 1: ', len(rares))

    def common_words(self):
        print('слова, которые встречаются в каждом документе: ')
        for key, value in self.data.items():
            # checking for 0 in frequencies
            if 0 not in value.values():
                print(key)

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
                d_names[name[0]] += sum(self.data[var].values())

        print('самый часто упоминающийся герой: ', max(d_names, key=d_names.get))
