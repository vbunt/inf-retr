from nltk.tokenize import word_tokenize
import os
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation

morph = pymorphy2.MorphAnalyzer()
STOPWORDS = stopwords.words('russian')
PUNCTUATION = punctuation + "...``«»\'\'—"


class Corpus:

    def __init__(self, path, filenames=False):
        # path to main directory
        self.data = []
        self.names = []
        for dir_ in os.listdir(path):
            for file_ in os.listdir(f'{path}/{dir_}'):
                self.names.append(file_)
                with open(f'{path}/{dir_}/{file_}',
                          mode='r',
                          encoding='utf-8-sig') as f:
                    f = f.read()
                    # delete ends
                    f = f.replace('9999\n00:00:0,500 --> 00:00:2,00\nwww.tvsubtitles.net', '')
                    self.data.append(Corpus.clean_text(f))

        # make file with document names from corpus
        if filenames:
            with open('episode_names.txt', 'w') as file:  # creates file with filenames from corpus
                for name in self.names:
                    file.write(name)
                    file.write('\n')

    @staticmethod
    def clean_text(file):
        text = []
        for word in word_tokenize(file):
            if word not in PUNCTUATION:  # delete punctuation
                lemma = morph.parse(word)[0].normal_form
                # pymorphy lowers strings
                if lemma not in STOPWORDS:  # deleted stop-words
                    text.append(lemma)
        return ' '.join(text)
