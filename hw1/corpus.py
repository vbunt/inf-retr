from nltk.tokenize import word_tokenize
import os
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation

morph = pymorphy2.MorphAnalyzer()
STOPWORDS = stopwords.words('russian')
PUNCTUATION = punctuation + "...``«»\'\'—"


class Corpus:

    def __init__(self, path):
        # path to main directory
        self.data = []
        for dir_ in os.listdir(path):
            for file_ in os.listdir(f'{path}/{dir_}'):
                with open(f'{path}/{dir_}/{file_}',
                          mode='r',
                          encoding='utf-8-sig') as f:
                    f = f.read()
                    # delete from ends of files
                    f = f.replace('9999\n00:00:0,500 --> 00:00:2,00\nwww.tvsubtitles.net', '')
                    self.data.append(Corpus.clean_text(f))

    @staticmethod
    def clean_text(file):
        text = []
        for word in word_tokenize(file):
            if word not in PUNCTUATION:  # deleted punctuation
                lemma = morph.parse(word)[0].normal_form
                # pymorphy automatically lowers
                if lemma not in STOPWORDS:  # deleted stop-words
                    text.append(lemma)
        return ' '.join(text)
