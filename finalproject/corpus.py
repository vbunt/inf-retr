from nltk.tokenize import word_tokenize
from json import loads
from tqdm import tqdm
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation

morph = pymorphy2.MorphAnalyzer()
STOPWORDS = stopwords.words('russian')
PUNCTUATION = punctuation + "...``«»\'\'—"


class Corpus:

    def __init__(self, json_path, clean_path, dirty_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            corpus = list(f)[:52000]

        with open(clean_path, 'w', encoding='utf-8') as f:
            with open(dirty_path, 'w', encoding='utf-8') as d:
                for q_ in tqdm(corpus):
                    max_ = 0
                    ans = ''
                    q = loads(q_)
                    for answer in q['answers']:
                        if answer['author_rating']['value'] and int(answer['author_rating']['value']) > max_:
                            ans = Corpus.clean_text(answer['text'])
                            name = answer['text']
                            max_ = int(answer['author_rating']['value'])
                    if ans:
                        f.write(ans)
                        f.write('\n')
                        d.write(name)
                        d.write('\n')

    @staticmethod
    def clean_text(file, stopwords=True):
        text = []
        for word in word_tokenize(file):
            if word not in PUNCTUATION:  # delete punctuation
                lemma = morph.parse(word)[0].normal_form
                # pymorphy lowers strings
                if stopwords:
                    if lemma not in STOPWORDS:  # deleted stop-words
                        text.append(lemma)
                else:
                    text.append(lemma)
        return ' '.join(text)
