from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz
import pickle
from corpus import Corpus


class Matrix:

    def __init__(self,
                 doc_names,
                 corpus=None,
                 save=False,
                 indexed_corpus=None,
                 vectorizer=None):

        # file with document names is required
        with open(doc_names, 'r') as file:
            self.names = file.read().split('\n')[:-1]

        if bool(indexed_corpus) != bool(vectorizer):
            raise Exception('indexed-corpus and vectorizer: use either none or both')

        # if we already have indexed corpus and vectorizer
        elif indexed_corpus and vectorizer:
            self.full = load_npz(indexed_corpus)
            with open(vectorizer, 'rb') as file:
                self.vectorizer = pickle.load(file)
            self.words = self.vectorizer.get_feature_names()

        # if we do not have matrix and vectorizer
        elif corpus:
            vectorizer = TfidfVectorizer()

            self.full = vectorizer.fit_transform(corpus)
            self.words = vectorizer.get_feature_names()
            self.vectorizer = vectorizer

            # we can save matrix and vectorizer for further usage
            if save:
                save_npz('indexed_corpus.npz', self.full)
                with open('vectorizer.pickle', 'wb') as file:
                    pickle.dump(self.vectorizer, file)

        else:
            raise Exception('you need either corpus or indexed corpus + vectorizer')

    def ask(self, inp):
        # get vector of query
        vector = self.vectorizer.transform([Corpus.clean_text(inp)])
        # get cosine similarity
        sim = cosine_similarity(vector, self.full)
        # make a dict using document names and similarity matrix
        names_sim = {self.names[i]: sim[0][i] for i in range(len(self.names))}
        # returns sorted list of document names
        return sorted(names_sim.keys(), key=lambda key: names_sim[key], reverse=True)
