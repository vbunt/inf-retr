from corpus import Corpus
import numpy as np


class Search:

    def __init__(self, matrix, vectorizer, doc_names):
        self.matrix = matrix
        self.vectorizer = vectorizer
        self.doc_names = np.array(doc_names)

    def search(self, query, n=5):
        query_count_vec = self.vectorizer.transform([Corpus.clean_text(query, stopwords=False)]).toarray()

        scores = self.matrix * query_count_vec.transpose()
        sorted_scores_indx = np.argsort(scores, axis=0)[::-1]

        return self.doc_names[sorted_scores_indx.ravel()][:n]
