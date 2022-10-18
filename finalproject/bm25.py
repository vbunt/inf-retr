from corpus import Corpus
from numpy import array, argsort
from scipy.sparse import load_npz
from pickle import load


class BM25:

    def __init__(self, documents, size):
        self.embeddings = load_npz(f'{size}/bm25.npz')
        with open(f'{size}/bm25.pickle', 'rb') as file:
            self.vectorizer = load(file)
        self.documents = array(documents)

    def search(self, query):
        query_count_vec = self.vectorizer.transform([Corpus.clean_text(query, stopwords=False)]).toarray()
        scores = self.embeddings * query_count_vec.transpose()
        sorted_scores_indx = argsort(scores, axis=0)[::-1]
        return self.documents[sorted_scores_indx.ravel()][:5]

# with open('dirty_answers.txt', 'r') as file:
#     dirty_documents = file.read().split('\n')[:-1]
# print(BM25(dirty_documents).search('возможно'))
