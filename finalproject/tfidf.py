from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from pickle import load
from corpus import Corpus


class TfIdf:

    def __init__(self, documents, size):
        self.documents = documents
        self.embeddings = load_npz(f'{size}/tfidf.npz')
        with open(f'{size}/tfidf.pickle', 'rb') as file:
            self.vectorizer = load(file)

    def search(self, query):
        vector = self.vectorizer.transform([Corpus.clean_text(query)])
        sim = cosine_similarity(vector, self.embeddings)
        names_sim = {self.documents[i]: sim[0][i] for i in range(len(self.documents))}
        return sorted(names_sim.keys(), key=lambda key: names_sim[key], reverse=True)[:5]


# with open('big_dirty_answers.txt', 'r') as file:
#     dirty_documents = file.read().split('\n')[:-1]
# print(TfIdf(dirty_documents).search('возможно'))

# print(len(dirty_documents))