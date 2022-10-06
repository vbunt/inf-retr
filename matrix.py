# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import pickle

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(norm='l2')

k = 2
b = 0.75


class Matrix:

    def __init__(self, corpus, matrix_path=None, vectorizer_path=None, save=False):
        tf = count_vectorizer.fit_transform(corpus)
        tfidf_vectorizer.fit_transform(corpus)
        idf = tfidf_vectorizer.idf_

        len_d = tf.sum(axis=1)
        avdl = len_d.mean()

        A = tf.astype('float64')
        for i, j in zip(*tf.nonzero()):
            A[i, j] = tf[i, j] * idf[j] * (k + 1)

        B_1 = b * len_d / avdl
        for i, j in zip(*B_1.nonzero()):
            B_1[i, j] = (B_1[i, j] - b + 1) * k

        # B = B_1 + tf
        B = tf.astype('float64')
        for i, j in zip(*B.nonzero()):
            B[i, j] = B[i, j] + B_1[i, 0]

        matrix = A
        for i, j in zip(*A.nonzero()):
            matrix[i, j] = A[i, j] / B[i, j]

        self.matrix = matrix
        self.vect = count_vectorizer

        if save:
            sparse.save_npz(matrix_path, matrix)
            with open(vectorizer_path, 'wb') as file:
                pickle.dump(count_vectorizer, file)
