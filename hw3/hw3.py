from corpus import Corpus
from matrix import Matrix
from search import Search
import pickle
from scipy import sparse


def main():
    if input('create corpus? y/n ') == 'y':
        json_path = input('path to .jsonl with data: ')
        docs_path = input('path to .txt for documents: ')
        names_path = input('path to .txt for document names: ')
        Corpus(json_path=json_path, docs_path=docs_path, names_path=names_path)
        print('created corpus')

    if input('create matrix & vectorizer files? y/n ') == 'y':
        corpus_path = input('path to .txt with documents: ')
        matrix_path = input('path to .npz for matrix: ')
        vect_path = input('path to .pickle for vectorizer: ')
        with open(corpus_path, 'r') as f:
            corpus = f.read().split('\n')[:-1]
        m = Matrix(corpus=corpus, matrix_path=matrix_path, vectorizer_path=vect_path, save=True)
        matrix = m.matrix
        vectorizer = m.vect
        print('created matrix & vectorizer')

    else:
        matrix_path = input('path to .npz with matrix: ')
        matrix = sparse.load_npz(matrix_path)
        vect_path = input('path to .pickle with vectorizer: ')
        with open(vect_path, 'rb') as file:
            vectorizer = pickle.load(file)

    docs_path = input('path do .txt with document names: ')
    with open(docs_path, 'r') as file:
        doc_names = file.read().split('\n')[:-1]

    # print(doc_names[:10])

    s = Search(matrix=matrix, vectorizer=vectorizer, doc_names=doc_names)

    print("let's search!")
    q = 1

    while q:
        query = input('your query: ')
        print(s.search(query=query))
        print('\n')
        if input('continue searching? y/n ') != 'y':
            q = 0
            print('thanks!')

if __name__ == '__main__':
    main()