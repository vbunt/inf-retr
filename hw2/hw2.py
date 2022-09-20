from matrix import Matrix


def main():
    doc_names = input('path to file with filenames: ')
    indexed_corpus = input('path to .npz with indexed corpus: ')
    vectorizer = input('path to .pickle with vectorizer: ')
    # making a Matrix object
    # two ways to make: from corpus of documents OR matrix + vectorizer
    # hare we use matrix + vectorizer
    m = Matrix(doc_names=doc_names, indexed_corpus=indexed_corpus, vectorizer=vectorizer)
    while True:
        i = input('ask a question? y/n ')
        if i == 'y':
            # show three most similar
            print(m.ask(input('ask! '))[:3])
            print('\n')
        else:
            print('thanks!')
            break


if __name__ == '__main__':
    main()
