from corpus import Corpus
from as_matrix import Matrix
from as_dictionary import Dictionary


def main():
    path = input('path to directory: ')
    c = Corpus(path=path)  # preprocess corpus
    m = Matrix(c.data)  # make a matrix
    d = Dictionary(matrix=m)  # make a dictionary
    print()

    print('-- Из матрицы --')
    print()
    m.most_frequent_word()
    print()
    m.rare()
    print()
    m.common_words()
    print()
    m.most_frequent_name()
    print('\n')

    print('-- Из словаря --')
    print()
    d.most_frequent_word()
    print()
    d.rare()
    print()
    d.common_words()
    print()
    d.most_frequent_name()


if __name__ == '__main__':
    main()
