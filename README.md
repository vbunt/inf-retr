# Information Retrieval

### hw 3

4 modules:
1. corpus.py - Corpus class; preprocessing; creates a Corpus object; takes .jsonl and creates two .txt files - one with preprocessed documents and another with document names (here: the same documents but raw)
2. matrix.py - Matrix class; creates a Matrix object from corpus (a list of preprocessed documents); two attributes: matrix (a matrix for a matrix for computing BM25) and vect (a fit count vectorizer)
3. search.py - Search class; creates a Search object from matrix, vectorizer and doc_names (document names); method search() takes a query and returns a top n (default 5) best matches according to BM25
4. hw3.py - main module; will ask if you want to create a corpus from .jsonl, then if you want to create a matrix and a vectorizer, then will ask for a query

3 files:
- *doc_names.txt* - file with document names
- *matrix.npz* - indexed matrix for computing BM25
- *vect.pickle* - fit countvectorizer

Run hw3. Choose n, n, then specify paths to matrix (.npz), vectorizer (.pickle) and file with document names (.txt), then ask.

### hw 2

3 modules:
1. corpus.py - Corpus class; preprocessing; creates a Corpus object; can create a file with document names (.txt) if *filenames=True* (default *False*)
2. matrix.py - Matrix class; creates a Matrix object from 1. Corpus object and file with document names (.txt) or 2. matrix file (.npz), vectorizer (.pickle) and file with document names (.txt). can save created matrix and vectorizer if *save=True* (default *False*). function *ask* works with queries.
3. hw2.py - main module; requires three filepaths, then a query

3 files:
- *document_names.txt* - file with document names
- *indexed_matrix.npz* - indexed document-term matrix
- *vectorizer.pickle* - TfIdf-vectorizer

Run hw2.py. Specify paths to the 3 files above. Ask.

### hw 1

4 modules: 
1. corpus.py - preprocessing of text files
2. as_matrix.py - turn corpus into a document-term matrix
3. as_dictionary.py - turn corpus into a dictionary {term : {document : frequency}}
4. hw1.py - main module; run to get *most frequent word*, *name*, *least frequent words* and *words that are in every document*

Run hw1.py. Specify path to *friends-data* directory. 
