# Information Retrieval

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
