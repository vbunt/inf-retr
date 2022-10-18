import streamlit as st
from tfidf import TfIdf
from bm25 import BM25
from bert import Bert
import time
import sys


@st.cache(show_spinner=False, allow_output_mutation=True)
def initialize(size):
    with st.spinner('Initializing...'):
        with open(f'{size}/dirty_answers.txt', 'r') as file:
            documents = file.read().split('\n')[:-1]
        tfidf_engine = TfIdf(documents, size)
        bm_engine = BM25(documents, size)
        bert_engine = Bert(documents, size)
        return tfidf_engine, bm_engine, bert_engine


st.title('Boongle')

try:
    size = sys.argv[1]
except IndexError:
    size = 'mid'

tfidf_engine, bm_engine, bert_engine = initialize(size=size)


st.text('Input your query')
query = st.text_input(label='Input your query', label_visibility='collapsed')
st.text('Pick a search method')
answer = ''
dur = 0


col1, col2, col3, col4 = st.columns([1, 1, 1, 7])

with col1:
    tfidf_but = st.button('TfIdf')
with col2:
    bm25_but = st.button('BM25')
with col3:
    bert_but = st.button('Bert')


if bert_but:
    with st.spinner(f'looking for {query} using bert'):
        t1 = time.time()
        answer = bert_engine.search(query)
        dur = time.time() - t1
        st.text('searched with bert')

elif tfidf_but:
    with st.spinner(f'looking for {query} using tfidf'):
        t1 = time.time()
        answer = tfidf_engine.search(query)
        dur = time.time() - t1
        st.text('searched with tfidf')

elif bm25_but:
    with st.spinner(f'looking for {query} using bm25'):
        t1 = time.time()
        answer = bm_engine.search(query)
        dur = time.time() - t1
        st.text('searched with bm25')

if len(answer) != 0:
    for ans in answer:
        st.markdown(ans)

if dur:
    st.text(f'The search took {dur:.4} seconds')
