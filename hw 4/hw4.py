from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from torch.nn.functional import normalize


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_sentence(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embedding


def make_bert_tensor(file, tokenizer, model):
    with open(file, 'r') as f:
        sentences = f.read().split('\n')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    filename = file.split('.')[0].split('/')[1]
    torch.save(sentence_embeddings, f'corpus.pt')


def main():
    print('downloading tokenizer & model')
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    if input('make tensor? y/n (pls no) ') == 'y':
        print('why')
        path = input('path to .txt with corpus: ')
        make_bert_tensor(path, tokenizer=tokenizer, model=model)

    sentence_embeddings = torch.load(input('path to .pt with tensor: '))
    sentences_path = input('path to .txt with answers: ')
    with open(sentences_path) as file:
        sentences = file.read().split('\n')[:-1]

    q = True
    while q:
        query = input('ask! ')
        query_vec = encode_sentence(query, tokenizer=tokenizer, model=model)

        out = np.dot(normalize(query_vec), normalize(sentence_embeddings).T)

        sorted_scores_indx = np.argsort(out, axis=1)
        sentences = np.array(sentences)
        print(*sentences[sorted_scores_indx.ravel()][:-6:-1])

        if input('ask more? y/n ') != 'y':
            q = False


if __name__ == '__main__':
    main()
