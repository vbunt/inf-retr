from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from numpy import dot, argsort
from torch.nn.functional import normalize


def encode_sentence(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embedding


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def main():

    print('downloading model & tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    tensors_path = input('path to .pt with tensors: ')
    questions_path = input('path to .txt with questions: ')

    sentence_embeddings = torch.load(tensors_path)
    with open(questions_path, 'r') as file:
        names = file.read().split('\n')[:-1]

    print('started evaluating')

    start = 0
    n = 0
    while start < 1000:
        d = defaultdict(list)
        for i in range(start, start + 500):
            d[names[i]].append(i)

        query_vec = encode_sentence(list(d.keys()), tokenizer=tokenizer, model=model)
        out = dot(normalize(query_vec), normalize(sentence_embeddings).T)
        sorted_scores_indx = argsort(out, axis=1)[0:, 0:10]

        for right, test in zip(d.values(), sorted_scores_indx):
            if set(right) & set(test):
                n += len(set(right) & set(test))
        start += 500

    print(n)
    print('incredible accuracy!!')


if __name__ == '__main__':
    main()
