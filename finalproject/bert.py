from transformers import AutoTokenizer, AutoModel
from torch import load, sum, clamp, no_grad
from numpy import array, argsort, dot
from torch.nn.functional import normalize


class Bert:

    def __init__(self, documents, size):
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru") # загружать с ноутбука?
        self.model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        self.embeddings = normalize(load(f'{size}/bert_tensor.pt')).T
        self.documents = array(documents)

    def search(self, query):
        query_vec = Bert.encode_sentence(query, tokenizer=self.tokenizer, model=self.model)
        out = dot(normalize(query_vec), self.embeddings)
        sorted_scores_indx = argsort(out, axis=1)
        return self.documents[sorted_scores_indx.ravel()][:-6:-1]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def encode_sentence(sentence, tokenizer, model):
        encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with no_grad():
            model_output = model(**encoded_input)
        sentence_embedding = Bert.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding

# with open('dirty_answers.txt', 'r') as file:
#     dirty_documents = file.read().split('\n')[:-1]
#
# print(Bert(dirty_documents).search('кто виноват и что делать'))
