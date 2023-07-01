from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer
import torch

def get_bert_embeddings(sentence):

    print("get_bert_embeddings() received sentence as input: [" + sentence + "]")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(sentence, return_tensors='pt')
    # print("inputs:")
    # print(inputs)

    outputs = model(**inputs)
    # print("outputs:")
    # print(outputs)

    # The BERT model returns 2 values:
    # 1. last_hidden_state (sequence of hidden-states at the output of the last layer of the model)
    # 2. pooler_output (last layer hidden-state of the first token of the sequence, further processed by a Linear layer and a Tanh activation function)
    # The outputs are instances of BaseModelOutputWithPoolingAndCrossAttentions
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    return last_hidden_state, pooler_output

def get_sentence_similarity(sentence1, sentence2):

    print("get_sentence_similarity() received these sentences as input:")
    print("[" + sentence1 + "]")
    print("[" + sentence2 + "]")

    _, pooler_output1 = get_bert_embeddings(sentence1)
    _, pooler_output2 = get_bert_embeddings(sentence2)

    # We detach the tensors from the computational graph and convert them to numpy arrays for use with scipy
    pooler_output1 = pooler_output1.detach().numpy().squeeze()
    pooler_output2 = pooler_output2.detach().numpy().squeeze()

    similarity = 1 - cosine(pooler_output1, pooler_output2)

    return similarity

if __name__ == "__main__":
    sentence = "This is a sample sentence for BERT."
    last_hidden_state, pooler_output = get_bert_embeddings(sentence)
    # print("last_hidden_state:")
    # print(last_hidden_state)
    # print("pooler_output:")
    # print(pooler_output)
    print("Last hidden state shape:", last_hidden_state.shape)
    print("Pooler output shape:", pooler_output.shape)

    sentence1 = "This is a sample sentence for BERT."
    sentence2 = "BERT is used in this sample sentence."
    sentence3 = "I love to play football."
    print("\n\n")
    print("Sentence1 == " + sentence1)
    print("Sentence2 == " + sentence2)
    print("Sentence3 == " + sentence3)

    print("Similarity between sentence 1 and 2:", get_sentence_similarity(sentence1, sentence2))
    print("Similarity between sentence 1 and 3:", get_sentence_similarity(sentence1, sentence3))
