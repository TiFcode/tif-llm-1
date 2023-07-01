from transformers import BertModel, BertTokenizer

def get_bert_embeddings(sentence):

    print("get_bert_embeddings() received sentence as input: [" + sentence + "]")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(sentence, return_tensors='pt')

    print("inputs:")
    print(inputs)
    outputs = model(**inputs)
    print("outputs:")
    print(outputs)

    # The BERT model returns 2 values:
    # 1. last_hidden_state (sequence of hidden-states at the output of the last layer of the model)
    # 2. pooler_output (last layer hidden-state of the first token of the sequence, further processed by a Linear layer and a Tanh activation function)
    # The outputs are instances of BaseModelOutputWithPoolingAndCrossAttentions
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    return last_hidden_state, pooler_output

if __name__ == "__main__":
    sentence = "This is a sample sentence for BERT."
    last_hidden_state, pooler_output = get_bert_embeddings(sentence)
    print("last_hidden_state:")
    print(last_hidden_state)
    print("pooler_output:")
    print(pooler_output)
    print("Last hidden state shape:", last_hidden_state.shape)
    print("Pooler output shape:", pooler_output.shape)
