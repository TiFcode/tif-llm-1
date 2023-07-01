from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer, BertForQuestionAnswering
import torch




def get_bert_answer(question, context):

    print("\n\n")
    print("get_bert_answer() received this Question and this Context as input:")
    print("Question: [" + question + "]")
    print("Context: [" + context + "]")
    print("\n\n")


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


    # Encode the question and context to get input IDs and attention mask
    inputs = tokenizer(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the model's predictions
    outputs = model(input_ids, attention_mask=attention_mask)

    # The model returns the predicted start and end indices of the answer
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    # Use the tokenizer to convert the indices to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(tokens)

    return answer


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

    sentence1 = "Today the weather will be fine."
    sentence2 = "Today the weather will be nice."
    sentence3 = "I am hungry."

    print("\n\n")
    print("Sentence1 == " + sentence1)
    print("Sentence2 == " + sentence2)
    print("Sentence3 == " + sentence3)
    print("\n\n")

    print("Similarity between sentence 1 and 2:", get_sentence_similarity(sentence1, sentence2))
    print("\n\n")
    print("Similarity between sentence 1 and 3:", get_sentence_similarity(sentence1, sentence3))
    print("\n\n")


    question = "Who won the world series in 2020?"
    context = "The 2020 World Series was won by the Los Angeles Dodgers."

    answer = get_bert_answer(question, context)

    print("Answer:", answer)
