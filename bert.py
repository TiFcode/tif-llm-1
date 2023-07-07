from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
import torch

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
chunk_max_length = 500 # 510 is the max length that BERT can handle minus 2 for [CLS] and [SEP]

def get_bert_answer(question, context):

    # print()
    # print("get_bert_answer() received Question and Context as input.")
    # print("Question: [" + question + "]")
    #print("Context: [" + context + "]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids'].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #print("\n\n[Test tokens:]", text_tokens, "\n\n")
    outputs = model(**inputs)

    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    score = answer_start_scores[0, answer_start].item() + answer_end_scores[0, answer_end-1].item()

    # print("[Answer:]", answer)
    # print("[Score:]", score)

    # print(f"\n{score:.2f}: {answer}")

    return answer, score


def chunk_text(text, max_length):
    #print("[START] Creating chunks")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        tokens = tokenizer.tokenize(word)
        current_length += len(tokens)
        if current_length <= max_length - 2: # Reserve space for [CLS] and [SEP]
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(tokens)

    if current_chunk: # Append any remaining words as a chunk
        chunks.append(" ".join(current_chunk))

    print(f"INFO: Splitted the context.txt into {len(chunks)} chunks, each having {chunk_max_length} tokens max.")
    #print("[END] Creating chunks")

    return chunks


def get_bert_embeddings(sentence):

    print("get_bert_embeddings() received sentence as input: [" + sentence + "]")

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

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
    # sentence = "This is a sample sentence for BERT."
    # last_hidden_state, pooler_output = get_bert_embeddings(sentence)
    # # print("last_hidden_state:")
    # # print(last_hidden_state)
    # # print("pooler_output:")
    # # print(pooler_output)
    # print("Last hidden state shape:", last_hidden_state.shape)
    # print("Pooler output shape:", pooler_output.shape)

    # sentence1 = "Today the weather will be fine."
    # sentence2 = "Today the weather will be nice."
    # sentence3 = "I am hungry."

    # print("\n\n")
    # print("Sentence1 == " + sentence1)
    # print("Sentence2 == " + sentence2)
    # print("Sentence3 == " + sentence3)
    # print("\n\n")

    # print("Similarity between sentence 1 and 2:", get_sentence_similarity(sentence1, sentence2))
    # print("\n\n")
    # print("Similarity between sentence 1 and 3:", get_sentence_similarity(sentence1, sentence3))
    # print("\n\n")

    #question = "Who won the world series in 2020?"
    question = "I am a content creator. How can I create semantic annotations for Ontosense?"
    print("[Question]")
    print(question)
    print("\n[Processing]")

    #context = "The 2020 World Series was won by the Los Angeles Dodgers."
    # Read the context from a text file
    with open('context.txt', 'r') as file:
        context = file.read().replace('\n', ' ')

    # Split context into chunks
    chunks = chunk_text(context, chunk_max_length) 

    # Use BERT to answer question for each chunk
    print(f"INFO: Using model [{model_name}].")
    answers = [get_bert_answer(question, chunk) for chunk in chunks]

    # Rank answers by score and select best one
    best_answer = max(answers, key=lambda x: x[1])

    # Sort answers by score in descending order
    answers = sorted(answers, key=lambda x: x[1], reverse=True)


    # Determine the maximum score
    max_score = max(score for _, score in answers)

    # Generate histogram
    histogram = ""
    for answer, score in answers:
        normalized_score = score / max_score
        bar_length = int(normalized_score * 20)  # Scale the score to fit a bar length of 20
        histogram += f"{score:.2f}: {answer}\n"
        histogram += f"{'*' * bar_length}\n"

    # Print the histogram
    print("\n[Answers]")
    print(histogram)

    # Format and print the best answer
    best_answer, best_score = best_answer
    print("[Best answer]")
    print(f"{best_score:.2f}: {best_answer}")