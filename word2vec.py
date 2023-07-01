from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)

vector = model.wv['cat']
print(vector)

similar_words = model.wv.most_similar('cat')
print(similar_words)

similar_words = model.wv.most_similar('meow')
print(similar_words)