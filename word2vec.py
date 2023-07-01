from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["kitten", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)

vector = model.wv['cat']
#print('vector cat:')
#print(vector)

vector = model.wv['kitten']
#print('vector kitten:')
#print(vector)

vector = model.wv['dog']
#print('vector dog:')
#print(vector)

similar_words = model.wv.most_similar('cat')
print("similar to cat:")
print(similar_words)

similar_words = model.wv.most_similar('meow')
print("similar to meow:")
print(similar_words)

