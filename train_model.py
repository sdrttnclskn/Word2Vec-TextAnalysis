#python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import gensim.models.word2vec as v
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processes

#pretrained word embeddings
pretrained_emb = "toy_data/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

#input corpus
train_corpus = "toy_data/train_docs.txt"

#output model
saved_path = "toy_data/model.bin"

#enable logging

#train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)
#model= v.Word2Vec.load("toy_data/model.bin")
#save model
model.save(saved_path)

#############graf##################
def tsne_plot(model):
  labels = []
  tokens = []

  for word in model.wv.vocab:
    tokens.append(model[word])
    labels.append(word)

  tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
  new_values = tsne_model.fit_transform(tokens)
   
  x = []
  y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])

  plt.figure(figsize=(10, 10)) 
  for i in range(100):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
               xytext=(5, 2),
               textcoords='offset points',
                  va='bottom')
  plt.show()
            
tsne_plot(model)