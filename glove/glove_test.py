import numpy as np
import glove_optimizer
import data_helpers

'''
# Mock corpus (shamelessly stolen from Gensim word2vec tests)
test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

corpus = [sent.split() for sent in test_corpus]
'''


print("Loading data...")
x, y = data_helpers.load_data()
corpus = x + y
# Split train/test set


model = glove_optimizer.GloVeModel(embedding_size=10, context_size=3, min_occurrences=0,
                            learning_rate=0.05, batch_size=2)
model.fit_to_corpus(corpus)

model.train(num_epochs=1, log_dir="log/example", summary_batch_interval=10000, tsne_epoch_interval=1)

#print model.embedding_for("graph")
#print model.embeddings()
#print(model.embeddings()[model.id_for_word('graph')])