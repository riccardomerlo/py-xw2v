from model_torch import Word2VecModel
from dataset_torch import create_skipgram, read_corpus
import torch
import pickle

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)

"""
Define CONSTANTS
"""

BATCH_SIZE = 32
NEGATIVES = 5
EPOCHS = 2
SAMPLING_RATE = 1E-3
MIN_FREQ = 1
WINDOW_SIZE = 3
LEARNING_RATE = 1E-3
# liste_termini_weat
S = ["technology"]
T = ["art"]
A = ["man", "he", "him", "his"]
B = ["woman", "she", "her"]

WEATLIST = S+T+A+B

"""
Read CORPUS
"""
text = read_corpus('./corpus/nyt_articles_31.txt')
print(text)

"""
Create DATASET
"""
data, unigram_counts, vocab, inv_vocab = create_skipgram(
    text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

print(unigram_counts)


with open('data_small.pkl', 'wb') as f:
  pickle.dump(data, f)

with open('unigram_counts_small.pkl', 'wb') as f:
  pickle.dump(unigram_counts, f)

with open('vocab_small.pkl', 'wb') as f:
  pickle.dump(vocab, f)

with open('inv_vocab_small.pkl', 'wb') as f:
  pickle.dump(inv_vocab, f)



"""
Create MODEL
"""
word2vec = Word2VecModel(unigram_counts,
                         hidden_size=300,
                         batch_size=BATCH_SIZE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE)

"""
Train MODEL
"""
word2vec.train(data, save=True)
