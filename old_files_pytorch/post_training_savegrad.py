import nltk
nltk.download('punkt')
from scipy import spatial
import statistics
import numpy as np
import pickle
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter

from pytorch.dataset_torch import read_corpus, split_given_size
from pytorch.after_training_torch import _full_loss_torch, _true_loss_torch, _negative_sampling_loss_torch, post_training_simple

BATCH_SIZE = 256
NEGATIVES = 5
EPOCHS = 3
SAMPLING_RATE = 1E-3
MIN_FREQ = 10
WINDOW_SIZE = 5
LEARNING_RATE = 1E-3
HIDDEN_SIZE = 300
# liste_termini_weat
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
WEATLIST = S+T+A+B

syn0 = np.load('syn0_final_torch.npy')
syn1 = np.load('syn1_final_torch.npy')

with open("data.pkl", "rb") as f:
  data = pickle.load(f)

with open("vocab.pkl", "rb") as v:
  vocab = pickle.load(v)

with open("unigram_counts.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open("inv_vocab.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

vocab_words = [key for key in vocab]
vocab_len = len(vocab_words)
print("vocab len: ", vocab_len)

"""Create list of most common words to be used together with weat words list"""

most_common_words = []
un = unigram_counts.copy()
count = 50
while True:
  if len(most_common_words) == 50:
    break
  idx = np.argmax(un)
  if inv_vocab[idx] not in WEATLIST.copy():
    if inv_vocab[idx] not in most_common_words:
      most_common_words.append(inv_vocab[idx])
  un.pop(idx)

list_index_weat = []
for word in S+T+A+B+most_common_words:
  list_index_weat.append(vocab[word])

flat_data = [x for xs in data for x in xs]
print('total number of tuples: ', len(flat_data))

diz_unique = Counter([(x[0], x[1]) for x in flat_data])

list_uniqueweat = []
for step, training_point in enumerate(list(diz_unique.keys())):
  inputs = training_point[0]
  for word in list_index_weat:
    if word==inputs:
      list_uniqueweat.append(training_point)
print("Tot. number of inputs to consider in total (no batches): ", len(list_uniqueweat))

with open("list_unique_weat.pkl", "wb") as v:
  pickle.dump(list_uniqueweat, v)

weights = [torch.from_numpy(syn0).requires_grad_(), torch.from_numpy(syn1).requires_grad_()]


k=0
print("post training for k: ", k)

post_training_simple(k, vocab, inv_vocab, list_uniqueweat, unigram_counts, None, list_index_weat, _true_loss_torch, weights)

k=5
print("post training for k: ", k)

post_training_simple(k, vocab, inv_vocab, list_uniqueweat, unigram_counts, k, list_index_weat, _negative_sampling_loss_torch, weights)

k=vocab_len-1
print("post training for k: ", k)

post_training_simple(k, vocab, inv_vocab, list_uniqueweat, unigram_counts, None, list_index_weat, _full_loss_torch, weights)
