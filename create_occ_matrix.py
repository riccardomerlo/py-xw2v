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
import itertools

from dataset_torch import read_corpus, split_given_size
from collections import defaultdict as ddict

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

flat_data = [x for xs in data for x in xs]
print('total number of tuples: ', len(flat_data))

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

list_flatweat = []
for step, training_point in enumerate(flat_data):
  inputs = training_point[0]
  for word in list_index_weat:
    if word==inputs: # keep a batch if it contains at least one weat word
      list_flatweat.append(training_point)
print("Tot. number of inputs to consider in total (no batches): ", len(list_flatweat))

with open("list_unique_weat.pkl", "rb") as v:
  list_uniqueweat = pickle.load(v)


group = ddict(list)

for target, label, nsent in list_flatweat:
  group[nsent].append((target, label))


class MultiLabelCounter():
    def __init__(self, classes=None):
        self.classes_ = classes

    def fit(self,y):
        self.classes_ = sorted(set(itertools.chain.from_iterable(y)))
        self.mapping = dict(zip(self.classes_,
                                         range(len(self.classes_))))
        return self

    def transform(self,y):
        yt = []
        for labels in y:
            data = [0]*len(self.classes_)
            for label in labels:
                data[self.mapping[label]] +=1
            yt.append(data)
        return yt

    def fit_transform(self,y):
        return self.fit(y).transform(y)

mlc = MultiLabelCounter()
occ_matrix = mlc.fit_transform(list(group.values()))

with open("group_nsent.pkl", "wb") as v:
  pickle.dump(group, v)

with open("occ_matrix.pkl", "wb") as v:
  pickle.dump(occ_matrix, v)
