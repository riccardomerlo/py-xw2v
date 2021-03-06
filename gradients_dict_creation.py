"""
# Load libraries
"""

import nltk
nltk.download('punkt')
from scipy import spatial
import statistics
#from word_vectors import WordVectors
import numpy as np
import pickle
import pandas as pd
import torch
import torch.utils.data

from typing import List, \
    Union
import os
import sys

import datetime;
import matplotlib.pyplot as plt
# import seaborn as sns
import random
from collections import defaultdict
from collections import Counter


from dataset_torch import read_corpus, split_given_size, get_vocab, apply_reduction, subsample_prob, cache_subsample_prob, get_sampled_sent, get_dynamic_window
from after_training_torch import _negative_sampling_loss_torch, fixed_unigram_candidate_sampler, _full_loss_torch

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                     else torch.FloatTensor)


MODE = 'train' # 'post', 'train'
LOSS = 'NG' # 'NG', 'FULL'

print("MODE", MODE, "---- LOSS", LOSS)

"""# Load data"""

BATCH_SIZE = 256
BATCH_N_SENTENCE = 10
NEGATIVES = 20
EPOCHS = 5
SAMPLING_RATE = 1E-3
MIN_FREQ = 100
WINDOW_SIZE = 5
LEARNING_RATE = 1E-2
HIDDEN_SIZE = 300
_random_seed = 0
# liste_termini_weat
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
WEATLIST = S+T+A+B

output_dir = '/home/apera/py-xw2v/test256_valerio+neg/'

syn0 = np.load(output_dir+'syn0_final_torch.npy')
syn1 = np.load(output_dir+'syn1_final_torch.npy')

with open(output_dir+"vocab_shuffle.pkl", "rb") as v:
  vocab = pickle.load(v)

with open(output_dir+"unigram_counts_shuffle.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open(output_dir+"inv_vocab_shuffle.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

"""
Read CORPUS
"""
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])


list_approx_words = list(set(list_index_weat.copy()))


weights = [torch.from_numpy(syn0).requires_grad_(), torch.from_numpy(syn1).requires_grad_()]

"""## Create post training data"""

def contains_weat(sentence, whitelist):
      for x in sentence:
        if x in whitelist:
          return True

      return False

def get_text(text):

    return iter(text)


def build_dataset(text, max_window, whitelist=[], min_freq=1, sampling_rate=1e-3, dynamic_window=True, batch_n_sentence=None):
    """
    Builds post training dataset by creating tuples that only contain weat words.
    """

    data = []
    vocab = get_vocab(text)
    count_vocab, to_remove_words, to_keep_words = apply_reduction(
        text, vocab, whitelist.copy(), min_freq)

    count_vocab = dict(sorted(count_vocab.items(),key=lambda item: item[1], reverse=False))

    ord_vocab = {key: i for i, key in enumerate(sorted(count_vocab.keys())) }

    unigram_counts = [count_vocab[x] for x in ord_vocab]

    inv_vocab = {v: k for k, v in ord_vocab.items()}

    _text = [[s for s in sentence if s in to_keep_words] for sentence in text]
    _vocab = ord_vocab
    _unigram_counts = unigram_counts
    _vocab_size = len(unigram_counts)
    _inv_vocab = inv_vocab

    S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
        "experiment", "astronomy"]
    T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
    A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    
    _data = []
    _data_weat = []

    step_log = 1000

    subsample_cache = {}
    if sampling_rate != 0:
        subsample_cache = cache_subsample_prob(count_vocab, sampling_rate)

    #cache_sentence = []
    _sent_batch = []
    tmp_batch_sentence = 0
    for n_sent, sentence in enumerate(get_text(_text)):
        if sampling_rate != 0:
            sentence = get_sampled_sent(n_sent, sentence, subsample_cache)
            #cache_sentence.append(sentence)
        
        if batch_n_sentence == 1:
            _sent_batch = [] #support array to keep all tuples of single sentence
        if contains_weat(sentence, whitelist):
          for i, t in enumerate(iter(sentence)):

              if dynamic_window:
                window = get_dynamic_window(n_sent, i, max_window)
              else:
                window = max_window
              contexts = list(range(i-window, i + window+1))
              contexts = [c for c in contexts if c >=
                          0 and c != i and c < len(sentence)]
              for c in contexts:
                  # TARGET, CONTEXT, nsent
                  if BATCH_SIZE == 'auto':
                      _sent_batch.append([_vocab[t], _vocab[sentence[c]], n_sent])
                  else:
                      if t in S+T+A+B: # save tuples with weat
                          _data_weat.append([_vocab[t], _vocab[sentence[c]], n_sent])
                      else:
                          _data.append([_vocab[t], _vocab[sentence[c]], n_sent])

          if (BATCH_SIZE == 'auto') and (batch_n_sentence == 1):
              _data.append(_sent_batch)
            
          tmp_batch_sentence += 1
          if batch_n_sentence!=None and batch_n_sentence < tmp_batch_sentence:
              _data.append(_sent_batch)
              _sent_batch = []
              tmp_batch_sentence = 0
          
          if n_sent % step_log == 0:
              print(round(n_sent/len(_text)*100, 2), end=' ')

    if BATCH_SIZE == 'auto':
        #each sentence is a batch
        #each item is a sentence

        #shuffle data
        random.seed(_random_seed) #ensures reproducibility
        random.shuffle(_data)

        #batch size is the number of tuples of each sentence
        _data_batch = _data

    else:
        #shuffle data before batching
        random.seed(_random_seed) #ensures reproducibility
        random.shuffle(_data)
        
        _data = _data_weat + _data # put weat tuples first
        #batch _inputs and _labels
        _data_batch = split_given_size(_data, BATCH_SIZE)
    
        #remove batch if size < batch_size
        _data_batch = [x for x in _data_batch if len(x) == BATCH_SIZE]

        _data_batch = list(reversed(_data_batch)) # change order of tuples to have weat as the last ones


    #repeat epoch times (done during training)

    _data = _data_batch

    return _data, _vocab, _inv_vocab, _unigram_counts


"""
Build DATASET
"""
data, vocab, inv_vocab, unigram_counts = build_dataset(text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, 0, dynamic_window=False)
print('dataset built')


with open(output_dir+"data_v2.pkl", "wb") as han:
    pickle.dump(data, han)
with open(output_dir+"vocab_v2.pkl", "wb") as han:
    pickle.dump(vocab, han)
with open(output_dir+"inv_vocab_v2.pkl", "wb") as han:
    pickle.dump(inv_vocab, han)
with open(output_dir+"unigram_counts_v2.pkl", "wb") as han:
    pickle.dump(unigram_counts, han)


# """If post training data is needed:"""

# with open("data_v2.pkl", "rb") as han:
#     data = pickle.load(han)
# with open("vocab_v2.pkl", "rb") as han:
#     vocab_v2 = pickle.load(han)
# with open("inv_vocab_v2.pkl", "rb") as han:
#     inv_vocab_v2 = pickle.load(han)
# with open("unigram_counts_v2.pkl", "rb") as han:
#     unigram_counts_v2 = pickle.load(han)


"""If training data is needed (dynamic window and subsampling frequent words):"""

if MODE == 'train':
  with open(output_dir+"data_shuffle.pkl", "rb") as f:
    data = pickle.load(f)


flat_data = [x for y in data for x in y] # 3 secondi
tuple_set = {(x[0], x[1]) for x in iter(flat_data)} # 12 secondi
tuple_set_1 = [x for x in tuple_set if x[0] in list_approx_words] #23s secondi

"""# Dictionaries with counts + gradients
## Creazione dizionari count (tuple_sent, sent_tuple)
"""

# flat data only for words of interest
flat_data_word = [x for x in flat_data if x[0] in iter(list_approx_words)]

diz_count_sent = Counter([tuple(x) for x in flat_data_word])

diz_count_reshaped = np.array([[(x[0][0], x[0][1]), (x[0][2], x[1])] for x in list(diz_count_sent.items())])

tupleDict = defaultdict(list)

for key, val in diz_count_reshaped:
    tupleDict[tuple(key)].append(tuple(val))

tupleDict_reordered = {tuple(k): tupleDict[tuple(k)] for k in tuple_set_1}

"""Save dictionary as pickle"""

if MODE == 'post':
#post training data
  with open(output_dir+"dict_tuple_sent_count_shuffle.pkl", "wb") as f:
    pickle.dump(tupleDict_reordered, f)

if MODE == 'train':
# training data
  with open(output_dir+"dict_tuple_sent_count_traindata_shuffle.pkl", "wb") as f:
    pickle.dump(tupleDict_reordered, f)

"""Dizionario per frasi"""

diz_sent_reshaped = np.array([[x[0][2], ((x[0][0], x[0][1]), x[1])] for x in list(diz_count_sent.items())])

sentDict = defaultdict(list)

for key, val in diz_sent_reshaped:
    sentDict[key].append(tuple(val))

"""Save """

if MODE == 'post':
# post training data
  with open(output_dir+"dict_sent_tuple_count_shuffle.pkl", "wb") as f:
    pickle.dump(sentDict, f)

if MODE == 'train':
# training data
  with open(output_dir+"dict_sent_tuple_count_traindata_shuffle.pkl", "wb") as f:
    pickle.dump(sentDict, f)

"""## Compute gradient"""

tuple_set_ord = tuple_set_1.copy()
tuple_set_ord.sort(key=lambda i:i[1],reverse=True)

conx = [x[1] for x in tuple_set_ord]

d_conx = np.diff(conx)

ind_conx = np.argwhere(d_conx != 0).reshape(-1)
new_ind_conx = np.concatenate([[0], ind_conx + 1, [len(tuple_set_ord)]])

full_batch = [tuple_set_ord[new_ind_conx[x]:new_ind_conx[x+1]] for x in range(len(new_ind_conx) -1)]

full_batch_flat = [x for y in full_batch for x in y]

weat_tuple_counts = Counter([inv_vocab[x[0]] for x in full_batch_flat])

array_reduced_full = np.zeros((len(full_batch_flat), HIDDEN_SIZE))

i=0
progress = 0
for batch in full_batch:
  inputs = [x[0] for x in batch]
  labels = [x[1] for x in batch]

  if LOSS == 'NG':
    loss_ng = _negative_sampling_loss_torch(inputs, labels, len(inputs),
                                            unigram_counts, NEGATIVES, weights, len(vocab)) 

  if LOSS == 'FULL':
    loss_ng = _full_loss_torch(inputs, labels, len(inputs),
                                            unigram_counts, None, weights, len(vocab)) 

  grad = torch.autograd.grad(loss_ng.sum(dim=1).reshape(1, len(inputs))[0],
                           weights[0], grad_outputs=torch.ones_like(loss_ng.sum(dim=1).reshape(1, len(inputs)))[0],
                           create_graph=True, retain_graph=True)[0]

  keys_words = Counter([x[0] for x in torch.nonzero(grad).numpy()]).keys()
  for idx in keys_words:
    array_reduced_full[progress] = grad[idx].detach().numpy()
    progress+=1

  i+=1
  print(i, datetime.datetime.now())

if MODE == 'post' and LOSS == 'NG':
# post training data
  with open(output_dir+"array_gradients_all_tuples_shuffle.pkl", "wb") as f:
    pickle.dump(array_reduced_full, f)

if MODE == 'train' and LOSS == 'NG':
# training data
  with open(output_dir+"array_gradients_all_tuples_traindata_shuffle.pkl", "wb") as f:
    pickle.dump(array_reduced_full, f)

if MODE == 'post' and LOSS == 'FULL':
# post training data
  with open(output_dir+"array_gradients_all_tuples_full_shuffle.pkl", "wb") as f:
    pickle.dump(array_reduced_full, f)

if MODE == 'train' and LOSS == 'FULL':
# training data
  with open(output_dir+"array_gradients_all_tuples_traindata_full_shuffle.pkl", "wb") as f:
    pickle.dump(array_reduced_full, f)

"""## Save unique tuples and corresponding gradients index"""

dict_tuple_index = {key: i for i, key in enumerate(full_batch_flat)}

if MODE == 'post':
# post training data
  with open(output_dir+"dict_all_tuples_shuffle.pkl", "wb") as f:
    pickle.dump(dict_tuple_index, f)

if MODE == 'train':
# training data
  with open(output_dir+"dict_all_tuples_traindata_shuffle.pkl", "wb") as f:
    pickle.dump(dict_tuple_index, f)
