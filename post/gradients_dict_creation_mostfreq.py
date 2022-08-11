"""
# Load libraries
"""

import nltk
nltk.download('punkt')
from scipy import spatial
#from word_vectors import WordVectors
import numpy as np
import pickle
import pandas as pd
import torch
import torch.utils.data
from gensim.models import Word2Vec

import datetime;
import matplotlib.pyplot as plt
# import seaborn as sns
import random
from collections import Counter


from pytorch.dataset_torch import read_corpus, split_given_size, get_dynamic_window
from get_hessians_utils import _negative_sampling_loss_torch


torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                     else torch.FloatTensor)
def hash_uni(astring):
   return ord(astring[0])


MODE = 'post' # 'post', 'train', no mode 'train' when gensim is used
LOSS = 'NG' # 'NG', 'FULL'

print("MODE", MODE, "---- LOSS", LOSS)

load_dataset = True

sent_id = 412444

"""# Load data"""


BATCH_SIZE = 2048
BATCH_N_SENTENCE = 10
NEGATIVES = 20
WINDOW_SIZE = 5
MIN_FREQ = 15

#LEARNING_RATE = 1E-3 
HIDDEN_SIZE = 300
_random_seed = 0
# liste_termini_weat
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
WEATLIST = S+T+A+B

output_dir =  '/home/rmerlo/py-xw2v/gensim_t2/'

syn0 = np.load(output_dir+'syn0_final_torch.npy')
syn1 = np.load(output_dir+'syn1_final_torch.npy')

with open(output_dir+"vocab_gensim.pkl", "rb") as v:
  vocab = pickle.load(v)

with open(output_dir+"unigram_counts_gensim.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open(output_dir+"inv_vocab_gensim.pkl", "rb") as iv:
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


def get_text(text):

    return iter(text)


def build_dataset_gensim(text, vocab, unigram_counts, inv_vocab, max_window, whitelist=[], min_freq=1, sampling_rate=1e-3, dynamic_window=True, batch_n_sentence=None, fixed_batch_size=False):
    """
    Builds post training dataset by creating tuples that only contain weat words.
    """
    # _, to_remove_words, to_keep_words = apply_reduction(
    #     text, vocab, unigram_counts, whitelist.copy(), min_freq)
    
    # print(to_remove_words)

    _text = [[s for s in sentence if s in vocab] for sentence in text] # nb: già vocab contiene solo parole con una certa min freq
    _vocab = vocab
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

    #cache_sentence = []
    _sent_batch = []
    tmp_batch_sentence = 0
    for n_sent, sentence in enumerate(get_text(_text)):
        
      #  if contains_weat(sentence, whitelist):
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
        
        #shuffle data weat
        random.seed(_random_seed) #ensures reproducibility
        random.shuffle(_data_weat)

        _data = _data_weat + _data # put weat tuples first
        #batch _inputs and _labels
        _data_batch = split_given_size(_data, BATCH_SIZE)
    
        if fixed_batch_size:
          _data_batch = [x for x in _data_batch if len(x) == BATCH_SIZE]

        _data_batch = list(reversed(_data_batch)) # change order of tuples to have weat as the last ones


    #repeat epoch times (done during training)

    _data = _data_batch

    return _data, _vocab, _inv_vocab, _unigram_counts


"""
Build DATASET
"""

if load_dataset:

  with open(output_dir+"data_complete.pkl", "rb") as han:
      data = pickle.load(han)
  with open(output_dir+"vocab_complete.pkl", "rb") as han:
      vocab = pickle.load(han)
  with open(output_dir+"inv_vocab_complete.pkl", "rb") as han:
      inv_vocab = pickle.load(han)
  with open(output_dir+"unigram_counts_complete.pkl", "rb") as han:
      unigram_counts = pickle.load(han)

  print('dataset loaded')

else:
  data, vocab, inv_vocab, unigram_counts = build_dataset_gensim(text, vocab, unigram_counts, inv_vocab, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, 0, dynamic_window=False)
  print('dataset built')

  with open(output_dir+"data_complete.pkl", "wb") as han:
      pickle.dump(data, han)
  with open(output_dir+"vocab_complete.pkl", "wb") as han:
      pickle.dump(vocab, han)
  with open(output_dir+"inv_vocab_complete.pkl", "wb") as han:
      pickle.dump(inv_vocab, han)
  with open(output_dir+"unigram_counts_complete.pkl", "wb") as han:
      pickle.dump(unigram_counts, han)



# consider both words of the given sentence
sent_words = text[sent_id]
#list_top50 = [x for x in list(vocab.keys())[:50]]
list_interest_words = [vocab[word] for word in sent_words if word in vocab]

flat_data = [x for y in data for x in y] 
tuple_set = {(x[0], x[1]) for x in iter(flat_data) if x[2]==sent_id} 
tuple_set_1 = [x for x in tuple_set if x[0] in list_interest_words] 

"""Compute gradients"""


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
  with open(output_dir+"array_gradients_sent_"+str(sent_id)+"_shuffle.pkl", "wb") as f:
    pickle.dump(array_reduced_full, f)

"""## Save unique tuples and corresponding gradients index"""

dict_tuple_index = {key: i for i, key in enumerate(full_batch_flat)}

if MODE == 'post':
# post training data
  with open(output_dir+"dict_sent_"+str(sent_id)+"_shuffle.pkl", "wb") as f:
    pickle.dump(dict_tuple_index, f)
