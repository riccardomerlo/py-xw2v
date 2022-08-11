"""
MAIN
"""

from get_hessians_utils import get_full_hessian, get_full_hessian_negatives
from pytorch.dataset_torch import read_corpus
import pickle
import torch
import numpy as np
import os
import sys
from datetime import datetime

MODE = 'post' # 'post', 'train'
LOSS = 'NG' # 'NG', 'FULL'

W = 0 # W0 = 0; W1 = 1

NEGATIVES = 20

sent_id = 412444

output_dir = '/home/rmerlo/py-xw2v/gensim_t2/'


with open(output_dir+'data_complete.pkl', 'rb') as han: 
  data = pickle.load(han)

with open(output_dir + 'vocab_complete.pkl', 'rb') as han:
  vocab = pickle.load(han)

with open(output_dir + 'inv_vocab_complete.pkl', 'rb') as han:
  inv_vocab = pickle.load(han)

with open(output_dir + 'unigram_counts_complete.pkl', 'rb') as han:
  unigram_counts = pickle.load(han)

weights = [torch.from_numpy(np.load(output_dir+'syn0_final_torch.npy')).requires_grad_().cuda(),
           torch.from_numpy(np.load(output_dir+'syn1_final_torch.npy')).requires_grad_().cuda() ]


flat_data = [x for y in data for x in y] # 6 secondi
tuple_set = [(x[0], x[1]) for x in iter(flat_data)] # 32 secondi

# prendo solo tuple che hanno parola WEAT come target
# tuple_set_1 = [x for x in tuple_set if x[0] in list_approx_words] #1m 7s secondi


"""
Read CORPUS with only sentences containing WEAT words
"""
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

sent_text = text[sent_id]

# get 50 most freq words and sentence words
list_top50 = [x for x in list(vocab.keys())[:50]]
list_interest_words = {vocab[x] for x in sent_text+list_top50 if x in vocab} # se esiste nel vocabolario == non e' min freq

batch_size = 262144 
for _target in list_interest_words: # for each word
  # skip if already existing
  if os.path.exists(output_dir+str(_target)+"_W"+str(W)+"_hessian_shuffle_top50.pkl") or os.path.exists(output_dir+str(_target)+"_W"+str(W)+"_hessian_shuffle.pkl"):
    continue

  start = datetime.now()
  print('start', _target, start)
  
  hess = get_full_hessian_negatives(W, LOSS, _target, tuple_set, batch_size, weights, unigram_counts, NEGATIVES)
  # hess = get_full_hessian(W, LOSS, _target, tuple_set, batch_size, weights, unigram_counts_v2, NEGATIVES)
  end = datetime.now()
  print("done", _target, inv_vocab[_target], end)
  print("took", end-start)


  with open(output_dir+str(_target)+"_W"+str(W)+"_hessian.pkl", "wb") as han:
    pickle.dump(hess, han)
