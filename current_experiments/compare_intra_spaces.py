import numpy as np
import pandas as pd
import pickle
import itertools
import sys

from dataset_torch import read_corpus
from word_vectors import WordVectors

output_dir='/home/rmerlo/py-xw2v/gensim_t2/'

sent_id = 412444

# get magnitude of embeddings for low freq and high freq words
get_magnitude = False

# define which method to use
brunetlike = True
sgdlike = False
sgdlike_invloss = False
hybrid = False

# define which loss to consider for sgdlike_invloss and hybrid
ng = True # use negative sampling inverse loss, if False use true context inverse loss

with open(output_dir+"vocab_complete.pkl", "rb") as v:
  vocab = pickle.load(v)

print('loaded', 'vocab')

with open(output_dir+"inv_vocab_complete.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

print('loaded', 'inv_vocab')

"""
Read CORPUS with only sentences containing WEAT words
"""
with open(output_dir+"tokenized_text.pkl", "rb") as v:
  text = pickle.load(v)

def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


syn0_og = np.load(output_dir+'syn0_final_torch.npy')

data = []

low_freq = ['scannon', 'doctorate', 'residency']
high_freq = ['he', 'his', 'just']

# define lr values list depending on the method
if brunetlike or hybrid:
  lr_list = [0.1, 1, 10, 100, 1000, 10000]
elif sgdlike or sgdlike_invloss:
  lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# define read/write names depending on the method
if brunetlike:
  name_load = str(sent_id)+'_emb_brunetlike'
  name_write = str(sent_id)+'_brunetlike'
elif hybrid:
  if ng:
    name_load = str(sent_id)+'_emb_hybrid_ng'
    name_write = str(sent_id)+'_hybrid_ng'
  else:
    name_load = str(sent_id)+'_emb_hybrid_true'
    name_write = str(sent_id)+'_hybrid_true'
if sgdlike:
  name_load = str(sent_id)+'_emb_sgdlike'
  name_write = str(sent_id)+'_sgdlike'
elif sgdlike_invloss:
  if ng:
    name_load = str(sent_id)+'_emb_sgdlike_inv_ng'
    name_write = str(sent_id)+'_sgdlike_inv_ng'
  else:
    name_load = str(sent_id)+'_emb_sgdlike_inv_true'
    name_write = str(sent_id)+'_sgdlike_inv_true'


# run for the sentence
for _id in [sent_id]:

    sentence_words = [x for x in text[_id] if x in vocab]

    syn0_retrain = np.load(output_dir+'syn0_final_torch_retrain_'+str(_id)+'_control_ord.npy')
    with open(output_dir+name_load+'.pkl', "rb") as v:
        syn0_approx = pickle.load(v)

    wv0_og = WordVectors(syn0_og, list(vocab.keys()))
    wv0_retrain = WordVectors(syn0_retrain, list(vocab.keys()))

    if get_magnitude:
      for word in low_freq+high_freq:
        for lr in lr_list:
          print(word, lr, 'og:', wv0_og.get_embedding(word), 'retrain:', wv0_retrain.get_embedding(word), 'approx:', syn0_approx[str(lr)][vocab[word]])

    for pair in itertools.combinations(sentence_words, 2):
        
        sim_og = cos_similarity(wv0_og.get_embedding(pair[0]), wv0_og.get_embedding(pair[1]))
        sim_retrain = cos_similarity(wv0_retrain.get_embedding(pair[0]), wv0_retrain.get_embedding(pair[1]))

        for lr in lr_list:
          sim_approx = cos_similarity(syn0_approx[str(lr)][vocab[pair[0]]], syn0_approx[str(lr)][vocab[pair[1]]])
          data.append([_id, pair[0],  pair[1], sim_og, sim_retrain, lr, sim_approx])

df = pd.DataFrame(data, columns=['sentence_id', 'w1', 'w2', 'og', 'retrain', 'lr', 'approx'])
df.to_csv('compare_influence_'+name_write+'.csv', index=False)
