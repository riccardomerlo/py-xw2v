"""# Import libraries and methods"""

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

import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from collections import Counter
from current_files_gensim.word_vectors import WordVectors
from pytorch.model_torch import Word2VecModel
from pytorch.dataset_torch import create_skipgram, read_corpus

def read_corpus(path, min_len = 3):
    text = []
    #cut_min = lambda x: x if len(x) > min_len
    try:
        with open(path, 'r') as f:
            #text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]
            for x in f.readlines():
                res = nltk.tokenize.word_tokenize(x.strip())
                if len(res) > min_len:
                    text.append(res)
    except Exception as e:
        print(e)
    return text

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                    else torch.FloatTensor)

import datetime;


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

def get_emb_og(wv, word):
    """
    Function allowing to get the embedding of a given word.
    word: word of interest
    return: word embedding
    """

    # same as wv[word]
    return wv._syn0_final[wv._rev_vocab[word]]


def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


def mean_cos_similarity(tar, att):
    '''
    Calculates the mean of the cosine similarity between the target and the range of attributes
    '''
    mean_cos = np.mean([cos_similarity(tar, attribute) for attribute in att])
    return mean_cos


def association(tar, att1, att2):
    '''
    Calculates the mean association between a single target and all of the attributes
    '''
    association = mean_cos_similarity(
        tar, att1) - mean_cos_similarity(tar, att2)

    return association


def effect_sizev2(t1, t2, att1, att2):
    '''
    Calculates the effect size (d) between the two target variables and the attributes
    Parameters:
        t1 (np.array): first target variable matrix
        t2 (np.array): second target variable matrix
        att1 (np.array): first attribute variable matrix
        att2 (np.array): second attribute variable matrix
    Returns:
        effect_size (float): The effect size, d.
    Example:
        t1 (np.array): Matrix of word embeddings for professions "Programmer, Scientist, Engineer"
        t2 (np.array): Matrix of word embeddings for professions "Nurse, Librarian, Teacher"
        att1 (np.array): matrix of word embeddings for males (man, husband, male, etc)
        att2 (np.array): matrix of word embeddings for females (woman, wife, female, etc)
    '''
    combined = np.concatenate([t1, t2])
    num1 = np.mean([association(target, att1, att2) for target in t1])
    num2 = np.mean([association(target, att1, att2) for target in t2])
    combined_association = np.array(
        [association(target, att1, att2) for target in combined])

    dof = combined_association.shape[0]
    denom = np.sqrt(
        ((dof-1)*np.std(combined_association, ddof=1) ** 2) / (dof-1))
    effect_size = (num1 - num2) / denom

    return effect_size

"""# Load data"""

BATCH_SIZE = 256
NEGATIVES = 5
EPOCHS = 3
SAMPLING_RATE = 1E-3
MIN_FREQ = 60
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

text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

with open("./content/data_correct_post_train/data_v2.pkl", "rb") as f:
  data = pickle.load(f)

with open("unigram_counts.pkl", "rb") as f:
  unigram_counts = pickle.load(f)

with open("vocab.pkl", "rb") as f:
  vocab = pickle.load(f)

with open("inv_vocab.pkl", "rb") as f:
  inv_vocab = pickle.load(f)

S_A = S.copy()+A.copy()
random.seed(0)
random.shuffle(S_A)
print(S_A)

def get_tuples(sentence, vocab):
  _data = []

  window = 5
  for i, t in enumerate(sentence):
      
      contexts = list(range(i-window, i + window+1))
      contexts = [c for c in contexts if c >=
                  0 and c != i and c < len(sentence)]
      for c in contexts:
          # TARGET, CONTEXT, nsent
          _data.append((vocab[t], vocab[sentence[c]], 1))

  return _data

sent_tuples = get_tuples(S_A, vocab)

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])


list_approx_words = list(set(list_index_weat.copy()))

flat_data = [x for y in data for x in y] # 6 secondi
for tu in sent_tuples:
    flat_data.append(np.array([tu[0], tu[1], -1]))
tuple_set = {(x[0], x[1]) for x in iter(flat_data)} # 32 secondi
tuple_set_1 = [x for x in tuple_set if x[0] in list_approx_words] #1m 7s secondi

tuple_set_ord = tuple_set_1.copy()
tuple_set_ord.sort(key=lambda i:i[1],reverse=True)


list_sim_sent_retrain = []
sent_id =  -1
sent_text = text[sent_id]

log_per_steps= 1000#10000  # Every `log_per_steps` steps to log the value of loss to be minimized.

#flat_data = [x for xs in data for x in xs]
#flat_data_rt = [x for x in flat_data if x[2]!=sent_id]
data_rt = split_given_size(flat_data, BATCH_SIZE)
data_rt = [x for x in data_rt if len(x) == BATCH_SIZE]
unigram_counts_rt = unigram_counts.copy()
vocab_rt = vocab.copy()
for key in vocab:
    if key in sent_text:
        unigram_counts_rt[vocab[key]] = unigram_counts[vocab[key]]+1
inv_vocab_rt = {v: k for k, v in vocab_rt.items()}

        #data_rt, unigram_counts_rt, vocab_rt, inv_vocab_rt = create_skipgram(text_rt, WINDOW_SIZE, WEATLIST, MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

word2vec = Word2VecModel(hidden_size=300,
                         batch_size=BATCH_SIZE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE)

word2vec._data = data_rt
word2vec._vocab = vocab_rt
word2vec._unigram_counts = unigram_counts_rt
word2vec._vocab_size = len(unigram_counts_rt)
word2vec._inv_vocab = inv_vocab_rt
word2vec._text = text

word2vec.build_weights()
word2vec.train(EPOCHS, save=False)

print("Re-training for sentence "+str(sent_id)+" completed")

w0 = word2vec.syn0.cpu().detach().numpy()
w1 = word2vec.syn1.cpu().detach().numpy()

vocab_words = [word for word in vocab_rt]

wv = WordVectors(w0, vocab_words)

# compute original effect size and similarities:
t1 = np.array([get_emb_og(wv, word) for word in S])
t2 = np.array([get_emb_og(wv, word) for word in T])
att1 = np.array([get_emb_og(wv, word) for word in A])
att2 = np.array([get_emb_og(wv, word) for word in B])

simSA = np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean()
simTA = np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean()
simSB = np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean()
simTB = np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean()

print("effect size retraining: ", effect_sizev2(t1, t2, att1, att2))
print("similarity S-A origin: ", simSA)
print("similarity T-A origin: ", simTA)
print("similarity S-B origin: ", simSB)
print("similarity T-B origin: ", simTB)
