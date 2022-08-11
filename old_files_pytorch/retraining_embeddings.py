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

text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')


with open(output_dir+"data_shuffle.pkl", "rb") as f:
  data = pickle.load(f)

with open(output_dir+"unigram_counts_shuffle.pkl", "rb") as f:
  unigram_counts = pickle.load(f)

with open(output_dir+"vocab_shuffle.pkl", "rb") as f:
  vocab = pickle.load(f)

with open(output_dir+"inv_vocab_shuffle.pkl", "rb") as f:
  inv_vocab = pickle.load(f)


list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])

list_approx_words = list(set(list_index_weat.copy()))

list_sim_sent_retrain = []
sent_id = 151078
#sent_ids = set([x[2] for x in data[batch_id]])
sent_text = text[sent_id]
print(sent_text)

log_per_steps= 1000#10000  # Every `log_per_steps` steps to log the value of loss to be minimized.

flat_data = [x for xs in data for x in xs]

# print([(inv_vocab[x[0]], inv_vocab[x[1]]) for x in flat_data if x[2]==sent_id])
flat_data_rt = [x for x in flat_data if x[2]!=sent_id]

flat_data_rt = list(reversed(flat_data_rt))

data_rt = split_given_size(flat_data_rt, BATCH_SIZE)
data_rt = [x for x in data_rt if len(x) == BATCH_SIZE]

data_rt = list(reversed(data_rt)) # tuples with weat words are again the last ones

#data_rt = [data[i] for i in range(len(data)) if i!=batch_id] # remove whole batch related to sentence
unigram_counts_rt = unigram_counts.copy()
vocab_rt = vocab.copy()
for key in vocab:
    for sent in sent_text:
        if key in sent:
            unigram_counts_rt[vocab[key]] = unigram_counts[vocab[key]]-1
inv_vocab_rt = {v: k for k, v in vocab_rt.items()}

        #data_rt, unigram_counts_rt, vocab_rt, inv_vocab_rt = create_skipgram(text_rt, WINDOW_SIZE, WEATLIST, MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

word2vec = Word2VecModel(hidden_size=HIDDEN_SIZE,
                         batch_size=BATCH_SIZE,
                         batch_n_sentence=BATCH_N_SENTENCE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE,
                         output_dir=output_dir)

word2vec._data = data_rt
word2vec._vocab = vocab_rt
word2vec._unigram_counts = unigram_counts_rt
word2vec._vocab_size = len(unigram_counts_rt)
word2vec._inv_vocab = inv_vocab_rt
word2vec._text = text

word2vec.build_weights()
word2vec.train(EPOCHS, save=False, retrain=True)

print("Re-training for batch "+str(sent_id)+" completed")

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
