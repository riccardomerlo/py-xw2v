# -*- coding: utf-8 -*-
"""post_training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lDIyUqy-JjCyX1OdoZGMBrTyd5_IQ1F6

# Connect to drive
"""


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
from word_vectors import WordVectors

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

syn0 = np.load('syn0_final_torch.npy')
syn1 = np.load('syn1_final_torch.npy')

#with open("data.pkl", "rb") as f:
#  data = pickle.load(f)

with open("./content/data_correct_post_train/vocab_v2.pkl", "rb") as v:
  vocab = pickle.load(v)

with open("./content/data_correct_post_train/unigram_counts_v2.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open("./content/data_correct_post_train/inv_vocab_v2.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

with open("dict_all_tuples_v2.pkl", "rb") as iv:
  dict_all_tuples = pickle.load(iv)

with open("dict_sent_tuple_count_v2.pkl", "rb") as iv:
  dict_sent_tuple_count = pickle.load(iv)

with open("dict_tuple_sent_count_v2.pkl", "rb") as iv:
  dict_tuple_sent_count = pickle.load(iv)

with open("array_gradients_all_tuples_v2.pkl", "rb") as iv:
  array_gradients = pickle.load(iv)

"""# Try with one sent"""

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])


list_approx_words = list(set(list_index_weat.copy()))


weights = [torch.from_numpy(syn0).requires_grad_(), torch.from_numpy(syn1).requires_grad_()]


"""
Read CORPUS with only sentences containing WEAT words
"""
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

syn0_final = np.load('syn0_final_torch.npy')

vocab_words = [key for key in vocab]
wv = WordVectors(syn0_final, vocab_words)

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
    association = mean_cos_similarity(tar, att1) - mean_cos_similarity(tar, att2)

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
    combined_association = np.array([association(target, att1, att2) for target in combined])

    dof = combined_association.shape[0]
    denom = np.sqrt(((dof-1)*np.std(combined_association, ddof=1) ** 2 ) / (dof-1))
    effect_size = (num1 - num2) / denom

    return effect_size


def get_emb_og(wv, word):
    """
    Function allowing to get the embedding of a given word.

    word: word of interest
    return: word embedding
    """

    # same as wv[word]
    return wv._syn0_final[wv._rev_vocab[word]]

# our training
t1 = np.array([get_emb_og(wv, word) for word in S])
t2 = np.array([get_emb_og(wv, word) for word in T])
att1 = np.array([get_emb_og(wv, word) for word in A])
att2 = np.array([get_emb_og(wv, word) for word in B])
ef_full = effect_sizev2(t1, t2, att1, att2)
print("effect size full corpus: ", ef_full)


def get_sent_grad(diz_sent_tuple, diz_mapping, arr_grad, sent_id):
    """
    Computes the gradient for the sentence to be removed by aggregating the values in diz_gradients.
    diz_gradients: dictionary of type {(target, context_word, nsent): gradient}
    sent_id: id of sentence in the corpus
    return: dictionary of sentence gradient by target word
    """
    diz_grad_sent = {}
    sent = diz_sent_tuple[sent_id]
    for tu in sent:
      couple = tu[0]
      count = tu[1]
      idx = diz_mapping[couple]
      target = inv_vocab[couple[0]]
      # sum over the gradients of instances which are in the sentence I'm removing
      if target not in diz_grad_sent:
        diz_grad_sent[target] = arr_grad[idx]*count
      else:
        diz_grad_sent[target] += arr_grad[idx]*count

    return diz_grad_sent


def get_hessian(word):
  """
  Creates dictionary of hessian matrices.

  list_index_weat: list
  """
  idx_w = vocab[word]
  with open(str(idx_w)+'_hessian.pkl', "rb") as f:
      hessian_word = pickle.load(f)
  return hessian_word

def get_emb_pert(perturbed_emb, word):
    """
    Apply dictionary to given word to get the embedding.

    word: word of interest
    return: word embedding
    """
    return perturbed_emb[word]


def get_perturbed_emb_sent(wv, WEATLIST, diz_sent_tuple, diz_tuple_sent, diz_mapping, arr_grad, sent_id, sent_text):
    diz_grad_sent = get_sent_grad(diz_sent_tuple, diz_mapping, arr_grad, sent_id)
    perturbed_emb = {}  # dictionary {word: emb}
    #hessian_diz = get_hessian_words(list_index_weat, diz_tuple_sent, diz_mapping, arr_grad)
    for word in WEATLIST:
        emb = get_emb_og(wv, word)
        if word in sent_text:  # single entity
            V = len(wv._vocab)
            if word in diz_grad_sent:
                # gradient for the word of interest
                grad_sent = diz_grad_sent[word]
                #print(grad_sent)
                hessian = np.linalg.inv(get_hessian(word))
            else:  # in case the term is appearing only in the sentence to be removed
                grad_sent = np.zeros(HIDDEN_SIZE)
                hessian = np.zeros((HIDDEN_SIZE, HIDDEN_SIZE))
            # decide whether grad_sent should be multiplied by V or not - technically yes to fully follow Brunet et al.
            emb = emb + (1/V)*np.dot(hessian, grad_sent)
            #emb = emb + (1/V)*grad_sent
        perturbed_emb[word] = emb

    return perturbed_emb


# list_sent = []
# list_sentid = []
# list_diffbias = []
# list_simSA = []
# list_simTA = []
# list_simSB = []
# list_simTB = []


# i=0
# for sent_id in range(len(text)):
#     sent_text = text[sent_id]
#     first=True
#     for weat_w in WEATLIST:
#         if weat_w in sent_text and first==True:
#             perturbed_emb = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count, dict_tuple_sent_count, dict_all_tuples, array_gradients, sent_id, sent_text)

#             t1 = np.array([get_emb_pert(perturbed_emb, word) for word in S])
#             t2 = np.array([get_emb_pert(perturbed_emb, word) for word in T])
#             att1 = np.array([get_emb_pert(perturbed_emb, word) for word in A])
#             att2 = np.array([get_emb_pert(perturbed_emb, word) for word in B])

#             eff_size_pert = effect_sizev2(t1, t2, att1, att2)

#             list_simSA.append(np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean())
#             list_simTA.append(np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean())
#             list_simSB.append(np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean())
#             list_simTB.append(np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean())
#             # effect size perturbed
#             #ef_perturbed = effect_size(S, T, A, B, get_emb_pert, perturbed_emb)

#             # differential bias
#             diff_bias = ef_full-eff_size_pert

#             # print("Sentence: ", sent_id, sent_text)
#             # print("Effect size full corpus: ", ef_full)
#             # print("Effect size perturbed corpus: ", eff_size_pert)
#             # print("Differential bias: ", diff_bias)

#             list_sent.append(' '.join(sent_text))
#             list_sentid.append(sent_id)
#             list_diffbias.append(diff_bias)          
#             i+=1

#             if i%1000 == 0:
#                 print(i)
#             first=False

# df = pd.DataFrame(columns=['sent_id', 'sent_text', 'diff_bias'])
# df['sent_id'] = list_sentid
# df['sent_text'] = list_sent
# df['diff_bias'] = list_diffbias
# df['sim S-A'] = list_simSA
# df['sim T-A'] = list_simTA
# df['sim S-B'] = list_simSB
# df['sim T-B'] = list_simTB

# df.to_csv("differ_bias.csv", index=False)


sent_id = 534994
sent_text = text[sent_id]

perturbed_emb = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count, dict_tuple_sent_count, dict_all_tuples, array_gradients, sent_id, sent_text)

t1 = np.array([get_emb_pert(perturbed_emb, word) for word in S])
t2 = np.array([get_emb_pert(perturbed_emb, word) for word in T])
att1 = np.array([get_emb_pert(perturbed_emb, word) for word in A])
att2 = np.array([get_emb_pert(perturbed_emb, word) for word in B])

eff_size_pert = effect_sizev2(t1, t2, att1, att2)

syn0_retrain = np.load('syn0_final_torch_retrain.npy')

wv_r = WordVectors(syn0_retrain, vocab_words)

# cosine similarity his, him, science
print('his :', cos_similarity(get_emb_pert(perturbed_emb, "his"), get_emb_og(wv_r, "his")))
print('him :', cos_similarity(get_emb_pert(perturbed_emb, "him"), get_emb_og(wv_r, "him")))
print('science :', cos_similarity(get_emb_pert(perturbed_emb, "science"), get_emb_og(wv_r, "science")))


# effect size perturbed
#ef_perturbed = effect_size(S, T, A, B, get_emb_pert, perturbed_emb)

# differential bias
diff_bias = ef_full-eff_size_pert

print("Sentence: ", sent_id, sent_text)
print("Effect size full corpus: ", ef_full)
print("Effect size perturbed corpus: ", eff_size_pert)
print("Differential bias: ", diff_bias)
    
