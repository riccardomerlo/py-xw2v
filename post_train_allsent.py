# -*- coding: utf-8 -*-

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
NEGATIVES = 20
EPOCHS = 5
SAMPLING_RATE = 1E-3
MIN_FREQ = 100
WINDOW_SIZE = 5
LEARNING_RATE = 1E-2
HIDDEN_SIZE = 300
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

with open(output_dir+"data_shuffle.pkl", "rb") as f:
    data = pickle.load(f)

with open(output_dir+"vocab_v2.pkl", "rb") as v:
  vocab = pickle.load(v)

with open(output_dir+"unigram_counts_v2.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open(output_dir+"inv_vocab_v2.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

# load post train data
with open(output_dir+"dict_all_tuples_shuffle.pkl", "rb") as iv:
  dict_all_tuples_posttrain = pickle.load(iv)

with open(output_dir+"dict_sent_tuple_count_shuffle.pkl", "rb") as iv:
  dict_sent_tuple_count_posttrain = pickle.load(iv)

with open(output_dir+"dict_tuple_sent_count_shuffle.pkl", "rb") as iv:
  dict_tuple_sent_count_posttrain = pickle.load(iv)

with open(output_dir+"array_gradients_all_tuples_shuffle.pkl", "rb") as iv:
  array_gradients_posttrain = pickle.load(iv)

# load train data
with open(output_dir+"dict_all_tuples_traindata_shuffle.pkl", "rb") as iv:
  dict_all_tuples_train = pickle.load(iv)

with open(output_dir+"dict_sent_tuple_count_traindata_shuffle.pkl", "rb") as iv:
  dict_sent_tuple_count_train = pickle.load(iv)

with open(output_dir+"dict_tuple_sent_count_traindata_shuffle.pkl", "rb") as iv:
  dict_tuple_sent_count_train = pickle.load(iv)

with open(output_dir+"array_gradients_all_tuples_traindata_shuffle.pkl", "rb") as iv:
  array_gradients_train = pickle.load(iv)

"""# Try with one sent"""

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])

list_approx_words = list(set(list_index_weat.copy()))

weights = [torch.from_numpy(syn0).requires_grad_(), torch.from_numpy(syn1).requires_grad_()]


"""
Read CORPUS with only sentences containing WEAT words
"""
#text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

with open(output_dir+"text_shuffle.pkl", "rb") as iv:
  text = pickle.load(iv)

syn0_final = np.load(output_dir+'syn0_final_torch.npy')

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
    sent = [diz_sent_tuple[si] for si in sent_id]
    sent = [y for x in sent for y in x]
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


def get_hessian_posttrain(word):
  """
  Creates dictionary of hessian matrices.

  list_index_weat: list
  """
  idx_w = vocab[word]
  with open(output_dir+str(idx_w)+'_hessian_shuffle.pkl', "rb") as f:
      hessian_word = pickle.load(f)
  return hessian_word


def get_hessian_train(word):
  """
  Creates dictionary of hessian matrices.

  list_index_weat: list
  """
  idx_w = vocab[word]
  with open(output_dir+str(idx_w)+'_hessian_traindata_shuffle.pkl', "rb") as f:
      hessian_word = pickle.load(f)
  return hessian_word

def get_emb_pert(perturbed_emb, word):
    """
    Apply dictionary to given word to get the embedding.

    word: word of interest
    return: word embedding
    """
    return perturbed_emb[word]


def get_perturbed_emb_sent(wv, WEATLIST, diz_sent_tuple, diz_tuple_sent, diz_mapping, arr_grad, sent_id, sent_text, post_train=True):
    diz_grad_sent = get_sent_grad(diz_sent_tuple, diz_mapping, arr_grad, sent_id)
    set_words = {w for w in [y for x in [text[si] for si in sent_id] for y in x]}
    #print(diz_grad_sent.keys())
    perturbed_emb = {}  # dictionary {word: emb}
    #hessian_diz = get_hessian_words(list_index_weat, diz_tuple_sent, diz_mapping, arr_grad)
    for word in WEATLIST:
        emb = get_emb_og(wv, word)
        if word in set_words:  # single entity
            V = len(wv._vocab)
            if word in diz_grad_sent:
                # gradient for the word of interest
                grad_sent = diz_grad_sent[word]
                #print(grad_sent)
                if post_train:
                    hessian = np.linalg.inv(get_hessian_posttrain(word))
                else:
                    hessian = np.linalg.inv(get_hessian_train(word))
            else:  # in case the term is appearing only in the sentence to be removed
                grad_sent = np.zeros(HIDDEN_SIZE)
                hessian = np.zeros((HIDDEN_SIZE, HIDDEN_SIZE))
            # decide whether grad_sent should be multiplied by V or not - technically yes to fully follow Brunet et al.
            emb = emb + (1/V)*np.dot(hessian, grad_sent)
            #emb = emb + (1/V)*grad_sent
        perturbed_emb[word] = emb

    return perturbed_emb


# list_sentid = []
# #list_diffbias = []
# list_efpert_posttrain = []
# list_efpert_train = []
# list_tuple_posttrain = []
# list_tuple_train = []
# #list_simSA = []
# #list_simTA = []
# #list_simSB = []
# #list_simTB = []


# flat_data = [y for x in data for y in x]
# set_sent_ids = set([x[2] for x in flat_data])

# i=0
# for sent_id in set_sent_ids:
#   sent_text = text[sent_id]
#   first=True
#   for weat_w in WEATLIST:
#     if weat_w in sent_text and first==True:
#       # post train data
#       post_train_tuples = [(inv_vocab[x[0][0]], inv_vocab[x[0][1]]) for x in dict_sent_tuple_count_posttrain[sent_id]]
#       perturbed_emb_posttrain = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count_posttrain, dict_tuple_sent_count_posttrain, dict_all_tuples_posttrain, array_gradients_posttrain, [sent_id], text, post_train=True)

#       t1 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in S])
#       t2 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in T])
#       att1 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in A])
#       att2 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in B])

#       # for i in range(len(t1)):
#       #   dict_post[S[i]] = t1[i]
#       # for i in range(len(t2)):
#       #   dict_post[T[i]] = t2[i]
#       # for i in range(len(att1)):
#       #   dict_post[A[i]] = att1[i]
#       # for i in range(len(att2)):
#       #   dict_post[B[i]] = att2[i]

#       eff_size_pert_posttrain = effect_sizev2(t1, t2, att1, att2)

#       # train data
#       train_tuples = [(inv_vocab[x[0][0]], inv_vocab[x[0][1]]) for x in dict_sent_tuple_count_train[sent_id]]
#       perturbed_emb_train = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count_train, dict_tuple_sent_count_train, dict_all_tuples_train, array_gradients_train, [sent_id], text, post_train=False)

#       t1 = np.array([get_emb_pert(perturbed_emb_train, word) for word in S])
#       t2 = np.array([get_emb_pert(perturbed_emb_train, word) for word in T])
#       att1 = np.array([get_emb_pert(perturbed_emb_train, word) for word in A])
#       att2 = np.array([get_emb_pert(perturbed_emb_train, word) for word in B])

#       # for i in range(len(t1)):
#       #   dict_train[S[i]] = t1[i]
#       # for i in range(len(t2)):
#       #   dict_train[T[i]] = t2[i]
#       # for i in range(len(att1)):
#       #   dict_train[A[i]] = att1[i]
#       # for i in range(len(att2)):
#       #   dict_train[B[i]] = att2[i]

#       eff_size_pert_train = effect_sizev2(t1, t2, att1, att2)

#       # list_simSA.append(np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean())
#       # list_simTA.append(np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean())
#       # list_simSB.append(np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean())
#       # list_simTB.append(np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean())
#       # effect size perturbed
#       #ef_perturbed = effect_size(S, T, A, B, get_emb_pert, perturbed_emb)

#       # differential bias post training
#       diff_bias_posttrain = ef_full-eff_size_pert_posttrain

#       # print("Sentence: ", sent_id, sent_text)
#       # print("Effect size full corpus: ", ef_full)
#       # print("Effect size perturbed corpus: ", eff_size_pert)
#       # print("Differential bias: ", diff_bias)

#     #  list_sent.append(' '.join(sent_text))
#       list_sentid.append(sent_id)
#     # list_diffbias.append(diff_bias)    
#       list_efpert_posttrain.append(eff_size_pert_posttrain)
#       list_efpert_train.append(eff_size_pert_train) 
#       list_tuple_posttrain.append(post_train_tuples)  
#       list_tuple_train.append(train_tuples)      
#       i+=1

#       if i%1000 == 0:
#           print(i)
#       first=False

# # with open("dict_post_emb", "wb") as f:
# #   pickle.dump(dict_post, f)

# # with open("dict_train_emb", "wb") as f:
# #   pickle.dump(dict_train, f)

# df = pd.DataFrame(columns=['sent_id', 'tuples_post', 'tuples_train', 'effect_size_pert_posttrain', 'effect_size_pert_train'])
# df['sent_id'] = list_sentid
# #df['diff_bias'] = list_diffbias
# df['tuples_post'] = list_tuple_posttrain
# df['tuples_train'] = list_tuple_train
# df['effect_size_pert_posttrain'] = list_efpert_posttrain
# df['effect_size_pert_train'] = list_efpert_train
# # # df['sim S-A'] = list_simSA
# # # df['sim T-A'] = list_simTA
# # # df['sim S-B'] = list_simSB
# # # df['sim T-B'] = list_simTB

# df.to_csv("effect_size_batch256.csv", index=False)



i=0
sent_id = 321712
sent_text = text[sent_id]
first=True
dict_train = {}
dict_post = {}
for weat_w in WEATLIST:
  if weat_w in sent_text and first==True:
#       # post train data
    post_train_tuples = [(inv_vocab[x[0][0]], inv_vocab[x[0][1]]) for x in dict_sent_tuple_count_posttrain[sent_id]]
    perturbed_emb_posttrain = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count_posttrain, dict_tuple_sent_count_posttrain, dict_all_tuples_posttrain, array_gradients_posttrain, [sent_id], text, post_train=True)

    t1 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in S])
    t2 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in T])
    att1 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in A])
    att2 = np.array([get_emb_pert(perturbed_emb_posttrain, word) for word in B])

    for i in range(len(t1)):
      dict_post[S[i]] = t1[i]
    for i in range(len(t2)):
      dict_post[T[i]] = t2[i]
    for i in range(len(att1)):
      dict_post[A[i]] = att1[i]
    for i in range(len(att2)):
      dict_post[B[i]] = att2[i]

    eff_size_pert_posttrain = effect_sizev2(t1, t2, att1, att2)

    # train data
    train_tuples = [(inv_vocab[x[0][0]], inv_vocab[x[0][1]]) for x in dict_sent_tuple_count_train[sent_id]]
    perturbed_emb_train = get_perturbed_emb_sent(wv, WEATLIST, dict_sent_tuple_count_train, dict_tuple_sent_count_train, dict_all_tuples_train, array_gradients_train, [sent_id], text, post_train=False)

    t1 = np.array([get_emb_pert(perturbed_emb_train, word) for word in S])
    t2 = np.array([get_emb_pert(perturbed_emb_train, word) for word in T])
    att1 = np.array([get_emb_pert(perturbed_emb_train, word) for word in A])
    att2 = np.array([get_emb_pert(perturbed_emb_train, word) for word in B])

    for i in range(len(t1)):
      dict_train[S[i]] = t1[i]
    for i in range(len(t2)):
      dict_train[T[i]] = t2[i]
    for i in range(len(att1)):
      dict_train[A[i]] = att1[i]
    for i in range(len(att2)):
      dict_train[B[i]] = att2[i]

    eff_size_pert_train = effect_sizev2(t1, t2, att1, att2)

    # list_simSA.append(np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean())
    # list_simTA.append(np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean())
    # list_simSB.append(np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean())
    # list_simTB.append(np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean())
    # effect size perturbed
    #ef_perturbed = effect_size(S, T, A, B, get_emb_pert, perturbed_emb)

    # differential bias post training
    diff_bias_posttrain = ef_full-eff_size_pert_posttrain

    print("Sentence: ", sent_id, sent_text)
    print("Effect size full corpus: ", ef_full)
    print("Effect size approx train: ", eff_size_pert_train)
    print("Effect size approx post: ", eff_size_pert_posttrain)

  # #  list_sent.append(' '.join(sent_text))
  #   list_sentid.append(sent_id)
  # # list_diffbias.append(diff_bias)    
  #   list_efpert_posttrain.append(eff_size_pert_posttrain)
  #   list_efpert_train.append(eff_size_pert_train) 
  #   list_tuple_posttrain.append(post_train_tuples)  
  #   list_tuple_train.append(train_tuples)      
    i+=1

    if i%1000 == 0:
        print(i)
    first=False

with open("dict_post_emb", "wb") as f:
  pickle.dump(dict_post, f)

with open("dict_train_emb", "wb") as f:
  pickle.dump(dict_train, f)
