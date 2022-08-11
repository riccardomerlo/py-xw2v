# -*- coding: utf-8 -*-
"""watch_diffbias.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jd2qhdip9Xcb4LtqmqAqmW-XJmZCaBv_
"""


from scipy import spatial
import statistics

import numpy as np
import pickle
import pandas as pd
import torch
import torch.utils.data

from typing import List, \
    Union
import os
import sys
import gc

import random
from collections import defaultdict
from collections import Counter
from current_files_gensim.word_vectors import WordVectors


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


"""# get gradients"""


def fixed_unigram_candidate_sampler(
        true_classes,
        inputs,
        num_samples,
        unigrams,
        distortion=1.):

    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.detach().cpu().numpy()
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    # print(indices)
    result = np.zeros((len(inputs), num_samples))
    for idx in range(len(inputs)):
        value = inputs[idx]
        unigrams_new = unigrams.copy()
        # così non prende la true class come possibile negative per se stessa
        unigrams_new[true_classes[idx]] = 0
        torch.manual_seed(value)
        sampler = torch.utils.data.WeightedRandomSampler(
            unigrams, num_samples)
        candidates = np.array(list(sampler))
        result[idx] = candidates
    return result


def _negative_sampling_loss_torch(input, label, batch_size, unigram_counts, negatives, weights, vocab_len):
    """Builds the negative sampling loss.
    Args:
      input: int of shape [batch_size] => 1 (skip_gram)
      label: int of shape [batch_size] => 1
      batch_size: batch size chosen
      unigram_counts: list of sorted counts of words in vocab
      negatives: number of negatives to consider
      weights: list of weights, syn0 and syn1 matrices
    Returns:
      loss: float tensor of shape [batch_size, vocab_size].
    """
    syn0 = weights[0]
    syn1 = weights[1]

    true_classes_array = torch.unsqueeze(torch.tensor(label), 0)
    inputs_array = torch.unsqueeze(torch.tensor(input), 0)
    sampled_values = fixed_unigram_candidate_sampler(true_classes=true_classes_array,
                                                     inputs=inputs_array,
                                                     num_samples=negatives,
                                                     unigrams=unigram_counts,
                                                     distortion=0.75)

    inputs_syn0 = torch.index_select(
        syn0, 0, torch.from_numpy(np.array(input)).cuda())
    true_syn1 = torch.index_select(
        syn1, 0, torch.from_numpy(np.array(label)).cuda())

    # sampled_syn1 = syn1[sampled_values]
    list_sampled_syn1 = []
    for batch in sampled_values:
        sampled_syn1_batch = torch.index_select(syn1, 0, torch.tensor(
            torch.from_numpy(batch).clone().detach(), dtype=torch.int32).cuda())
        list_sampled_syn1.append(sampled_syn1_batch)
        del sampled_syn1_batch

    sampled_syn1 = torch.stack(list_sampled_syn1, 0)

    true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)

    del true_syn1

    # true_logits.requires_grad_()
    sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                  sampled_syn1.permute(0, 2, 1))

    del sampled_syn1
    del inputs_syn0
    # sampled_logits.requires_grad_()

    loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))

    del true_logits
    sampled_cross_entropy = loss(
        sampled_logits, torch.zeros_like(sampled_logits))

    del sampled_logits
    gc.collect()

    loss = torch.concat(
        [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)
    return loss


def get_grad(inp_label, unigram_counts, weights, vocab):

    loss_ng = _negative_sampling_loss_torch(inp_label[0], inp_label[1], 1,
                                            unigram_counts, 5, weights, len(vocab))

    grad = torch.autograd.grad(loss_ng.sum(),
                               weights[0],
                               create_graph=True,
                               retain_graph=True
                               )[0]

    return grad[inp_label[0]].detach().cpu().numpy()


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


def get_emb_og(wv, word):
    """
    Function allowing to get the embedding of a given word.
    word: word of interest
    return: word embedding
    """

    # same as wv[word]
    return wv._syn0_final[wv._rev_vocab[word]]


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
        #idx = diz_mapping[couple]
        target = inv_vocab[couple[0]]
        # sum over the gradients of instances which are in the sentence I'm removing
        if target not in diz_grad_sent:
            diz_grad_sent[target] = arr_grad[couple]*count
        else:
            diz_grad_sent[target] += arr_grad[couple]*count

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

def get_hessian_newsent(word):
    """
    Creates dictionary of hessian matrices.
    list_index_weat: list
    """
    idx_w = vocab[word]
    with open(str(idx_w)+'_hessian_newsent_SA.pkl', "rb") as f:
        hessian_word = pickle.load(f)
    return hessian_word

def get_emb_pert(perturbed_emb, word):
    """
    Apply dictionary to given word to get the embedding.
    word: word of interest
    return: word embedding
    """
    return perturbed_emb[word]


def get_perturbed_emb_sent(wv, WEATLIST, diz_sent_tuple, diz_tuple_sent, diz_mapping, arr_grad, sent_id, 
                            sent_text, newsent=False):
    diz_grad_sent = get_sent_grad(
        diz_sent_tuple, diz_mapping, arr_grad, sent_id)
    print(diz_grad_sent.keys())
    perturbed_emb = {}  # dictionary {word: emb}
    #hessian_diz = get_hessian_words(list_index_weat, diz_tuple_sent, diz_mapping, arr_grad)
    for word in WEATLIST:
        emb = get_emb_og(wv, word)
        if word in sent_text:  # single entity
            V = len(wv._vocab)
            if word in diz_grad_sent:
                # gradient for the word of interest
                grad_sent = diz_grad_sent[word]
                # print(grad_sent)
                if newsent==False:
                    hessian = np.linalg.inv(get_hessian(word))
                else:
                    hessian = np.linalg.inv(get_hessian_newsent(word))
            else:  # in case the term is appearing only in the sentence to be removed
                grad_sent = np.zeros(HIDDEN_SIZE)
                hessian = np.zeros((HIDDEN_SIZE, HIDDEN_SIZE))
            # decide whether grad_sent should be multiplied by V or not - technically yes to fully follow Brunet et al.
            emb = emb + (1/V)*np.dot(hessian, grad_sent)
        perturbed_emb[word] = emb

    return perturbed_emb


def do_eff_size(sent, WEATLIST, dict_sent_tuple_count, array_gradients, wv):

    perturbed_emb = get_perturbed_emb_sent(
        wv, WEATLIST, dict_sent_tuple_count, None, None, array_gradients, 1, sent)

    t1 = np.array([get_emb_pert(perturbed_emb, word) for word in S])
    t2 = np.array([get_emb_pert(perturbed_emb, word) for word in T])
    att1 = np.array([get_emb_pert(perturbed_emb, word) for word in A])
    att2 = np.array([get_emb_pert(perturbed_emb, word) for word in B])

    simSA = np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean()
    simTA = np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean()
    simSB = np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean()
    simTB = np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean()

    eff_size_pert = effect_sizev2(t1, t2, att1, att2)

    perturbed_emb_newsent = get_perturbed_emb_sent(
        wv, WEATLIST, dict_sent_tuple_count, None, None, array_gradients, 1, sent, newsent=True)

    print("cosine similarity: ")
    print("he ", cos_similarity(get_emb_pert(perturbed_emb, "he"), get_emb_pert(perturbed_emb_newsent, "he")))
    print("science ", cos_similarity(get_emb_pert(perturbed_emb, "science"), get_emb_pert(perturbed_emb_newsent, "science")))
    print("him ", cos_similarity(get_emb_pert(perturbed_emb, "him"), get_emb_pert(perturbed_emb_newsent, "him")))
    print("nasa ", cos_similarity(get_emb_pert(perturbed_emb, "nasa"), get_emb_pert(perturbed_emb_newsent, "nasa")))
    print("she ", cos_similarity(get_emb_pert(perturbed_emb, "she"), get_emb_pert(perturbed_emb_newsent, "she")))
    print("art ", cos_similarity(get_emb_pert(perturbed_emb, "art"), get_emb_pert(perturbed_emb_newsent, "art")))

    print("Effect size perturbed corpus: ", eff_size_pert)
    print("Similarity S-A: ", simSA)
    print("Similarity T-A: ", simTA)
    print("Similarity S-B: ", simSB)
    print("Similarity T-B: ", simTB)
    print(' '.join(sent))


S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
     "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance",
     "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

WEATLIST = S.copy() + T.copy() + A.copy() + B.copy()


with open('./content/data_correct_post_train/vocab_v2.pkl', 'rb') as han:
    vocab = pickle.load(han)

with open('./content/data_correct_post_train/inv_vocab_v2.pkl', 'rb') as han:
    inv_vocab = pickle.load(han)

with open('./content/data_correct_post_train/unigram_counts_v2.pkl', 'rb') as han:
    unigram_counts = pickle.load(han)

weights = [torch.from_numpy(np.load('syn0_final_torch.npy')).requires_grad_().cuda(),
           torch.from_numpy(np.load('syn1_final_torch.npy')).requires_grad_().cuda()]


BATCH_SIZE = 256
NEGATIVES = 5
EPOCHS = 3
SAMPLING_RATE = 1E-3
MIN_FREQ = 60
WINDOW_SIZE = 5
LEARNING_RATE = 1E-3
HIDDEN_SIZE = 300

"""# cose post_train_allsent"""

list_index_weat = [vocab[word] for word in WEATLIST]

list_approx_words = list(set(list_index_weat.copy()))

"""
Read CORPUS with only sentences containing WEAT words
"""


syn0_final = np.load('syn0_final_torch.npy')

vocab_words = [key for key in vocab]
wv = WordVectors(syn0_final, vocab_words)

"""# proviamo con frasi fittizie (S+A, T+A)"""

S_A = S.copy()+A.copy()
random.seed(0)
random.shuffle(S_A)
print(S_A)

T_A = T.copy()+A.copy()
random.seed(1)
random.shuffle(T_A)
print(T_A)

S_B = S.copy()+B.copy()
random.seed(2)
random.shuffle(S_B)
print(S_B)

T_B = T.copy()+B.copy()
random.seed(3)
random.shuffle(T_B)
print(T_B)


# compute original effect size and similarities:
t1 = np.array([get_emb_og(wv, word) for word in S])
t2 = np.array([get_emb_og(wv, word) for word in T])
att1 = np.array([get_emb_og(wv, word) for word in A])
att2 = np.array([get_emb_og(wv, word) for word in B])

simSA = np.array([mean_cos_similarity(tar, att1) for tar in t1]).mean()
simTA = np.array([mean_cos_similarity(tar, att1) for tar in t2]).mean()
simSB = np.array([mean_cos_similarity(tar, att2) for tar in t1]).mean()
simTB = np.array([mean_cos_similarity(tar, att2) for tar in t2]).mean()

print("effect size corpus orginale: ", effect_sizev2(t1, t2, att1, att2))
print("similarity S-A origin: ", simSA)
print("similarity T-A origin: ", simTA)
print("similarity S-B origin: ", simSB)
print("similarity T-B origin: ", simTB)

# for sentence in [S_A, T_A, S_B, T_B]:

#     sent_tuples = get_tuples(sentence, vocab)
#     grads = {(inp_label[0], inp_label[1]): get_grad(
#         inp_label, unigram_counts, weights, vocab) for inp_label in sent_tuples}

#     diz_count_sent = Counter(sent_tuples)

#     diz_sent_reshaped = np.array(
#         [[x[0][2], ((x[0][0], x[0][1]), x[1])] for x in list(diz_count_sent.items())])

#     sentDict = defaultdict(list)
#     for key, val in diz_sent_reshaped:
#         sentDict[key].append(tuple(val))

#     do_eff_size(sentence, WEATLIST, sentDict, grads, wv)

#grads  # array gradients

for sentence in [S_A]:

    sent_tuples = get_tuples(sentence, vocab)
    grads = {(inp_label[0], inp_label[1]): get_grad(
        inp_label, unigram_counts, weights, vocab) for inp_label in sent_tuples}

    diz_count_sent = Counter(sent_tuples)

    diz_sent_reshaped = np.array(
        [[x[0][2], ((x[0][0], x[0][1]), x[1])] for x in list(diz_count_sent.items())])

    sentDict = defaultdict(list)
    for key, val in diz_sent_reshaped:
        sentDict[key].append(tuple(val))

    do_eff_size(sentence, WEATLIST, sentDict, grads, wv)

