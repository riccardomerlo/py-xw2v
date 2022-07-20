# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.utils.data

import gc
import os
import sys

from after_training_torch import  _full_loss_torch

import random

MODE = 'train' # 'post', 'train'
LOSS = 'NG' # 'NG', 'FULL'
NEGATIVES = 20

print("MODE", MODE, "---- LOSS", LOSS)

output_dir = '/home/apera/py-xw2v/test256_valerio+neg/'

with open('./corpus/nyt_dal_90_ad_oggi.txt', 'r') as han:
  text = [x.strip() for x in han.readlines()]

if MODE == 'post':
  with open(output_dir+'dict_sent_tuple_count_shuffle.pkl', 'rb') as han:
    sent_tuple_count = pickle.load(han)
if MODE == 'train':
  with open(output_dir+'dict_sent_tuple_count_traindata_shuffle.pkl', 'rb') as han:
    sent_tuple_count = pickle.load(han)

if MODE == 'post':
  with open(output_dir+'data_v2.pkl', 'rb') as han: 
    data_v2 = pickle.load(han)
if MODE == 'train':
  with open(output_dir+'data_shuffle.pkl', 'rb') as han: 
    data_v2 = pickle.load(han)


with open(output_dir + 'vocab_shuffle.pkl', 'rb') as han:
  vocab_v2 = pickle.load(han)

with open(output_dir + 'inv_vocab_shuffle.pkl', 'rb') as han:
  inv_vocab_v2 = pickle.load(han)

with open(output_dir + 'unigram_counts_shuffle.pkl', 'rb') as han:
  unigram_counts_v2 = pickle.load(han)

weights = [torch.from_numpy(np.load(output_dir+'syn0_final_torch.npy')).requires_grad_().cuda(),
           torch.from_numpy(np.load(output_dir+'syn1_final_torch.npy')).requires_grad_().cuda() ]



# liste_termini_weat
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa", 
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab_v2[word])


list_approx_words = list(set(list_index_weat.copy()))

flat_data = [x for y in data_v2 for x in y] # 6 secondi
tuple_set = {(x[0], x[1]) for x in iter(flat_data)} # 32 secondi
tuple_set_1 = [x for x in tuple_set if x[0] in list_approx_words] #1m 7s secondi

tuple_set_ord = tuple_set_1.copy()
tuple_set_ord.sort(key=lambda i:i[1],reverse=True)

conx = [x[1] for x in tuple_set_ord]

d_conx = np.diff(conx)

ind_conx = np.argwhere(d_conx != 0).reshape(-1)
new_ind_conx = np.concatenate([[0], ind_conx + 1, [len(tuple_set_ord)]])

full_batch = [tuple_set_ord[new_ind_conx[x]:new_ind_conx[x+1]] for x in range(len(new_ind_conx) -1)]

print('fine load -----------------')





def fixed_unigram_candidate_sampler(
        true_classes,
        inputs,
        num_samples,
        unigrams,
        distortion = 1.):

    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.detach().cpu().numpy()
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    #print(indices)
    result = np.zeros((len(inputs), num_samples))
    for idx in range(len(inputs)):
        value = inputs[idx]
        unigrams_new = unigrams.copy()
        unigrams_new[true_classes[idx]] = 0 # così non prende la true class come possibile negative per se stessa
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
        syn0 = weights[0].cuda()
        syn1 = weights[1].cuda()


        true_classes_array = torch.unsqueeze(torch.tensor(label).clone().detach(), 1)
        inputs_array = torch.unsqueeze(torch.tensor(input).clone().detach(), 1)
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
            sampled_syn1_batch = torch.index_select(syn1, 0, torch.tensor(torch.from_numpy(batch).clone().detach(), dtype=torch.int32).cuda().clone().detach())
            list_sampled_syn1.append(sampled_syn1_batch)
            del sampled_syn1_batch

        sampled_syn1 = torch.stack(list_sampled_syn1, 0)

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)

        del true_syn1

        #true_logits.requires_grad_()
        sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                      sampled_syn1.permute(0,2,1))
        
        del sampled_syn1
        del inputs_syn0
        #sampled_logits.requires_grad_()

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

def get_losses(t, batch_size, weights, unigram_counts):

  _input = [x[0] for x in t]
  _label = [x[1] for x in t]

  if LOSS == 'NG':
    loss = _negative_sampling_loss_torch(_input, _label, batch_size, # to change if full_loss is needed
                      unigram_counts, NEGATIVES, weights, len(unigram_counts))
  if LOSS == 'FULL':
    loss = _full_loss_torch(_input, _label, batch_size, unigram_counts, None, weights, len(unigram_counts)) 

  return loss.sum()

def get_grad(loss, target, weights):
  #print(loss.requires_grad)
  grad = torch.autograd.grad(
      loss.requires_grad_(), 
      weights[0], 
      create_graph=True,  # molto lento con questo ma serve per calcolare i gradienti succesivi
      retain_graph=True # obbligatorio per fare grad ancora
      )[0]
  
  return grad[target]

def get_batch_hessian(target, grads, weights):
  hessian = []
  for step, w in enumerate(grads):
      hess_i = torch.autograd.grad(
          w, 
          weights[0], 
          create_graph=False,
          retain_graph=True # necessario per calcolare nuovamente i gradienti di un altro grado
          )[0].detach().cpu().numpy()
      hessian.append(hess_i[target])

      torch.cuda.empty_cache()
      #print('done', step)

  hess = np.stack(hessian)

  del hessian
  del grads
  gc.collect()
  torch.cuda.empty_cache()
    
  return hess

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_full_hessian(target, batch_size, weights, unigram_counts):

  target_tuples = [x for x in tuple_set_1 if x[0] == target]

  print('# instances', len(target_tuples))
  losses = None
  grads = None
  hessians = []
  tuple_batched = batch(target_tuples, batch_size)

  for step, t in enumerate(tuple_batched):

    #print('loss', step)
    if losses == None:
      losses = get_losses(t, batch_size, weights, unigram_counts)
    else:
      losses += get_losses(t, batch_size, weights, unigram_counts) 
    
    gc.collect()
    if (step % 10 == 0 or step == int(len(target_tuples)/batch_size)):  
      grads = get_grad(losses, target, weights)
      print(grads.shape)
      del losses
      #gc.collect()
      #torch.cuda.empty_cache()

      hessians.append(get_batch_hessian(target, grads, weights))

      del grads
      gc.collect()
      torch.cuda.empty_cache()
      losses = None


  
  return np.sum(hessians, axis=0)

"""
MAIN
"""
batch_size = 16384 # dà errore di cuda out of memory
for _target in list_approx_words: # for each WEAT word
  hess = get_full_hessian(_target, batch_size, weights, unigram_counts_v2)
  print("done", _target)
  if MODE == 'post' and LOSS == 'NG':
    with open(output_dir+str(_target)+"_hessian_shuffle.pkl", "wb") as han:
        pickle.dump(hess, han)
  if MODE == 'train' and LOSS == 'NG':
    with open(output_dir+str(_target)+"_hessian_traindata_shuffle.pkl", "wb") as han:
        pickle.dump(hess, han)
  if MODE == 'post' and LOSS == 'FULL':
    with open(output_dir+str(_target)+"_hessian_full_shuffle.pkl", "wb") as han:
        pickle.dump(hess, han)
  if MODE == 'train' and LOSS == 'FULL':
    with open(output_dir+str(_target)+"_hessian_traindata_full_shuffle.pkl", "wb") as han:
        pickle.dump(hess, han)




