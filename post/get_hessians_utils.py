import numpy as np
import pickle

import torch
import torch.utils.data

import gc
import os
import sys

from after_training_torch import  _full_loss_torch

import random

from collections import Counter
from datetime import datetime

NEGATIVE_SAMPLE = 0 #0.75


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
                                                         distortion=NEGATIVE_SAMPLE)


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

def get_losses(LOSS, t, batch_size, weights, unigram_counts, NEGATIVES=None):

  _input = [x[0] for x in t]
  _label = [x[1] for x in t]

  if LOSS == 'NG':
    if not NEGATIVES:
      print("negatives cant be none")
    loss = _negative_sampling_loss_torch(_input, _label, batch_size, # to change if full_loss is needed
                      unigram_counts, NEGATIVES, weights, len(unigram_counts))
  if LOSS == 'FULL':
    loss = _full_loss_torch(_input, _label, batch_size, unigram_counts, None, weights, len(unigram_counts)) 


  return loss.sum()

def get_losses_negative(LOSS, t, count_t, batch_size, weights, unigram_counts, NEGATIVES=None):

  _input = [x[0] for x in t]
  _label = [x[1] for x in t]

  if LOSS == 'NG':
    if not NEGATIVES:
      print("negatives cant be none")
    loss = _negative_sampling_loss_torch(_input, _label, batch_size, # to change if full_loss is needed
                      unigram_counts, NEGATIVES, weights, len(unigram_counts))
  if LOSS == 'FULL':
    loss = _full_loss_torch(_input, _label, batch_size, unigram_counts, None, weights, len(unigram_counts)) 

  print(loss.shape) 
  # do element wise multiplication with count
  count_t = [np.repeat(x, NEGATIVES+1) for x in count_t ]
  counts = torch.from_numpy(np.array(count_t)).cuda()
  loss = loss * counts

  return loss.sum()

def get_grad(W, loss, target, weights):
  #print(loss.requires_grad)
  grad = torch.autograd.grad(
      loss.requires_grad_(), 
      weights[W], 
      create_graph=True,  # molto lento con questo ma serve per calcolare i gradienti succesivi
      retain_graph=True # obbligatorio per fare grad ancora
      )[0]
  
  return grad[target]


def get_hessian(W, target, grads, weights):

  print(grads.shape)
  hess_i = torch.autograd.grad(
          grads, 
          weights[W], 
          create_graph=False,
          retain_graph=True # necessario per calcolare nuovamente i gradienti di un altro grado
          )[0].detach().cpu().numpy()
  print(hess_i.shape)
  return hess_i #[target]

def get_batch_hessian(W, target, grads, weights):
  hessian = []
  for step, w in enumerate(grads):

      start = datetime.now()
      hess_i = torch.autograd.grad(
          w, 
          weights[W], 
          create_graph=False,
          retain_graph=True # necessario per calcolare nuovamente i gradienti di un altro grado
          )[0].detach().cpu().numpy()
      hessian.append(hess_i[target])

      end = datetime.now()
      print('done', step, end-start)

      # torch.cuda.empty_cache()
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

def batch_list(iterable, n=1):
    l = len(iterable)
    return [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]

def get_full_hessian(W, LOSS, target, tuple_set, batch_size, weights, unigram_counts, NEGATIVES=None):

  target_tuples = [x for x in tuple_set if x[W] == target]

  print('# instances', len(target_tuples))
  losses = None
  grads = None
  hessians = []
  tuple_batched = batch(target_tuples, batch_size)

  for step, t in enumerate(tuple_batched):

    #print('loss', step)
    if losses == None:
      losses = get_losses(LOSS, t, batch_size, weights, unigram_counts, NEGATIVES)
    else:
      losses += get_losses(LOSS, t, batch_size, weights, unigram_counts, NEGATIVES) 
    
    gc.collect()
    if (step % 10 == 0 or step == int(len(target_tuples)/batch_size)):  
      grads = get_grad(W, losses, target, weights)

      #print(grads.shape)
      del losses
      #gc.collect()
      #torch.cuda.empty_cache()
      
      hessians.append(get_batch_hessian(W, target, grads, weights))
      
      del grads
      gc.collect()
      torch.cuda.empty_cache()
      losses = None


  
  return np.sum(hessians, axis=0)


def get_full_hessian_negatives(W, LOSS, target, tuple_set, batch_size, weights, unigram_counts, NEGATIVES=None):

  target_tuples = [x for x in tuple_set if x[W] == target]

  print('# instances', len(target_tuples))
  losses = None
  grads = None
  hessians = []

  set_tuples = list(set(target_tuples))
  tuple_batched = batch(set_tuples, batch_size)

  tuples_counter = Counter(target_tuples)

  # fare lista con stesso ordine di target_tuple con il relativo count
  count_tuples = [tuples_counter[x] for x in set_tuples]
  count_tuples_batched = batch_list(count_tuples, batch_size)

  for step, t in enumerate(tuple_batched):

    if losses == None:
      losses = get_losses_negative(LOSS, t, count_tuples_batched[step], batch_size, weights, unigram_counts, NEGATIVES)
    else:
      losses += get_losses_negative(LOSS, t, count_tuples_batched[step], batch_size, weights, unigram_counts, NEGATIVES)
    
    print('done losses', step, len(t))
    # gc.collect()
    if (step % 10 == 0 or step == int(len(target_tuples)/batch_size)):  
      grads = get_grad(W, losses, target, weights)
      print('done grads', datetime.now())

      del losses
      start = datetime.now()
      hessians.append(get_batch_hessian(W, target, grads, weights))
      print('hess took total', datetime.now() - start)
      print('done app hessians', datetime.now())

      del grads
      gc.collect()
      torch.cuda.empty_cache()
      losses = None

  print('done hessians', datetime.now())
  
  return np.sum(hessians, axis=0)
