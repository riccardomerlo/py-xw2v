"""
computes approximation calculating loss with inverse logits for each tuple of the removed sentence

"""


from word_vectors import WordVectors
import pickle
import numpy as np
import torch
from get_hessians_utils import fixed_unigram_candidate_sampler
from dataset_torch import read_corpus
from collections import Counter
import gc
import sys

import os

NEGATIVES = 20 # 20 if ng

NEGATIVE_SAMPLE = 0 # all vocabs have equal probability of being sampled TODO tweak negative_sample (maybe do many trials)

tabeluga = False # if mult by V 

def _negative_sampling_loss_torch(_input, _label, unigram_counts, negatives, weights, vocab_len):
        """Builds the negative sampling loss with regular logits (true > ones, negs > zeros).
        Args:
          input: int of shape [batch_size] (skip_gram)
          label: int of shape [batch_size] 
          batch_size: batch size chosen
          unigram_counts: list of sorted counts of words in vocab
          negatives: number of negatives to consider
          weights: list of weights, syn0 and syn1 matrices
        Returns:
          loss: float tensor of shape [batch_size, vocab_size].
        """
        syn0 = weights[0].cuda()
        syn1 = weights[1].cuda()

        true_classes_array = torch.unsqueeze(torch.tensor(_label), 1)
        inputs_array = torch.unsqueeze(torch.tensor(_input), 1)
        sampled_values = fixed_unigram_candidate_sampler(true_classes=true_classes_array,
                                                         inputs=inputs_array,
                                                         num_samples=negatives,
                                                         unigrams=unigram_counts,
                                                         distortion=NEGATIVE_SAMPLE,
                                                         target_seed=False) # TODO cause why not?

        inputs_syn0 = torch.index_select(
            syn0, 0, torch.from_numpy(np.array(_input)).cuda())
        true_syn1 = torch.index_select(
            syn1, 0, torch.from_numpy(np.array(_label)).cuda())

        list_sampled_syn1 = []
        for batch in sampled_values:
            sampled_syn1_batch = torch.index_select(syn1, 0, torch.tensor(torch.from_numpy(batch), dtype=torch.int32).cuda())
            list_sampled_syn1.append(sampled_syn1_batch)
            del sampled_syn1_batch

        sampled_syn1 = torch.stack(list_sampled_syn1, 0)

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)

        del true_syn1

        sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                      sampled_syn1.permute(0,2,1))
        del sampled_syn1
        del inputs_syn0

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


def _negative_sampling_loss_torch_inverse(_input, _label, unigram_counts, negatives, weights, vocab_len):
        """Builds the negative sampling loss with inverse logits (true > zeros, negs > ones).
        Args:
          input: int of shape [batch_size] (skip_gram)
          label: int of shape [batch_size] 
          batch_size: batch size chosen
          unigram_counts: list of sorted counts of words in vocab
          negatives: number of negatives to consider
          weights: list of weights, syn0 and syn1 matrices
        Returns:
          loss: float tensor of shape [batch_size, vocab_size].
        """
        syn0 = weights[0].cuda()
        syn1 = weights[1].cuda()

        true_classes_array = torch.unsqueeze(torch.tensor(_label), 1)
        inputs_array = torch.unsqueeze(torch.tensor(_input), 1)
        sampled_values = fixed_unigram_candidate_sampler(true_classes=true_classes_array,
                                                         inputs=inputs_array,
                                                         num_samples=negatives,
                                                         unigrams=unigram_counts,
                                                         distortion=NEGATIVE_SAMPLE,
                                                         target_seed=False) # TODO cause why not?

        inputs_syn0 = torch.index_select(
            syn0, 0, torch.from_numpy(np.array(_input)).cuda())
        true_syn1 = torch.index_select(
            syn1, 0, torch.from_numpy(np.array(_label)).cuda())

        list_sampled_syn1 = []
        for batch in sampled_values:
            sampled_syn1_batch = torch.index_select(syn1, 0, torch.tensor(torch.from_numpy(batch), dtype=torch.int32).cuda())
            list_sampled_syn1.append(sampled_syn1_batch)
            del sampled_syn1_batch

        sampled_syn1 = torch.stack(list_sampled_syn1, 0)

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)

        del true_syn1

        sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                      sampled_syn1.permute(0,2,1))
        del sampled_syn1
        del inputs_syn0

        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        true_cross_entropy = loss(true_logits, torch.zeros_like(true_logits))
        del true_logits
        sampled_cross_entropy = loss(
            sampled_logits, torch.ones_like(sampled_logits))
        del sampled_logits
        gc.collect()

        loss = torch.concat(
            [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)
        return loss

def _true_loss_torch_inverse(_input, _label, weights):
        """Builds the negative sampling loss with inverse logits (true > zeros, negs > ones).
        Args:
          input: int of shape [batch_size] (skip_gram)
          label: int of shape [batch_size] 
          weights: list of weights, syn0 and syn1 matrices
        Returns:
          loss: float tensor of shape [batch_size, vocab_size].
        """

        syn0 = weights[0].cuda()
        syn1 = weights[1].cuda()

        true_classes_array = torch.unsqueeze(torch.tensor(_label), 1)
        inputs_array = torch.unsqueeze(torch.tensor(_input), 1)
        
        inputs_syn0 = torch.index_select(
            syn0, 0, torch.from_numpy(np.array(_input)).cuda())
        true_syn1 = torch.index_select(
            syn1, 0, torch.from_numpy(np.array(_label)).cuda())

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
        del true_syn1

        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        true_cross_entropy = loss(true_logits, torch.zeros_like(true_logits))
        del true_logits
        gc.collect()

        # loss = torch.concat(
        #     [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)

        loss = true_cross_entropy.unsqueeze(1)
        return loss

def _true_loss_torch(_input, _label, weights):
        """Builds the true context loss with regular logits (true > ones, negs > zeros).
        Args:
          input: int of shape [batch_size] (skip_gram)
          label: int of shape [batch_size] 
          weights: list of weights, syn0 and syn1 matrices
        Returns:
          loss: float tensor of shape [batch_size, vocab_size].
        """

        syn0 = weights[0].cuda()
        syn1 = weights[1].cuda()

        true_classes_array = torch.unsqueeze(torch.tensor(_label), 1)
        inputs_array = torch.unsqueeze(torch.tensor(_input), 1)
        
        inputs_syn0 = torch.index_select(
            syn0, 0, torch.from_numpy(np.array(_input)).cuda())
        true_syn1 = torch.index_select(
            syn1, 0, torch.from_numpy(np.array(_label)).cuda())

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
        del true_syn1

        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
        del true_logits
        gc.collect()

        # loss = torch.concat(
        #     [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)

        loss = true_cross_entropy.unsqueeze(1)
        return loss


def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


def get_grad(loss, weights, target):
  '''
  Calculates the gradient of loss of the tuples with target=target given the weights.
  '''
  grad = torch.autograd.grad(loss.sum(),
                            weights[0],
                            create_graph=True,
                            retain_graph=True
                            )[0]

  return grad[target].detach().cpu().numpy()

def get_hessian_posttrain(idx_w):
  """
  load hessian of the given word.

  idx_w: id of word in vocab
  """
  if os.path.exists(output_dir+str(idx_w)+"_W"+str(0)+"_hessian_shuffle_top50.pkl"):
    with open(output_dir+str(idx_w)+"_W"+str(0)+"_hessian_shuffle_top50.pkl", "rb") as han:
        hess = pickle.load(han)
  else:
    with open(output_dir+str(idx_w)+"_W"+str(0)+"_hessian.pkl", "rb") as han:
        hess = pickle.load(han)    

  return hess

output_dir = '/home/rmerlo/py-xw2v/gensim_t2/'

syn0 = np.load(output_dir+'syn0_final_torch.npy')
print('loaded', 'syn0')
syn1 = np.load(output_dir+'syn1_final_torch.npy')
print('loaded', 'syn1')

with open(output_dir+"data_complete.pkl", "rb") as f:
  data = pickle.load(f)

print('loaded', 'data')

with open(output_dir+"vocab_complete.pkl", "rb") as v:
  vocab = pickle.load(v)

print('loaded', 'vocab')

with open(output_dir+"unigram_counts_complete.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

print('loaded', 'unigram_counts')

with open(output_dir+"inv_vocab_complete.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)

print('loaded', 'inv_vocab')


"""# Try with one sent"""

weights = [torch.from_numpy(syn0).requires_grad_().cuda(), torch.from_numpy(syn1).requires_grad_().cuda()]

V = len(vocab)

wv1 = WordVectors(syn1, list(vocab.keys()))
wv0 = WordVectors(syn0, list(vocab.keys()))

"""
Read CORPUS with only sentences containing WEAT words
"""
with open(output_dir+"tokenized_text.pkl", "rb") as v:
  text = pickle.load(v)

print('loaded', 'text')


def get_approx(sentence_id, brunetlike, sgdlike, sgdlike_invloss, hybrid, ng):
  
  # get sentence's words
  sent_text = text[sentence_id]
  print('full sentence', sent_text)
  print('existing vocabs', [x for x in sent_text if x in vocab])

  # get all tuples of corpus
  flat_data = [x for y in data for x in y] # 6 secondi
  # get sentence's tuples
  sentence_tuples = [(x[0], x[1]) for x in iter(flat_data) if x[2] == sentence_id] 
  print('sent tuples computed')

  new_embeddings = {}

  # define lr values list depending on the method
  if brunetlike or hybrid:
    lr_list = [0.1, 1, 10, 100, 1000, 10000]
  elif sgdlike or sgdlike_invloss:
    lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
  
  for lr in lr_list:
    new_embeddings[str(lr)] = {}

    for word in set([vocab[x] for x in sent_text if x in vocab]):

      # get target grouped tuples for each word
      target_tuples = [(x[0], x[1]) for x in sentence_tuples if x[0] == word]
      targets = [x[0] for x in target_tuples]
      contexts = [x[1] for x in target_tuples]

      # calc loss for each tuple to be removed, depending on the chosen method
      if brunetlike or sgdlike:
        target_loss = _negative_sampling_loss_torch(targets, contexts, unigram_counts, NEGATIVES, weights, len(vocab))
      
      elif sgdlike_invloss or hybrid:
        if ng:
          target_loss = _negative_sampling_loss_torch_inverse(targets, contexts, unigram_counts, NEGATIVES, weights, len(vocab))
        else:
          target_loss = _true_loss_torch_inverse(targets, contexts, weights)

      # calc gradient 
      grad = get_grad(target_loss, weights, word)

      # get hessian and compute inverted hessian
      hessian = np.linalg.inv(get_hessian_posttrain(word))

      if brunetlike or hybrid:
        new_emb = wv0.get_embedding(inv_vocab[word]) + (1/V)*np.multiply(np.dot(hessian, grad), lr)
        if brunetlike:
          name = str(sentence_id)+'_emb_brunetlike'
        elif hybrid:
          if ng:
            name = str(sentence_id)+'_emb_hybrid_ng'
          else:
            name = str(sentence_id)+'_emb_hybrid_true'
      elif sgdlike or sgdlike_invloss:
        if sgdlike:
          name = str(sentence_id)+'_emb_sgdlike'
        elif sgdlike_invloss:
          if ng:
            name = str(sentence_id)+'_emb_sgdlike_inv_ng'
          else:
            name = str(sentence_id)+'_emb_sgdlike_inv_true'

        new_emb = wv0.get_embedding(inv_vocab[word]) + np.multiply(grad, lr)

      new_embeddings[str(lr)][word] = new_emb
    print('learning rate: ', lr, 'done')


  with open(output_dir+name+'.pkl', 'wb') as han:
    pickle.dump(new_embeddings, han)
  
  print('done', str(sentence_id))


for _id in [412444]: 
  get_approx(_id, brunetlike=True, sgdlike=False, sgdlike_invloss=False, hybrid=False,
                  ng=True)

