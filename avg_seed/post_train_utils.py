"""
# Load libraries
"""

import nltk
nltk.download('punkt')
from scipy import spatial
#from word_vectors import WordVectors
import numpy as np
import pickle
import torch
import torch.utils.data
from word_vectors import WordVectors
from get_hessians_utils import fixed_unigram_candidate_sampler
import gc

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                     else torch.FloatTensor)

text_dir = '/home/rmerlo/py-xw2v/gensim_t2/'

def hash_uni(astring):
   return ord(astring[0])

def get_text(text):
    return iter(text)

def build_dataset_gensim(text, vocab, unigram_counts, inv_vocab, max_window):
    """
    Builds post training dataset by creating tuples that only contain weat words.
    """

    _text = [[s for s in sentence if s in vocab] for sentence in text] # nb: già vocab contiene solo parole con una certa min freq
    _vocab = vocab
    _unigram_counts = unigram_counts
    _inv_vocab = inv_vocab
    
    _data = []

    for n_sent, sentence in enumerate(get_text(_text)):
      for i, t in enumerate(iter(sentence)):
        window = max_window
        contexts = list(range(i-window, i + window+1))
        contexts = [c for c in contexts if c >= 0 and c != i and c < len(sentence)]
        for c in contexts:
            _data.append([_vocab[t], _vocab[sentence[c]], n_sent])

    return _data, _vocab, _inv_vocab, _unigram_counts


def build_dataset_posttrain(output_dir, seed, window_size):

    with open(output_dir+"vocab_gensim_seed"+str(seed)+".pkl", "rb") as v:
        vocab = pickle.load(v)

    with open(output_dir+"unigram_counts_gensim_seed"+str(seed)+".pkl", "rb") as u:
        unigram_counts = pickle.load(u)

    with open(output_dir+"inv_vocab_gensim_seed"+str(seed)+".pkl", "rb") as iv:
        inv_vocab = pickle.load(iv)

    with open(text_dir+"tokenized_text.pkl", "rb") as v:
        text = pickle.load(v)

    data, vocab, inv_vocab, unigram_counts = build_dataset_gensim(text, vocab, unigram_counts, inv_vocab, window_size)

    with open(output_dir+"data_complete_seed"+str(seed)+".pkl", "wb") as han:
        pickle.dump(data, han)
    with open(output_dir+"vocab_complete_seed"+str(seed)+".pkl", "wb") as han:
        pickle.dump(vocab, han)
    with open(output_dir+"inv_vocab_complete_seed"+str(seed)+".pkl", "wb") as han:
        pickle.dump(inv_vocab, han)
    with open(output_dir+"unigram_counts_complete_seed"+str(seed)+".pkl", "wb") as han:
        pickle.dump(unigram_counts, han)


def _negative_sampling_loss_torch(_input, _label, unigram_counts, negatives, weights, negative_sample):
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
                                                         distortion=negative_sample,
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


def _negative_sampling_loss_torch_inverse(_input, _label, unigram_counts, negatives, weights, negative_sample):
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
                                                         distortion=negative_sample,
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

       # true_classes_array = torch.unsqueeze(torch.tensor(_label), 1)
       # inputs_array = torch.unsqueeze(torch.tensor(_input), 1)
        
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
        true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
        del true_logits
        gc.collect()

        # loss = torch.concat(
        #     [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)

        loss = true_cross_entropy.unsqueeze(1)
        return loss

def get_grad(loss, weights, target):
  grad = torch.autograd.grad(loss.sum(),
                            weights[0],
                            create_graph=True,
                            retain_graph=True
                            )[0]

  return grad[target].detach().cpu().numpy()


def get_grad_both(loss, weights, target, W):
  grad = torch.autograd.grad(loss.sum(),
                            weights[W],
                            create_graph=True,
                            retain_graph=True
                            )[0]

  return grad[target].detach().cpu().numpy()


def get_approx(sent_id, output_dir, seed, negatives, lr, negative_sample, inverse, negative):

    syn0 = np.load(output_dir+'syn0_final_gensim_seed'+str(seed)+'.npy')
    syn1 = np.load(output_dir+'syn1_final_gensim_seed'+str(seed)+'.npy')

    with open(output_dir+"data_complete_seed"+str(seed)+".pkl", "rb") as f:
        data = pickle.load(f)

    with open(output_dir+"vocab_complete_seed"+str(seed)+".pkl", "rb") as v:
        vocab = pickle.load(v)

    with open(output_dir+"unigram_counts_complete_seed"+str(seed)+".pkl", "rb") as u:
        unigram_counts = pickle.load(u)

    with open(output_dir+"inv_vocab_complete_seed"+str(seed)+".pkl", "rb") as iv:
        inv_vocab = pickle.load(iv)

    weights = [torch.from_numpy(syn0).requires_grad_().cuda(), torch.from_numpy(syn1).requires_grad_().cuda()]

    wv0 = WordVectors(syn0, list(vocab.keys()))

    with open(text_dir+"tokenized_text.pkl", "rb") as v:
        text = pickle.load(v)

    # get sentence's words
    sent_text = text[sent_id]

    print('full sentence', sent_text)

    print('existing vocabs', [x for x in sent_text if x in vocab])

    # get sentence's tuples
    sentence_tuples = [(x[0], x[1]) for x in iter(data) if x[2] == sent_id] 

    new_embeddings = {}
    for alpha in lr:
        new_embeddings[str(alpha)] = {}
        for word in set([vocab[x] for x in sent_text if x in vocab]):

            # get target grouped tuples for each word
            target_tuples = [(x[0], x[1]) for x in sentence_tuples if x[0] == word]

            targets = [x[0] for x in target_tuples]
            contexts = [x[1] for x in target_tuples]
            # calc loss for each tuple to be removed
            if inverse and negative:
                target_loss = _negative_sampling_loss_torch_inverse(targets, contexts, unigram_counts, negatives, weights, negative_sample)
                name = '_inverse_ng_'
            elif not inverse and negative:
                target_loss = _negative_sampling_loss_torch(targets, contexts, unigram_counts, negatives, weights, negative_sample)
                name = '_straight_ng_'
            elif inverse and not negative:
                target_loss = _true_loss_torch_inverse(targets, contexts, weights)
                name = '_inverse_true_'
            elif not inverse and not negative:
                target_loss = _true_loss_torch(targets, contexts, weights)
                name = '_straight_true_'

            # calc gradient 
            grad = get_grad(target_loss, weights, word)

            # approx target embedding (simulating SGD step)
            new_emb = wv0.get_embedding(inv_vocab[word]) + grad * alpha

            new_embeddings[str(alpha)][word] = new_emb
        print('learning rate: ', alpha, 'done')
    
    with open(output_dir+name+str(sent_id)+'_new_embeddings_seed'+str(seed)+'_SGD.pkl', 'wb') as han:
        pickle.dump(new_embeddings, han)


def get_approx_step(sent_id, output_dir, seed, lr):

    with open(output_dir+"data_complete_seed"+str(seed)+".pkl", "rb") as f:
        data = pickle.load(f)

    with open(output_dir+"vocab_complete_seed"+str(seed)+".pkl", "rb") as v:
        vocab = pickle.load(v)

    with open(output_dir+"unigram_counts_complete_seed"+str(seed)+".pkl", "rb") as u:
        unigram_counts = pickle.load(u)

    with open(output_dir+"inv_vocab_complete_seed"+str(seed)+".pkl", "rb") as iv:
        inv_vocab = pickle.load(iv)

    with open(text_dir+"tokenized_text.pkl", "rb") as v:
        text = pickle.load(v)

    # get sentence's words
    sent_text = text[sent_id]

    print('full sentence', sent_text)

    print('existing vocabs', [x for x in sent_text if x in vocab])

    # get sentence's tuples
    sentence_tuples = [(x[0], x[1]) for x in data if x[2] == sent_id] 

    ordered_vocabs = list(reversed([x for x in sent_text if x in vocab]))

    new_embeddings = {}
    for alpha in lr:
        new_embeddings[str(alpha)] = {}

        # set weights
        syn0 = np.load(output_dir+'syn0_final_gensim_seed'+str(seed)+'.npy')
        syn1 = np.load(output_dir+'syn1_final_gensim_seed'+str(seed)+'.npy')
        weights = [torch.from_numpy(syn0).requires_grad_().cuda(), torch.from_numpy(syn1).requires_grad_().cuda()]
        wv0 = WordVectors(syn0, list(vocab.keys()))
        wv1 = WordVectors(syn1, list(vocab.keys()))

        for word in ordered_vocabs:
            print('target word:', word)
            word_tuples = [tu for tu in sentence_tuples if tu[0]==vocab[word]]

            for tu in word_tuples:
                target = tu[0]
                context = tu[1]
                target_loss = _true_loss_torch_inverse(target, context, weights)

                # calc gradients
                grad_target = get_grad_both(target_loss, weights, target, 0)
                grad_context = get_grad_both(target_loss, weights, context, 1)

                # approx embeddings (simulating SGD step)
                new_emb_tar = wv0.get_embedding(inv_vocab[target]) + grad_target * alpha 
                new_emb_cont = wv1.get_embedding(inv_vocab[context]) + grad_context * alpha

                # update embedding matrix
                syn0[target] = new_emb_tar
                syn1[context] = new_emb_cont
                weights = [torch.from_numpy(syn0).requires_grad_().cuda(), torch.from_numpy(syn1).requires_grad_().cuda()]
                wv0 = WordVectors(syn0, list(vocab.keys()))
                wv1 = WordVectors(syn1, list(vocab.keys()))

            new_embeddings[str(alpha)][word] = wv0.get_embedding(word)
        print('learning rate: ', alpha, 'done')
    
    
    with open(output_dir+str(sent_id)+'_new_embeddings_seed'+str(seed)+'_SGDbetter.pkl', 'wb') as han:
        pickle.dump(new_embeddings, han)
