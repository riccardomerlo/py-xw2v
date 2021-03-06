import numpy as np
import torch
import pickle
import torch.utils.data
from typing import List, \
    Union
from dataset_torch import split_given_size, get_vocab, apply_reduction, subsample_prob, cache_subsample_prob, get_sampled_sent, get_dynamic_window
import random

from torch.optim import lr_scheduler
import sys

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)


def fixed_unigram_candidate_sampler(
        true_classes: Union[np.array, torch.Tensor],
        num_samples: int,
        unigrams: List[Union[int, float]],
        distortion: float = 1.):
    
    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.cpu().detach().numpy()

    #print(true_classes)
    # print(true_classes.shape)
    # print(num_samples)
    if true_classes.shape[0] != num_samples:
        raise ValueError(
            'true_classes must be a 2D matrix with shape (num_samples, num_true)')
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    # print('unigrams:', unigrams)
    indices = np.arange(num_samples)
    result = np.zeros(num_samples, dtype=np.int64)
    while len(indices) > 0:
        # print('len(indices):', len(indices))
        sampler = torch.utils.data.WeightedRandomSampler(
            unigrams, len(indices))
        candidates = np.array(list(sampler))
        candidates = np.reshape(candidates, (len(indices), 1))
        # print('candidates:', candidates)
        # print('true_classes:', true_classes[indices, :])
        result[indices] = candidates.T
        mask = (candidates == true_classes[indices, :])
        mask = mask.sum(1).astype(np.bool)
        # print('mask:', mask)
        indices = indices[mask]
    return result


class Word2VecModel(torch.nn.Module):

    def __init__(self,
                 hidden_size=300,
                 batch_size=256,
                 fixed_batch_size=True, #if truncate batches with len < batch_size
                 batch_n_sentence=1,
                 negatives=5,
                 power=0.75,
                 alpha=0.002,
                 random_seed=0,
                 output_dir=''):

        super(Word2VecModel, self).__init__()
        
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._fixed_batch_size = fixed_batch_size
        self._batch_n_sentence = batch_n_sentence
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._random_seed = random_seed
        self._output_dir = output_dir
        

        

    def build_weights(self):
        """
        instance the weights matrix
        """

        # syn0
        torch.manual_seed(self._random_seed)
        syn0 = torch.empty(self._vocab_size, self._hidden_size)
        self.syn0 = torch.nn.Parameter(data=torch.nn.init.uniform_(
            syn0, a=(-0.5/self._hidden_size), b=(0.5/self._hidden_size)), requires_grad=True)

        # syn1
        torch.manual_seed(self._random_seed)
        syn1 = torch.empty(self._vocab_size, self._hidden_size)
        self.syn1 = torch.nn.Parameter(data=torch.nn.init.uniform_(
            syn1, a=-0.1, b=0.1), requires_grad=True)

    def forward(self, inputs, labels, batch_size, unigram_counts, negatives, weights):
        loss = self._negative_sampling_loss_torch(
            inputs, labels, batch_size, unigram_counts, negatives, weights)
        return loss

    def _negative_sampling_loss_torch(self, inputs, labels, batch_size, unigram_counts, negatives, weights=None):
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
        if not weights:
            syn0 = self.syn0
            syn1 = self.syn1
        else:
            syn0 = weights[0]
            syn1 = weights[1]

        if (self._batch_size == 'auto') or (not self._fixed_batch_size):
            batch_size = len(inputs)

        torch.manual_seed(np.array(inputs).mean())  # TODO change seed?

        true_classes_array = torch.unsqueeze(
            torch.tensor(np.repeat(labels, negatives)), 1)
        # print(true_classes_array.shape)
        sampled_values = fixed_unigram_candidate_sampler(true_classes=true_classes_array,
                                                         num_samples=negatives*batch_size,
                                                         unigrams=unigram_counts,
                                                         distortion=self._power)
        sampled_values = split_given_size(sampled_values, batch_size)
        sampled_values = np.array(
            [x for x in sampled_values if len(x) == batch_size])
        sampled_values = torch.from_numpy(sampled_values).cuda()
        # sampled_mat = torch.reshape(sampled_values, (batch_size, negatives))

        inputs_syn0 = torch.index_select(
            syn0, 0, torch.from_numpy(np.array(inputs)).cuda())
        true_syn1 = torch.index_select(
            syn1, 0, torch.from_numpy(np.array(labels)).cuda())

        # sampled_syn1 = syn1[sampled_values]
        list_sampled_syn1 = [torch.index_select(syn1, 0, batch) for batch in sampled_values]
        # for batch in sampled_values:
        #     sampled_syn1_batch = torch.index_select(syn1, 0, batch)
        #     list_sampled_syn1.append(sampled_syn1_batch)

        sampled_syn1 = torch.stack(list_sampled_syn1, 0)

        true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
        true_logits.requires_grad_()

        sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                      sampled_syn1.permute(1, 2, 0))
        sampled_logits.requires_grad_()

        loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
        sampled_cross_entropy = loss(
            sampled_logits, torch.zeros_like(sampled_logits))

        loss = torch.concat(
            [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)
        return loss


    def build_dataset(self, text, max_window, whitelist=[], min_freq=1, sampling_rate=1e-3, reorder_weat=True):
        """
        """
        self._sampling_rate = sampling_rate
        data = []
        vocab = get_vocab(text)
        count_vocab, to_remove_words, to_keep_words = apply_reduction(
            text, vocab, whitelist.copy(), min_freq)

        self._to_remove_words = to_remove_words
        self._to_keep_words = to_keep_words

        count_vocab = dict(sorted(count_vocab.items(),key=lambda item: item[1], reverse=False))

        ord_vocab = {key: i for i, key in enumerate(sorted(count_vocab.keys())) }

        unigram_counts = [count_vocab[x] for x in ord_vocab]

        inv_vocab = {v: k for k, v in ord_vocab.items()}

        #un po' lento...
        self._text = [[s for s in sentence if s in to_keep_words] for sentence in text]
        self._vocab = ord_vocab
        self._unigram_counts = unigram_counts
        self._vocab_size = len(unigram_counts)
        self._inv_vocab = inv_vocab

        S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
            "experiment", "astronomy"]
        T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
        A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
        B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
        
        _data = []
        _data_weat = []

        step_log = 1000

        subsample_cache = {}
        if self._sampling_rate != 0:
            subsample_cache = cache_subsample_prob(count_vocab, self._sampling_rate)
        
        # change order of sents in text
#        if reorder_weat:
#            self._text = [[s for s in sentence if s in S+T+A+B] for sentence in self._text]

        #cache_sentence = []
        _sent_batch = []
        tmp_batch_sentence = 0
        for n_sent, sentence in enumerate(self.get_text()):
            # subsample (rimuovo parole in base alla loro probabilit??)
            if self._sampling_rate != 0:
                sentence = get_sampled_sent(n_sent, sentence, subsample_cache)
                #cache_sentence.append(sentence)

            if self._batch_n_sentence == 1:
                _sent_batch = [] #support array to keep all tuples of single sentence
            for i, t in enumerate(iter(sentence)):
                
                window = get_dynamic_window(n_sent, i, max_window)    

                contexts = list(range(i-window, i + window+1))
                contexts = [c for c in contexts if c >=
                            0 and c != i and c < len(sentence)]
                for c in contexts:
                    # TARGET, CONTEXT, nsent
                    if self._batch_size == 'auto':
                        _sent_batch.append([self._vocab[t], self._vocab[sentence[c]], n_sent])
                    else:
                        if t in S+T+A+B: # save tuples with weat
                            _data_weat.append([self._vocab[t], self._vocab[sentence[c]], n_sent])
                        else:
                            _data.append([self._vocab[t], self._vocab[sentence[c]], n_sent])
            if (self._batch_size == 'auto') and (self._batch_n_sentence == 1):
                _data.append(_sent_batch)
            
            tmp_batch_sentence += 1
            if (self._batch_size == 'auto') and (self._batch_n_sentence < tmp_batch_sentence):
                _data.append(_sent_batch)
                _sent_batch = []
                tmp_batch_sentence = 0
            
            if n_sent % step_log == 0:
                print(round(n_sent/len(self._text)*100, 2), end=' ')

        if self._batch_size == 'auto':
            #each sentence is a batch
            #each item is a sentence

            #shuffle data
            random.seed(self._random_seed) #ensures reproducibility
            random.shuffle(_data)            

            #batch size is the number of tuples of each sentence
            _data_batch = _data

        else:
            #shuffle data before batching
            random.seed(self._random_seed) #ensures reproducibility
            random.shuffle(_data)

            random.seed(self._random_seed) #ensures reproducibility
            random.shuffle(_data_weat) # ?? da fare?

            _data = _data_weat + _data # put weat tuples first

            _data_flat = _data.copy()

            if self._fixed_batch_size:
                #batch _inputs and _labels
                _data_batch = split_given_size(_data, self._batch_size)
            
                #remove batch if size < batch_size
                _data_batch = [x for x in _data_batch if len(x) == self._batch_size]

                _data_batch = list(reversed(_data_batch)) # change order of tuples to have weat as the last ones
            else:
                _data_batch = _data
            
        #repeat epoch times (done during training)

        self._data = _data_batch
        self._data_flat = _data_flat

        #with open("new_sentence.pkl", "wb") as han:
        #    pickle.dump(cache_sentence, han)

    def load_data(
                self,
                output_dir='',
                vocab=None,
                inv_vocab=None,
                text=None,
                data=None,
                unigram_counts=None
                ):
        if vocab:
            with open(output_dir+vocab, "rb") as han:
                self._vocab=pickle.load(han)
            self._vocab_size = len(self._vocab)
            print(vocab, "loaded")
        if inv_vocab:
            with open(output_dir+inv_vocab, "rb") as han:
                self._inv_vocab=pickle.load(han)
            self._vocab_size = len(self._inv_vocab)
            print(inv_vocab, "loaded")
        if text:
            with open(output_dir+text, "rb") as han:
                self._text=pickle.load(han)
            print(text, "loaded")
        if data:
            with open(output_dir+data, "rb") as han:
                self._data=pickle.load(han)
            print(data, "loaded")
        if unigram_counts:
            with open(output_dir+unigram_counts, "rb") as han:
                self._unigram_counts=pickle.load(han)
            self._vocab_size = len(self._unigram_counts)
            print(unigram_counts, "loaded")

    def get_text(self):

        return iter(self._text)

    def get_data(self):
        
        return iter(self._data)


    def train(self, epochs, save=True, retrain=False, save_batch1=False):
        """trains model
        Returns:
            syn0: target embeddings matrix
            syn1: context embeddings matrix
        """
        # average_loss = 0. #TODO
        optimizer = torch.optim.SGD(self.parameters(), lr=self._alpha)
        
        log_per_steps = 2500
        
        print('Total number of steps: ', len(self._text))
        
        inputs_batch = []
        labels_batch = []

        scheduler = lr_scheduler.StepLR(optimizer, step_size=len(self._data)/200, gamma=0.1)

        for epoch in range(epochs):
            # TODO: shuffling e re-batching anche per ciascuna epoch?
            print('-----start-epoch',epoch,'----')
            # scheduler is re-initialized at the beginning of each epoch
            #scheduler = lr_scheduler.StepLR(optimizer, step_size=len(self._data)/100, gamma=0.1)
            for step, batch in enumerate(self.get_data()):
            #TODO salvare [self._vocab[t], self._vocab[sentence[c]]]
            #TODO costruire input, label e poi ripeto epoch volte, quindi calcolo batch e addestro
                if len(batch) == 0:
                    continue
                
                target_batch = [x[0] for x in batch]
                context_batch = [x[1] for x in batch] 
                n_sent = [x[2] for x in batch]

                # reset gradients
                """
                As of v1.7.0, Pytorch offers the option to reset the gradients 
                to None optimizer.zero_grad(set_to_none=True) instead of filling 
                them with a tensor of zeroes. The docs claim that this setting 
                reduces memory requirements and slightly improves performance, 
                but might be error-prone if not handled carefully.
                """
                optimizer.zero_grad(set_to_none=True)

                neg_loss = self._negative_sampling_loss_torch(
                    target_batch, context_batch, self._batch_size, self._unigram_counts, self._negatives)
                neg_loss.sum().backward(create_graph=True, retain_graph=True)

                # update gradients
                optimizer.step()

                # scheduler
                scheduler.step()
                
                if  step % log_per_steps == 0:
                    print('% :', round(step/len(self._data)* 100,1))

                
                # np.save(self._output_dir+'syn0_final_torch_onestep', self.syn0.cpu().detach().numpy())
                # np.save(self._output_dir+'syn1_final_torch_onestep', self.syn1.cpu().detach().numpy())

                # sys.exit(0)
                
            print('-----end-epoch',epoch,'----')


        print("Training completed")

        # get weights matrixes as numpy array
        syn0_final = self.syn0.cpu().detach().numpy()
        syn1_final = self.syn1.cpu().detach().numpy()

        if save and retrain==False:
            np.save(self._output_dir+'syn0_final_torch', syn0_final)
            np.save(self._output_dir+'syn1_final_torch', syn1_final)
        elif save and retrain:
            np.save(self._output_dir+'syn0_final_torch_retrain', syn0_final)
            np.save(self._output_dir+'syn1_final_torch_retrain', syn1_final) 
        elif save and save_batch1:
            np.save(self._output_dir+'syn0_final_torch_batch1', syn0_final)
            np.save(self._output_dir+'syn1_final_torch_batch1', syn1_final)            

            
        return [syn0_final, syn1_final]
