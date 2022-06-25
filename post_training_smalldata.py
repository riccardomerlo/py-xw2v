# -*- coding: utf-8 -*-
import nltk
nltk.download('punkt')
from scipy import spatial
import statistics
from word_vectors import WordVectors
import numpy as np
import pickle
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from dataset_torch import create_skipgram, read_corpus
from model_torch import Word2VecModel
from after_training_torch import _full_loss_torch, _true_loss_torch, _negative_sampling_loss_torch, post_training, get_sim_matrix, get_sim_perturbed, get_emb_og, get_emb_pert, compute_corr, get_variation_sim_matrix, effect_size, get_perturbed_emb_sent


BATCH_SIZE = 32
NEGATIVES = 5
EPOCHS = 2
SAMPLING_RATE = 1E-3
MIN_FREQ = 1
WINDOW_SIZE = 3
LEARNING_RATE = 1E-3
# liste_termini_weat
S = ["technology"]
T = ["art"]
A = ["man", "he", "him", "his"]
B = ["woman", "she", "her"]
WEATLIST = S+T+A+B

syn0 = np.load('syn0_final_torch_small.npy')
syn1 = np.load('syn1_final_torch_small.npy')

with open("data_small.pkl", "rb") as f:
  data = pickle.load(f)

with open("vocab_small.pkl", "rb") as v:
  vocab = pickle.load(v)

with open("unigram_counts_small.pkl", "rb") as u:
  unigram_counts = pickle.load(u)

with open("inv_vocab_small.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)


"""# Post-training"""

text = read_corpus('./corpus/nyt_articles_31.txt')

vocab_words = [key for key in vocab]
print('length of dataset: ', len(data))
data_post = data[:int(len(data)/EPOCHS)]

flat_data = [x for xs in data for x in xs]

#data, unigram_counts, vocab, inv_vocab = create_skipgram(text, WINDOW_SIZE, WEATLIST, MIN_FREQ, SAMPLING_RATE, 1, 1)

"""Create list of most common words to be used together with weat words list"""

most_common_words = []
un = unigram_counts.copy()
count = 50
while True:
  if len(most_common_words) == 50:
    break
  idx = np.argmax(un)
  if inv_vocab[idx] not in WEATLIST.copy():
    most_common_words.append(inv_vocab[idx])
  un.pop(idx)

list_index_weat = []
for word in S+T+A+B:
  list_index_weat.append(vocab[word])

tot_count = 0
for step, training_point in enumerate(data_post):
  inputs = training_point[0][0]
  if inputs in list_index_weat:
    tot_count+=1
print("Tot. number of inputs to consider: ", tot_count)

weights = [torch.from_numpy(syn0).requires_grad_(), torch.from_numpy(syn1).requires_grad_()]

"""Study convergence for different k values
### k=0
"""

k=0
print("post training for k: ", k)

diz_gradients_0 = {} # dictionary like {(inputs, labels, nsent): gradient}
hessian_diz_0 = {} # dictionary like {inputs: hessian}

post_training(k, vocab, inv_vocab, data_post, 1, unigram_counts, None, list_index_weat, _true_loss_torch, weights, diz_gradients_0, hessian_diz_0)

"""### k=5 like in training"""

k=5
print("post training for k: ", k)

diz_gradients_5 = {} # dictionary like {(inputs, labels, nsent): gradient}
hessian_diz_5 = {} # dictionary like {inputs: hessian}

post_training(k, vocab, inv_vocab, data_post, 1, unigram_counts, k, list_index_weat, _negative_sampling_loss_torch, weights, diz_gradients_5, hessian_diz_5)

"""### k=V-1"""

vocab_len = len(vocab)

k=vocab_len-1
print("post training for k: ", k)

diz_gradients_V = {} # dictionary like {(inputs, labels, nsent): gradient}
hessian_diz_V = {} # dictionary like {inputs: hessian}

post_training(k, vocab, inv_vocab, data_post, 1, unigram_counts, None, list_index_weat, _full_loss_torch, weights, diz_gradients_V, hessian_diz_V)

"""# Build the approximation for the embedding
# Results
### Open data
"""

syn0_final = np.load('syn0_final_torch.npy') # actually this weight matrix is the same for each value of k because it's the one obtained during training
wv = WordVectors(syn0_final, vocab_words)
vocab = vocab_words

"""### Compute similarity matrices
Similarity matrix for full corpus
"""

sim_matrix_full = get_sim_matrix(S,T,A,B,most_common_words,vocab,get_emb_og, wv)
print("sim. matrix full corpus: ", sim_matrix_full)


"""### Define k value within [0, 5, V-1]
#### k=0
"""

# Store data (serialize)
with open('dict_gradients_0.pickle', 'rb') as handle_grad:
  diz_gradients_0 = pickle.load(handle_grad)

with open('dict_hessians_0.pickle', 'rb') as handle_hes:
  hessian_diz_0 = pickle.load(handle_hes)

"""#### k=5"""

# Store data (serialize)
with open('dict_gradients_5.pickle', 'rb') as handle_grad:
  diz_gradients_5 = pickle.load(handle_grad)

with open('dict_hessians_5.pickle', 'rb') as handle_hes:
  hessian_diz_5 = pickle.load(handle_hes)

"""#### k=V-1"""

# Store data (serialize)
vocab_len = len(vocab)
with open('dict_gradients_'+str(vocab_len-1)+'.pickle', 'rb') as handle_grad:
  diz_gradients_V = pickle.load(handle_grad)

with open('dict_hessians_'+str(vocab_len-1)+'.pickle', 'rb') as handle_hes:
  hessian_diz_V = pickle.load(handle_hes)

"""### Similarity matrix for perturbed corpus - depends on k value"""

# k=0
list_sim_sent_0 = get_sim_perturbed(0, text, S, T, A, B, most_common_words, wv, diz_gradients_0, hessian_diz_0, get_emb_pert, vocab)

# k=5
list_sim_sent_5 = get_sim_perturbed(5, text, S, T, A, B, most_common_words, wv, diz_gradients_5, hessian_diz_5, get_emb_pert, vocab)

# k=V-1
list_sim_sent_V = get_sim_perturbed(vocab_len-1, text, S, T, A, B, most_common_words, wv, diz_gradients_V, hessian_diz_V, get_emb_pert, vocab)

print("computed perturbed similarities")


"""### Re-training by sent - to be completed only once and then compared with different k values
Re-training should be done on corpus without one sentence, so we need to re-define the corpus itself.
"""

random.seed(0)
text_ids = random.sample(range(len(text)), 10)

list_sim_sent_retrain = []
for sent_id in text_ids:
  sent_text = text[sent_id]

  # check that my sentence contains at least one WEAT word
  first=True
  for word in S+T+A+B:
    if word in sent_text: # word as single occurrence (not part of another word), eg. otherwise "the" is selected because it contains "he"
      if first==True: # I consider a sentence if it has at least one occurrence of WEAT words

        # settings
        BATCH_SIZE = 256
        NEGATIVES = 5
        EPOCHS = 3
        SAMPLING_RATE = 1E-3
        MIN_FREQ = 10
        WINDOW_SIZE = 5
        LEARNING_RATE = 1E-3

        log_per_steps= 1000#10000  # Every `log_per_steps` steps to log the value of loss to be minimized.

        flat_data_rt = [x for x in flat_data if x[2]!=sent_id]
        data_rt = split_given_size(flat_data_rt, BATCH_SIZE)
        data_rt = [x for x in data_rt if len(x) == BATCH_SIZE]
        unigram_counts_rt = unigram_counts.copy()
        vocab_rt = vocab.copy()
        for key in vocab:
            if key in sent_text:
                unigram_counts_rt[vocab[key]] = unigram_counts[vocab[key]]-1
        inv_vocab_rt = {v: k for k, v in vocab_rt.items()}

        #data_rt, unigram_counts_rt, vocab_rt, inv_vocab_rt = create_skipgram(text_rt, WINDOW_SIZE, WEATLIST, MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

        word2vec = Word2VecModel(unigram_counts_rt,
              hidden_size=300,
              batch_size=BATCH_SIZE,
              negatives=NEGATIVES,
              power=0.75,
              alpha=LEARNING_RATE)

        w0, w1 = word2vec.train(data_rt, save=False)

        print("Re-training for sentence "+str(sent_id)+" completed")

        #syn0_final = word2vec.syn0.detach().numpy()
        vocab_words = [word for word in vocab_rt]

        wv = WordVectors(w0, vocab_words)

        sim_matrix_full_retrain = get_sim_matrix(S,T,A,B,most_common_words,vocab_rt,get_emb_og, wv)

        complete_list = []
        for target in S+T:
            list_target = sim_matrix_full_retrain.loc[target].values.flatten().tolist()
            complete_list.append(list_target)

        list_sim_sent_retrain.append([sent_id, complete_list])

        first=False

with open('list_similarity_sent_retrain_weat+top50.pkl', 'wb') as f:
  pickle.dump(list_sim_sent_retrain, f)

# change accordingly to which one we want to investigate
with open('list_similarity_sent_retrain_weat+top50.pkl', 'rb') as f:
  list_sim_sent_retrain = pickle.load(f)

"""### Compare similarities (with a given k value)
"""

target_set = S+T

"""#### Correlation"""

compute_corr(0, target_set, list_sim_sent_0, list_sim_sent_retrain)

compute_corr(5, target_set, list_sim_sent_5, list_sim_sent_retrain)

compute_corr(vocab_len-1, target_set, list_sim_sent_V, list_sim_sent_retrain)

print("correlation computed")
