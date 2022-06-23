import numpy as np
import pickle
import nltk
from dataset_torch import create_skipgram, read_corpus
from model_torch import Word2VecModel
import nltk
nltk.download('punkt')
from scipy import spatial
import statistics
from word_vectors import WordVectors                                                                                
import pandas as pd
import torch
import random
from after_training_torch import get_sim_matrix, get_emb_og

S = ["technology"]
T = ["art"]
A = ["man", "he", "him", "his"]
B = ["woman", "she", "her"]

WEATLIST = S+T+A+B

syn0 = np.load('syn0_final_torch.npy')
syn1 = np.load('syn1_final_torch.npy')

with open("data.pkl", "rb") as f:
  data = pickle.load(f)

with open("vocab.pkl", "rb") as v:
  vocab = pickle.load(v)
  
with open("unigram_counts.pkl", "rb") as u:
  unigram_counts = pickle.load(u) 
  
with open("inv_vocab.pkl", "rb") as iv:
  inv_vocab = pickle.load(iv)   
  
text = []
with open('./corpus/nyt_articles_31.txt', 'r') as f:
  text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]
  
flat_data = [x for xs in data for x in xs]

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

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))
  
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
        BATCH_SIZE = 32
        NEGATIVES = 5
        EPOCHS = 2
        SAMPLING_RATE = 1E-3
        MIN_FREQ = 1
        WINDOW_SIZE = 3
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
  

