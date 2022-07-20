import pickle
from collections import Counter 

output_dir = '/home/apera/py-xw2v/test256_valerio+neg+sched_epoch/'

with open(output_dir+"data_shuffle.pkl", "rb") as f:
  data = pickle.load(f)

with open(output_dir+"inv_vocab_shuffle.pkl", "rb") as f:
  inv_vocab = pickle.load(f)

with open(output_dir+"unigram_counts_shuffle.pkl", "rb") as f:
    unigram_counts = pickle.load(f)


flat_data = [y for x in data for y in x]

dict_counts = Counter([inv_vocab[x[0]] for x in flat_data])

from word_vectors import WordVectors                                                                                
import numpy as np  
import pickle      

# syn_final.npy: storing word embeddings, numpy array of shape [vocab_size, hidden_size]
# 'vocab.txt': text file storing words in vocabulary, one word per line

S = ["science", "technology", "physics", "chemistry", "einstein", "nasa", 
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]


num_similar_words = 10
syn0_final = np.load(output_dir+'syn0_final_torch.npy')
vocab_words = []                                                                                                   
with open(output_dir+'vocab_shuffle.pkl','rb') as f: 
     vocab_words = pickle.load(f) 

for query in S+T+A+B:                                                                                                          
     wv = WordVectors(syn0_final, list(vocab_words.keys()))   
     print(query, unigram_counts[vocab_words[query]], dict_counts[query], wv.most_similar(query, num_similar_words))
# print word, counts of occurrences in text, number of tuples with word as targer, most similar words
