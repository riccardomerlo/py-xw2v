from model_torch import Word2VecModel
from dataset_torch import create_skipgram, read_corpus

import torch
import pickle

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)

"""
Define CONSTANTS
"""

output_dir = 'test1/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



BATCH_SIZE = 'auto'
BATCH_N_SENTENCE = 10
NEGATIVES = 5
EPOCHS = 3
SAMPLING_RATE = 1E-3
MIN_FREQ = 60
WINDOW_SIZE = 5
LEARNING_RATE = 1E-3
HIDDEN_SIZE = 300
# liste_termini_weat
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa", 
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

WEATLIST = S+T+A+B

"""
Read CORPUS
"""
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')
#text = read_corpus('./corpus/nyt_articles_v2.txt')



"""
Create MODEL
"""
word2vec = Word2VecModel(hidden_size=HIDDEN_SIZE,
                         batch_size=BATCH_SIZE,
                         batch_n_sentence=BATCH_N_SENTENCE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE)

"""
Build DATASET
"""
word2vec.build_dataset(text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, SAMPLING_RATE)

with open(output_dir+"to_remove_words_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._to_remove_words, han)
with open(output_dir+"to_keep_words_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._to_keep_words, han)
with open(output_dir+"text_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._text, han)
with open(output_dir+"data_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._data, han)
with open(output_dir+"vocab_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._vocab, han)
with open(output_dir+"inv_vocab_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._inv_vocab, han)
with open(output_dir+"unigram_counts_batch1.pkl", "wb") as han:
    pickle.dump(word2vec._unigram_counts, han)

"""
Build WEIGHTS
"""
word2vec.build_weights()


"""
Train MODEL
"""
word2vec.train(EPOCHS, save=True)
