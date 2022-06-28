from model_torch import Word2VecModel
from dataset_torch import create_skipgram, read_corpus

import torch
import pickle

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)

"""
Define CONSTANTS
"""

BATCH_SIZE = 256
NEGATIVES = 5
EPOCHS = 3
SAMPLING_RATE = 1E-3
MIN_FREQ = 10
WINDOW_SIZE = 5
LEARNING_RATE = 1E-3
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
text = read_corpus('./corpus/nyt_articles_v2.txt')



"""
Create MODEL
"""
word2vec = Word2VecModel(hidden_size=300,
                         batch_size=BATCH_SIZE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE)

"""
Build DATASET
"""
word2vec.build_dataset(text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, SAMPLING_RATE)

with open("data.pkl", "wb") as han:
    pickle.dump(self._data, han)
with open("vocab.pkl", "wb") as han:
    pickle.dump(self._vocab, han)
with open("inv_vocab.pkl", "wb") as han:
    pickle.dump(self._inv_vocab, han)
with open("unigram_counts.pkl", "wb") as han:
    pickle.dump(self._unigram_counts, han)

"""
Build WEIGHTS
"""
word2vec.build_weights()


"""
Train MODEL
"""
word2vec.train(EPOCHS, save=True)
