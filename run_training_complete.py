from model_torch import Word2VecModel
from dataset_torch import create_skipgram, read_corpus

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
S = ["science", "technology", "physics", "chemistry", "einstein", "nasa", "experimeny",
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
Create DATASET
"""
data, unigram_counts, vocab, inv_vocab = create_skipgram(
    text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

"""
Create MODEL
"""
word2vec = Word2VecModel(unigram_counts,
                         hidden_size=300,
                         batch_size=BATCH_SIZE,
                         negatives=NEGATIVES,
                         power=0.75,
                         alpha=LEARNING_RATE)

"""
Train MODEL
"""
word2vec.train(data, save=True)