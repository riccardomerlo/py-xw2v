from gensim.models import Word2Vec
import nltk
import pickle
import numpy as np
import random 

output_dir = 'gensim_t1/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def read_corpus(path, min_len = 3):
    text = []
    try:
        with open(path, 'r') as f:
            
            for x in f.readlines():
                res = nltk.tokenize.word_tokenize(x.strip())
                if len(res) > min_len:
                    text.append(res)
    except Exception as e:
        print(e)
    return text

text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')
# text = []
# with open('./corpus/nyt_dal_90_ad_oggi.txt', 'r') as f:
#   text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]

S = ["science", "technology", "physics", "chemistry", "einstein", "nasa",
    "experiment", "astronomy"]
T = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]
A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
seed = 0

def contains_weat(sent, S, T, A, B):
    for word in sent:
        if word in S+T+A+B:
            return True
    return False

noweat = [sent for sent in text if not contains_weat(sent, S, T, A, B)]
weat = [sent for sent in text if contains_weat(sent, S, T, A, B)]

random.seed(seed) #ensures reproducibility
random.shuffle(noweat)    

random.seed(seed) #ensures reproducibility
random.shuffle(weat)   

text = noweat+weat #put weat words at the end of text to have better representations

w2v_model = Word2Vec(min_count=15,
                    sg = 1, 
                    window=5,
                    vector_size=300,
                    workers=18,
                    sample=1e-4, 
                    alpha=0.1, #0.025,                      
                    min_alpha=1e-5,
                    ns_exponent=0,
                    negative=20)

w2v_model.build_vocab(text, progress_per=10000)

w2v_model.train(text, total_examples=w2v_model.corpus_count, epochs=15, report_delay=1)

vocab = w2v_model.wv.key_to_index
inv_vocab = {v: k for k, v in vocab.items()} 
unigram_counts = []
for key in vocab:
    unigram_counts.append(w2v_model.wv.get_vecattr(key, "count"))

with open(output_dir+"vocab_gensim.pkl", "wb") as han:
    pickle.dump(vocab, han)
with open(output_dir+"inv_vocab_gensim.pkl", "wb") as han:
    pickle.dump(inv_vocab, han)
with open(output_dir+"unigram_counts_gensim.pkl", "wb") as han:
    pickle.dump(unigram_counts, han)

spec = """
        min_count=15,
        sg = 1, 
        window=5,
        vector_size=300,
        sample=1e-4,
        workers=18,
        alpha=0.1,                      
        min_alpha=1e-5,
        negative=20,
        ns_exponent=0,
        epochs=15
"""
with open(output_dir+"spec.txt","wb") as han:
    pickle.dump(spec, han)


for word in S+T+A+B:
	print(word, w2v_model.wv.most_similar(positive=[word], topn=10))

w2v_model.save(output_dir+"gensim_model")

np.save(output_dir+'syn0_final_torch', w2v_model.wv.vectors)
np.save(output_dir+'syn1_final_torch', w2v_model.syn1neg)
