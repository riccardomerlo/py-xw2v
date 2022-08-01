import numpy as np
from gensim.models import Word2Vec
import pickle
import pandas as pd
from dataset_torch import read_corpus

dir = "/home/rmerlo/py-xw2v/gensim_t2/"

def hash_uni(astring):
   return ord(astring[0])

def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score

sent_id = 412444

model_retrain = Word2Vec.load(dir+'gensim_model_retrain_'+str(sent_id)+'_1')
model_og = Word2Vec.load(dir+'gensim_model')

with open(dir+"vocab_gensim.pkl", "rb") as v:
  vocab = pickle.load(v)

# find 50 most common words and words in sent
list_top50 = model_og.wv.index_to_key[:50]
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')
sentence = text[sent_id]
list_words_sent = [x for x in sentence if x in vocab]

# similarities in original corpus
list_w1 = []
list_w2 = []
list_sim = []
for w1 in list_words_sent:
    for w2 in list_top50:
        list_w1.append(w1)
        list_w2.append(w2)
        list_sim.append(cos_similarity(model_og.wv[w1], model_og.wv[w2]))

df_og = pd.DataFrame(columns=['w1', 'w2', 'sim'])
df_og['w1'] = list_w1
df_og['w2'] = list_w2
df_og['sim'] = list_sim

df_og.to_csv("sim_og_sent_"+str(sent_id)+".csv", index=False)

# similarities in retraining
list_w1 = []
list_w2 = []
list_sim = []
for w1 in list_words_sent:
    for w2 in list_top50:
        list_w1.append(w1)
        list_w2.append(w2)
        list_sim.append(cos_similarity(model_retrain.wv[w1], model_retrain.wv[w2]))

df_retrain = pd.DataFrame(columns=['w1', 'w2', 'sim'])
df_retrain['w1'] = list_w1
df_retrain['w2'] = list_w2
df_retrain['sim'] = list_sim

df_retrain.to_csv("sim_retrain_"+str(sent_id)+".csv", index=False)
