import numpy as np
from gensim.models import Word2Vec
import pickle
import pandas as pd
from dataset_torch import read_corpus
from collections import Counter

def get_tuples(list_words, sentence_words, vocab):
    _data = []

    for word in list_words:
        for c in sentence_words:
            _data.append((vocab[word], vocab[c])) # così prende però anche una parola contesto di se stessa

    return _data

def hash_uni(astring):
   return ord(astring[0])

def get_approx(word, tuples_tokeep, vocab, model_og, mapping_grads, array_gradients, alpha, V, top50=False):
    
    tuples_word = [x for x in tuples_tokeep if x[0]==vocab[word]]
    sum_grad = np.zeros(300)
    emb_og = model_og.wv[word]

    for tu in tuples_word:
        idx = mapping_grads[(tu[0], tu[1])]
        grad = array_gradients[idx]
        sum_grad += grad
    sum_grad = np.multiply(sum_grad, alpha)

    # load hessian
    if top50:
        with open(dir+str(vocab[word])+"_W"+str(0)+"_hessian_shuffle_top50.pkl", "rb") as han:
            hess = pickle.load(han)
    else:
        with open(dir+str(vocab[word])+"_W"+str(0)+"_hessian.pkl", "rb") as han:
            hess = pickle.load(han)        

    hessian = np.linalg.inv(hess)

    # influence formula
    emb_pert = emb_og + (1/V)*np.dot(hessian, sum_grad)

    return emb_pert


def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score

dir = "/home/rmerlo/py-xw2v/gensim_t2/"

sent_id = 412444 #601555
#list_values_alpha = range(-100, 100, 10)
list_values_alpha = np.arange(0, 0.20, 1e-5)

model_og = Word2Vec.load(dir+'gensim_model')

with open(dir+"vocab_gensim.pkl", "rb") as v:
  vocab = pickle.load(v)

with open(dir+"inv_vocab_gensim.pkl", "rb") as v:
  inv_vocab = pickle.load(v)

with open(dir+"data_complete.pkl", "rb") as v:
  data = pickle.load(v)

flat_data = [x for y in data for x in y]
tuples_sent = [(x[0], x[1]) for x in flat_data if x[2]==sent_id]

# find 50 most common words
list_top50 = [x for x in list(vocab.keys())[:50]]

# get words in sent
text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')
sentence = text[sent_id]
list_words_sent = {x for x in sentence if x in vocab}

# overlap between top50 and sent
overlapping = list(set(list_top50) & set(list_words_sent))

# get gradients
with open(dir+"array_gradients_sent_"+str(sent_id)+"_shuffle.pkl", "rb") as f:
    array_gradients = pickle.load(f)

with open(dir+"dict_sent_"+str(sent_id)+"_shuffle.pkl", "rb") as f:
    mapping_grads = pickle.load(f)


# get approximation
list_alpha = []
list_w1 = []
list_w2 = []
list_sim = []
for alpha in list_values_alpha:
    emb_approx = {}
    for word in list_words_sent:
        if word in overlapping:
            emb_approx[word] = get_approx(word, tuples_sent, vocab, model_og, mapping_grads, array_gradients, alpha, len(vocab), top50=True)
        else:
            emb_approx[word] = get_approx(word, tuples_sent, vocab, model_og, mapping_grads, array_gradients, alpha, len(vocab))
    # similarities in original corpus
    for w1 in list_words_sent:
        for w2 in list_top50:
            list_w1.append(w1)
            list_w2.append(w2)
            if w2 in overlapping:
                emb_w2 = emb_approx[w2]
            else:
                emb_w2 = model_og.wv[w2]
            list_sim.append(cos_similarity(emb_approx[w1], emb_w2))
            list_alpha.append(alpha)
    
    
df_approx = pd.DataFrame(columns=['w1', 'w2', 'sim', 'alpha'])
df_approx['w1'] = list_w1
df_approx['w2'] = list_w2
df_approx['sim'] = list_sim
df_approx['alpha'] = list_alpha

df_approx.to_csv("compare_alpha_step1e-5_zoom.csv", index=False)
