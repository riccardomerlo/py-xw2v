from gensim.models import Word2Vec
import nltk
import pickle
import numpy as np
import random 
from current_files_gensim.word_vectors import WordVectors

output_dir = 'gensim_t1/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def read_corpus(path, min_len = 3):
    text = []
    #cut_min = lambda x: x if len(x) > min_len
    try:
        with open(path, 'r') as f:
            #text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]
            for x in f.readlines():
                res = nltk.tokenize.word_tokenize(x.strip())
                if len(res) > min_len:
                    text.append(res)
    except Exception as e:
        print(e)
    return text

text = read_corpus('./corpus/nyt_dal_90_ad_oggi.txt')

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

sent_id = 412444
print(text[sent_id])
text = [text[i] for i in range(len(text)) if i!=sent_id] # remove one sent

noweat = [sent for sent in text if not contains_weat(sent, S, T, A, B)]
weat = [sent for sent in text if contains_weat(sent, S, T, A, B)]

random.seed(seed) #ensures reproducibility
random.shuffle(noweat)    

random.seed(seed) #ensures reproducibility
random.shuffle(weat)   

text = noweat+weat#put weat words at the end of the text to have better representations

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

# for word in S+T+A+B:
# 	print(word, w2v_model.wv.most_similar(positive=[word], topn=10))

w2v_model.save(output_dir+"gensim_model_retrain_"+str(sent_id))

np.save(output_dir+'syn0_final_torch_retrain_'+str(sent_id), w2v_model.wv.vectors)
np.save(output_dir+'syn1_final_torch_retrain_'+str(sent_id), w2v_model.syn1neg)

def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


def mean_cos_similarity(tar, att):
    '''
    Calculates the mean of the cosine similarity between the target and the range of attributes
    '''
    mean_cos = np.mean([cos_similarity(tar, attribute) for attribute in att])
    return mean_cos


def association(tar, att1, att2):
    '''
    Calculates the mean association between a single target and all of the attributes
    '''
    association = mean_cos_similarity(tar, att1) - mean_cos_similarity(tar, att2)

    return association


def effect_sizev2(t1, t2, att1, att2):
    '''
    Calculates the effect size (d) between the two target variables and the attributes
    Parameters:
        t1 (np.array): first target variable matrix
        t2 (np.array): second target variable matrix
        att1 (np.array): first attribute variable matrix
        att2 (np.array): second attribute variable matrix

    Returns:
        effect_size (float): The effect size, d.

    Example:
        t1 (np.array): Matrix of word embeddings for professions "Programmer, Scientist, Engineer"
        t2 (np.array): Matrix of word embeddings for professions "Nurse, Librarian, Teacher"
        att1 (np.array): matrix of word embeddings for males (man, husband, male, etc)
        att2 (np.array): matrix of word embeddings for females (woman, wife, female, etc)
    '''
    combined = np.concatenate([t1, t2])
    num1 = np.mean([association(target, att1, att2) for target in t1])
    num2 = np.mean([association(target, att1, att2) for target in t2])
    combined_association = np.array([association(target, att1, att2) for target in combined])

    dof = combined_association.shape[0]
    denom = np.sqrt(((dof-1)*np.std(combined_association, ddof=1) ** 2 ) / (dof-1))
    effect_size = (num1 - num2) / denom

    return effect_size


def get_emb_og(wv, word):
    """
    Function allowing to get the embedding of a given word.

    word: word of interest
    return: word embedding
    """

    # same as wv[word]
    return wv._syn0_final[wv._rev_vocab[word]]



#syn0_final = np.load(output_dir+'syn0_final_torch_retrain_'+str(sent_id)+'.npy')

vocab = w2v_model.wv.key_to_index

# with open(output_dir+"vocab_gensim.pkl", "rb") as f:
#     vocab = pickle.load(f)

vocab_words = [key for key in vocab]
wv = WordVectors(w2v_model.wv.vectors, vocab_words)

# our training
t1 = np.array([get_emb_og(wv, word) for word in S])
t2 = np.array([get_emb_og(wv, word) for word in T])
att1 = np.array([get_emb_og(wv, word) for word in A])
att2 = np.array([get_emb_og(wv, word) for word in B])
ef_full = effect_sizev2(t1, t2, att1, att2)
print("effect size retraining: ", ef_full)
