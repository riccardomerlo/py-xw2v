mport pickle
import numpy as np
import pandas as pd
import itertools
from word_vectors import WordVectors

lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
multi_step = False  # set to True if, given a target, all tuples with that target are processed all together for the gradients (SGD-like, SGD-like inverse ng loss, SGD-like inverse true loss)
                    # set to False if, given a target, all tuples with that target are processed one by one for the gradients (SGD-like inverse loss step-by-step)


def cos_similarity(tar, att):
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


sent_id = 412444
output_dir = '/home/apera/py-xw2v/gensim_multi_seed/'
text_dir = '/home/rmerlo/py-xw2v/gensim_t2/'

with open(text_dir+"tokenized_text.pkl", "rb") as v:
  text = pickle.load(v)


if multi_step:
    name1 = '_straight_ng_' # SGD-like
    name2 = '_inverse_true_' # SGD-like inverse true loss
    name3 = '_inverse_ng_' # SGD-like inverse ng loss

    list_names = [name1, name2, name3]

    for name in list_names:
        data = []
        dict_couple_og = {}
        dict_couple_retrain = {}
        dict_couple_approx = {}

        for lr in lr_list:
            dict_couple_og[str(lr)] = {}
            dict_couple_retrain[str(lr)] = {}
            dict_couple_approx[str(lr)] = {}
        
        for seed in range(5):

            syn0_og = np.load(output_dir+'syn0_final_gensim_seed'+str(seed)+'.npy')
            syn0_retrain = np.load(output_dir+str(sent_id)+'syn0_final_gensim_retrain_seed'+str(seed)+'.npy')

            with open(output_dir+name+str(sent_id)+'_new_embeddings_seed'+str(seed)+'_SGD.pkl', "rb") as v:
                syn0_approx = pickle.load(v)
            with open(output_dir+"vocab_gensim_seed"+str(seed)+".pkl", "rb") as v:
                vocab = pickle.load(v)

            sentence_words = set([x for x in text[sent_id] if x in vocab])

            wv0_og = WordVectors(syn0_og, list(vocab.keys()))
            wv0_retrain = WordVectors(syn0_retrain, list(vocab.keys()))
            
            for pair in itertools.combinations(sentence_words, 2):
                
                for lr in lr_list:
                    sim_og = cos_similarity(wv0_og.get_embedding(pair[0]), wv0_og.get_embedding(pair[1]))
                    sim_retrain = cos_similarity(wv0_retrain.get_embedding(pair[0]), wv0_retrain.get_embedding(pair[1]))
                    sim_approx = cos_similarity(syn0_approx[str(lr)][vocab[pair[0]]], syn0_approx[str(lr)][vocab[pair[1]]])

                    if pair not in dict_couple_og[str(lr)].keys():
                        dict_couple_og[str(lr)][pair] = sim_og 
                        dict_couple_retrain[str(lr)][pair] = sim_retrain
                        dict_couple_approx[str(lr)][pair] = sim_approx 
                    else:
                        dict_couple_og[str(lr)][pair] += sim_og 
                        dict_couple_retrain[str(lr)][pair] += sim_retrain
                        dict_couple_approx[str(lr)][pair] += sim_approx  


        for lr in dict_couple_og.keys():
            for couple in dict_couple_og[lr].keys():
                data.append([lr, couple[0], couple[1], dict_couple_og[lr][couple]/len(range(5)), dict_couple_retrain[lr][couple]/len(range(5)), dict_couple_approx[lr][couple]/len(range(5))])

        df = pd.DataFrame(data, columns=['lr', 'w1', 'w2', 'og', 'retrain', 'approx'])
        df.to_csv('compare_SGD_'+name+str(sent_id)+'_avg_seed.csv', index=False)

else:
    data = []
    dict_couple_og = {}
    dict_couple_retrain = {}
    dict_couple_approx = {}

    for lr in lr_list:
        dict_couple_og[str(lr)] = {}
        dict_couple_retrain[str(lr)] = {}
        dict_couple_approx[str(lr)] = {}

    for seed in range(5):

        syn0_og = np.load(output_dir+'syn0_final_gensim_seed'+str(seed)+'.npy')
        syn0_retrain = np.load(output_dir+str(sent_id)+'syn0_final_gensim_retrain_seed'+str(seed)+'.npy')

        with open(output_dir+str(sent_id)+'_new_embeddings_seed'+str(seed)+'_alpha'+str(alpha)+'_SGDbetter.pkl', "rb") as v:
            syn0_approx = pickle.load(v)
        with open(output_dir+"vocab_gensim_seed"+str(seed)+".pkl", "rb") as v:
            vocab = pickle.load(v)

        sentence_words = set([x for x in text[sent_id] if x in vocab])

        wv0_og = WordVectors(syn0_og, list(vocab.keys()))
        wv0_retrain = WordVectors(syn0_retrain, list(vocab.keys()))

        for pair in itertools.combinations(sentence_words, 2):
            
            for lr in lr_list:
                sim_og = cos_similarity(wv0_og.get_embedding(pair[0]), wv0_og.get_embedding(pair[1]))
                sim_retrain = cos_similarity(wv0_retrain.get_embedding(pair[0]), wv0_retrain.get_embedding(pair[1]))
                sim_approx = cos_similarity(syn0_approx[str(lr)][vocab[pair[0]]], syn0_approx[str(lr)][vocab[pair[1]]])

                if pair not in dict_couple_og[str(lr)].keys():
                    dict_couple_og[str(lr)][pair] = sim_og 
                    dict_couple_retrain[str(lr)][pair] = sim_retrain
                    dict_couple_approx[str(lr)][pair] = sim_approx 
                else:
                    dict_couple_og[str(lr)][pair] += sim_og 
                    dict_couple_retrain[str(lr)][pair] += sim_retrain
                    dict_couple_approx[str(lr)][pair] += sim_approx  


    for lr in dict_couple_og.keys():
        for couple in dict_couple_og[lr].keys():
            data.append([lr, couple[0], couple[1], dict_couple_og[lr][couple]/len(range(5)), dict_couple_retrain[lr][couple]/len(range(5)), dict_couple_approx[lr][couple]/len(range(5))])

    df = pd.DataFrame(data, columns=['lr', 'w1', 'w2', 'og', 'retrain', 'approx'])
    df.to_csv('compare_SGDbetter_'+str(sent_id)+'_multilr_avg_seed.csv', index=False)
