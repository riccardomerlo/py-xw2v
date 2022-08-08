from gensim.models import Word2Vec
import nltk
import pickle
import numpy as np
import random 

text_dir = '/home/rmerlo/py-xw2v/gensim_t2/'

def contains_weat(sent, S, T, A, B):
    for word in sent:
        if word in S+T+A+B:
            return True
    return False


def hash_uni(astring):
   return ord(astring[0])


def prepare_for_training(output_dir, seed, sent_id,):

    with open(text_dir+"tokenized_text.pkl", "rb") as v:
        text = pickle.load(v)

    # remove sent of interest
    text_og = text.copy()
    text = text[:sent_id]+text[sent_id+1:]

    random.seed(seed) #ensures reproducibility
    random.shuffle(text)    

    # put sent of interest at the end
    text = text + text_og[sent_id]

    return text


def prepare_for_retraining(output_dir, seed, sent_id,):

    with open(text_dir+"tokenized_text.pkl", "rb") as v:
        text = pickle.load(v)

    # remove sent of interest
    text = text[:sent_id]+text[sent_id+1:]

    random.seed(seed) #ensures reproducibility
    random.shuffle(text)    

    return text



def training_model(output_dir, seed, sent_id, min_count, sg, window, vector_size,
                    workers, sample, alpha, min_alpha, ns_exponent, negative, epochs,
                    shrink_windows, retrain=False):
    
    if retrain:
        text = prepare_for_retraining(output_dir, seed, sent_id)
    else:
        text = prepare_for_training(output_dir, seed, sent_id)

    w2v_model = Word2Vec(min_count=min_count,
                        sg = sg, 
                        window=window,
                        vector_size=vector_size,
                        workers=workers,
                        sample=sample,  
                        alpha=alpha,               
                        min_alpha=min_alpha,
                        ns_exponent=ns_exponent,
                        hashfxn=hash_uni,
                        seed=seed,
                        negative=negative,
                        shrink_windows=shrink_windows)

    w2v_model.build_vocab(text, progress_per=10000)

    w2v_model.train(text, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)

    vocab = w2v_model.wv.key_to_index
    inv_vocab = {v: k for k, v in vocab.items()} 
    unigram_counts = []
    for key in vocab:
        unigram_counts.append(w2v_model.wv.get_vecattr(key, "count"))

    if retrain:
        with open(output_dir+str(sent_id)+"vocab_gensim_retrain_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(vocab, han)
        with open(output_dir+str(sent_id)+"inv_vocab_gensim_retrain_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(inv_vocab, han)
        with open(output_dir+str(sent_id)+"unigram_counts_gensim_retrain_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(unigram_counts, han)

        w2v_model.save(output_dir+str(sent_id)+"gensim_model_retrain_seed"+str(seed))

        np.save(output_dir+str(sent_id)+'syn0_final_gensim_retrain_seed'+str(seed), w2v_model.wv.vectors)
        np.save(output_dir+str(sent_id)+'syn1_final_gensim_retrain_seed'+str(seed), w2v_model.syn1neg)
    else:
        with open(output_dir+"vocab_gensim_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(vocab, han)
        with open(output_dir+"inv_vocab_gensim_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(inv_vocab, han)
        with open(output_dir+"unigram_counts_gensim_seed"+str(seed)+".pkl", "wb") as han:
            pickle.dump(unigram_counts, han)

        w2v_model.save(output_dir+"gensim_model_seed"+str(seed))

        np.save(output_dir+'syn0_final_gensim_seed'+str(seed), w2v_model.wv.vectors)
        np.save(output_dir+'syn1_final_gensim_seed'+str(seed), w2v_model.syn1neg)
