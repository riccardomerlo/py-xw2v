from gensim.models import Word2Vec
import nltk
import pickle
import numpy as np
import random 
from training_gensim import training_model
from post_train_utils import build_dataset_posttrain, get_approx


sent_id = 412444
first_sent = True # is it the first sentence you analyze? (i.e. first time training from scratch the whole corpus)
output_dir = '/home/apera/py-xw2v/gensim_multi_seed/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# set model params
min_count=15
sg = 1
window=5
vector_size=300
workers=1
sample=1e-4
alpha=0.1                    
min_alpha=1e-5
ns_exponent=0
negative=20
shrink_windows=False
epochs = 1

lr = 0.7
negative_sample = 0

for seed in range(5):
    print("computing for seed", seed)

    # train model
    if first_sent:
        training_model(output_dir, seed, sent_id, min_count, sg, window, vector_size,
                        workers, sample, alpha, min_alpha, ns_exponent, negative, epochs,
                        shrink_windows, retrain=False)
        print("training completed")
    else:
        print("training done already")

    # retrained model
    training_model(output_dir, seed, sent_id, min_count, sg, window, vector_size,
                    workers, sample, alpha, min_alpha, ns_exponent, negative, epochs,
                    shrink_windows, retrain=True)
    print("re-training completed")

    # post-train

    ## build dataset
    build_dataset_posttrain(output_dir, seed, window)
    print("post-train dataset built")

    ## get hessian matrices
    #TODO

    ## approximate

    ### SGD loss (negative sampling loss)
    get_approx(sent_id, output_dir, seed, negative, lr, negative_sample, inverse = False, negative = True)
    print('done SGD negative sampling loss')

    ### SGD inverse loss (true context loss)
    #get_approx(sent_id, output_dir, seed, negative, lr, negative_sample, inverse = True, negative = False)

