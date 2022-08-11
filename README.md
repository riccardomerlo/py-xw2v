Implementation of word2vec using pytorch and gensim, with a spin-off to explain the model post-training following the idea of Brunet et al. Directories:

* avg_seed: contains the code to run experiments with multiple seeds (then take the average of similarities) with the sentence of interest at the end of the dataset, training and retraining with 1 epoch and SGD-like or SGD-like inverse loss (both negative sampling and true context) approximation methods.
* corpus: contains the original NYT corpus (nyt_dal_90_ad_oggi.zip) and some samples of the corpus no longer in use.
* current_experiments: contains the code for approximation of the experiments brunet-like, SGD-like, SGD-like inverse loss and hybrid. Hessian matrices are taken from run_hessians.py located in the /post directory.
* old_files_gensim: contains post_training with gensim files that are no longer in use.
* old_files_pytorch: contains training, retraining and post_training with pytorch files that are no longer in use.
* post: post training useful scripts, in particular get_hessians_utils.py and run_hessians.py.
* useful_files+gensim: retraining with gensim file.


The file requirements.txt contains libraries requirements to be set for the working environment.
