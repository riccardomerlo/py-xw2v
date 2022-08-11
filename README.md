Implementation of word2vec using pytorch, with a spin-off to explain the model post-training following the idea of Brunet et al.

* model_torch: contains word2vec model methods

* dataset_torch: contains methods useful to building the dataset from corpus

* run_training_*: main file to create datasets, word2vec model and execute training

* gradients_dict_creation: computes gradient for each tuple having at least a WEAT word

* get_hessians: creates hessians from gradients dictionary 

* after_training_torch:

* evaluation_bias:

* post_training_*:

* retraining_embeddings:

* word_vectors: contains utils methods to explore the model embedding matrix W0
