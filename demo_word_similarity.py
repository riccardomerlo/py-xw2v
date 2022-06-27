from word_vectors import WordVectors                                                                                
import numpy as np  
import pickle                                                                                                

# syn_final.npy: storing word embeddings, numpy array of shape [vocab_size, hidden_size]
# 'vocab.txt': text file storing words in vocabulary, one word per line

query = 'technology'
num_similar_words = 10
syn0_final = np.load('syn0_final_torch.npy')
vocab_words = []                                                                                                   
with open('vocab.pkl','rb') as f: 
    vocab_words = pickle.load(f) 
                                                                                                                    
wv = WordVectors(syn0_final, list(vocab_words.keys()))   
print(wv.most_similar(query, num_similar_words))
