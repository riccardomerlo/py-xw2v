from sklearn.feature_extraction.text import CountVectorizer
import collections
import numpy as np
import nltk
nltk.download('punkt')
import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                     else torch.FloatTensor)


def get_vocab(text):
    raw_vocab = collections.Counter()
    for line in text:
        raw_vocab.update(line)
    raw_vocab = dict(raw_vocab.most_common())
    return raw_vocab


def subsampling(vocab, whitelist=[], rate=0.0001):
    """
    returns list of subsampled words to remove
    """

    # A storage to store tokens after subsampling
    new_tokens = whitelist.copy()
    # tokens is a list of word indexes from original text
    np.random.seed(0)
    for word in vocab.keys():
        frac = vocab[word]/len(vocab.keys())
        prob = (np.sqrt(frac/rate) + 1) * (rate/frac)
        if np.random.random() < prob:
            new_tokens.append(word)
    return list(set(vocab).difference(set(new_tokens)))


def high_freq(vocab, whitelist=[], min_freq=1):
    """
    keep words with frequency >= min_freq
    """
    return list(set([x for x in vocab.keys() if vocab[x] >= min_freq] + whitelist.copy()))

def low_freq(vocab, whitelist=[], min_freq=1):
    """
    keep words with frequency >= min_freq
    """
    return list(set([x for x in vocab.keys() if vocab[x] < min_freq]).difference(set(whitelist.copy())))

def apply_reduction(text, vocab, whitelist, min_freq, sampling_rate):
    """
    returns: new_vocab -> vocabolario senza le low freq words
              to_remove_words -> lista di parole da rimuovere (low freq + subsampling)
    """
    _low_freq = low_freq(vocab, whitelist.copy(), min_freq)
 
    new_vocab = vocab.copy()

    # new_vocab is the new vocab w/o low freq words
    for word in _low_freq:
      new_vocab.pop(word)
    
    subsamples = subsampling(new_vocab, whitelist.copy(), sampling_rate)

    to_remove_words = subsamples.copy() + _low_freq.copy()

    to_keep_words = set(new_vocab.keys()).difference(set(to_remove_words))
    
    return new_vocab, to_remove_words, to_keep_words


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def create_skipgram(text, window, whitelist=[], min_freq=1, sampling_rate=1e-3, epochs=1, batch_size=1):
    """
    returns: list of [target, context, sentence number], vocab
    """
    data = []
    vocab = get_vocab(text)
    new_vocab, to_remove_words, to_keep_words = apply_reduction(
        text, vocab, whitelist.copy(), min_freq, sampling_rate)

    new_vocab = dict(sorted(new_vocab.items(),key=lambda item: item[1], reverse=False))

    my_vec = {key: i for i, key in enumerate(sorted(new_vocab.keys())) }

    unigram_counts = [new_vocab[x] for x in my_vec]

    #
    # FIN QUI CI METTE POCO
    #
    with open('dataset_pre.txt', 'w') as f:
      for nsent, sentence in enumerate(text):
          sentence = [s for s in sentence if s in to_keep_words]
          for i, t in enumerate(sentence):
              contexts = list(range(i-window, i + window+1))
              contexts = [c for c in contexts if c >=
                          0 and c != i and c < len(sentence)]
              for c in contexts:
                  #data.append([my_vec[t],my_vec[sentence[c]], nsent])
                  f.write('\t'.join([str(my_vec[t]),
                              str(my_vec[sentence[c]]), str(nsent)]))
                  f.write('\n')
            
                
          if i > 2:
            sys.exit(0)

    data = split_given_size(data, batch_size)
    data = [x for x in data if len(x) == batch_size]
    data = data * epochs


    return data, unigram_counts, my_vec, {v: k for k, v in my_vec.items()} 


def get_weat_text(text, weatlist):
    """
    return text with only sentences which contain a weat word
    """
    data = []
    for sentence in text:
        for word in sentence:
            if word in weatlist:
                data.append(sentence)
                break

    return data

def get_tuple_weat(text, weatlist):
    data = []
    for nsent, sentence in enumerate(iter(text)):
        for i, t in enumerate(sentence):
            if t in weatlist:
                contexts = list(range(i-window, i + window+1))
                contexts = [c for c in contexts if c >=
                            0 and c != i and c < len(sentence)]
                for c in contexts:
                    data.append([my_vec[t],my_vec[sentence[c]], nsent])
    return data

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
