from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

text = []
with open('./corpus/nyt_articles_v2.txt', 'r') as f:
  text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]

w2v_model = Word2Vec(min_count=10,
                     window=5,
                     size=300,
                     sample=1e-3, 
                     alpha=1e-3, 
                     batch_words=256,                      
                     min_alpha=1e-3,
                     negative=5)

w2v_model.build_vocab(text, progress_per=10000)
w2v_model.train(text, total_examples=w2v_model.corpus_count, epochs=3, report_delay=1)
w2v_model.save('gensim_v2.model')

print("training completed")