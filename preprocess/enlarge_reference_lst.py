import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def word2vec_phishing_expedition(path_to_model, start_lst, similarity_threshold):
   model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
   similar_words_lst = []
   for start_w in start_lst:
    if model.has_index_for(start_w)==True:
      similar_words = model.most_similar(start_w, topn=5)
      for word, similarity in similar_words:
        if similarity > similarity_threshold:
          if word not in start_lst:
            similar_words_lst.append(word)
   return list(set(similar_words_lst))
