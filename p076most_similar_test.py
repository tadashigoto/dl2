# p076most_similar.py 類似単語のランキング表示テスト
import numpy as np
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p073cos_similarity import cos_similarity
from p075most_similar import most_similar
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
most_similar('you', word_to_id=word_to_id, id_to_word=id_to_word, word_matrix=C, top=5)