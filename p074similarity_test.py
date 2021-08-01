# p069count_base.py コサイン類似度のテスト
import numpy as np
import sys
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p073cos_similarity import cos_similarity
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus=corpus, vocab_size=len(word_to_id))
print(C)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
c2 = C[word_to_id['and']]
c3 = C[word_to_id['say']]
print(cos_similarity(c0, c1))
print(cos_similarity(c1, c2))
print(cos_similarity(c2, c3))
