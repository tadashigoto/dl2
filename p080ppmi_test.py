# p069count_base.py コサイン類似度のテスト
import numpy as np
import sys
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p073cos_similarity import cos_similarity
from p079ppmi import ppmi
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus=corpus, vocab_size=len(word_to_id))
W = ppmi(C)
np.set_printoptions(precision=3)
print('covariancd matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)