# p069count_base.py 共起行列のテスト
import numpy as np
import sys
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print('corpus:',corpus)
print('corpus_size:',len(corpus))
print(id_to_word)
print('len(id_to_word):', len(id_to_word))
C = create_co_matrix(corpus=corpus, vocab_size=len(id_to_word))
print(C)