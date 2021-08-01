# p084svd_test.py SVDによる次元削除
import numpy as np
import sys
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p073cos_similarity import cos_similarity
from p079ppmi import ppmi
import matplotlib.pyplot as plt
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus=corpus, vocab_size=len(word_to_id))
W = ppmi(C)
U, S, V = np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])
print(U[0, :2])
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:,1], alpha=0.5)
plt.show()