# p087ptb_test.py PTBデータセットのテスト
import numpy as np
import sys
sys.path.append('./sample')
from dataset import ptb
from p075most_similar import most_similar
from p072create_co_matrix import create_co_matrix
from p079ppmi import ppmi
window_size = 2
wordvec_size = 100
corpus, word_to_id, id_to_word = ptb.load_data('train')
vacab_size = len(word_to_id)
print('counting co-occurrencd ...')
C = create_co_matrix(corpus, vacab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)
print('calculating SVD ...')
try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    U, S, V = np.linalg.svd(W)
word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
