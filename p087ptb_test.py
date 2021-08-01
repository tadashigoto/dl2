# p087ptb_test.py PTBデータセットのテスト
import numpy as np
import sys
sys.path.append('./sample')
from dataset import ptb
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p073cos_similarity import cos_similarity
from p079ppmi import ppmi
import matplotlib.pyplot as plt
corpus, word_to_id, id_to_word = ptb.load_data('train')
print('corpus size',len(corpus))
print('corpus[:30]',corpus[:30])
print()
print('id_to_word[0]:',id_to_word[0])
print('id_to_word[1]:',id_to_word[1])
print('id_to_word[2]:',id_to_word[2])
print()
print("word_to_id['car']:",word_to_id['car'])
print("word_to_id['nappy']:",word_to_id['happy'])
print("word_to_id['lexus']:",word_to_id['lexus'])
