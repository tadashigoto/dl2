# p114convert_contexts_to_onehot.py : contextsをone-hot表現に変換する
import numpy as np
import sys
from p064preprocess import preprocess
from p072create_co_matrix import create_co_matrix
from p113create_contexts_target import create_contexts_target
sys.path.append('sample')
from common.util import convert_one_hot
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(target)
print(contexts)