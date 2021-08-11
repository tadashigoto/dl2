import numpy as np
from p064preprocess import preprocess
from p113create_contexts_target import create_contexts_target
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)
contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)