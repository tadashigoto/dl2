import numpy as np
def create_contexts_target(corpus, window_size):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            # print('idx:', idx, ' t:', t, ' -window_size:', -window_size, ' window_size+1:', window_size+1)
            if t == 0:
                continue
            cs.append(corpus[idx + t])
            # print('cs:', cs)

        contexts.append(cs)
    return np.array(contexts), np.array(target)
