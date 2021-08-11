import numpy as np
import sys
sys.path.append('sample\ch04')
from p157unigram_sampler import UnigramSampler

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2
sampler = UnigramSampler(corpus, power,sample_size)
targe =np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(targe)
print(negative_sample)