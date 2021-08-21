import sys
sys.path.append('sample')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from p211sinple_rnnlm import SimpleRnnlm
# ハイパーパラメータの設定
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100
# 学習データの読み込み（データセットを小さくする）
corpus, word_to_id, id_to_word = ptb.load_data('train')
print(corpus.shape)
print(type(word_to_id))
print(type(id_to_word))
[print('%d:%s'%(i,id_to_word[i])) for i in range(20)]
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size= int(max(corpus)+1)
xs = corpus[:-1] # corpusの最後の文字を取り除いた文字列
ts = corpus[1:]  # corpusの最初の文字を取り除いた文字列
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))
print('corpus.shape:',corpus.shape)
# 学習時に使用する変数
max_iters = data_size // (batch_size * time_size)
print('max_iters:',max_iters)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []
exit()
# モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
# ミニバッチの各サンプルの読み込み開始位置を計算
jump = (corpus_size -1 ) // batch_size
offsets = [i * jump for i in range(batch_size)]
print('offsets:',offsets)
for epoch in range(max_epoch):
    for iter in range(max_iters):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                # print('i:',i,' t:',t,' offset:',offset,' time_idx:',time_idx,' data_size:',data_size,' (offset + time_idx) % data_size',(offset + time_idx) % data_size)
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = xs[(offset + time_idx) % data_size]
            time_idx += 1
        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1,ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0