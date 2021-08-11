import sys
sys.path.append('.\sample')
sys.path.append('.\sample\common')
import numpy as np
from p040SGD import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from p043two_layer_net import TwoLayerNet
from pprint import pprint
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size // batch_size
print("max_iters:",max_iters)
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size) pu
    x = x[idx]
    t = t[idx]
    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        # 損失計算
        loss = model.forward(batch_x, batch_t)
        # 勾配取得
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
        if loss_count > 11:
            exit() 
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
x, t = spiral.load_data()
CLS_NUM = 3
N = 100
markers = ['o', 'x', '^']
plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(loss_list)),loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N,0], x[i*N:(i+1)*N,1], s=60, marker=markers[i])
plt.show()
