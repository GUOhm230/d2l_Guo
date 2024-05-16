"""
本部分的重点就是多GPU运行模型
1. 将数据分发到两个GPU进行计算
2023.10.6 created by G.tj
"""
import torch
from torch import nn
from d2l import torch as d2l

# 目的是将数据进行相加，然后广播到所有的device中。
def allreduce(data):
    for i in range(1, len(data)): # 将所有行的数据添加到第一行中。如果超过二维，一般第一维是batch_size，后面的数据维相加然后广播
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)


if __name__ == "__main__":
    print("-----------------------------数据同步allreduce之向量相加------------------------")
    devices = d2l.try_gpu()
    # devices = ['cuda:0', 'cpu']
    devices = [torch.device('cuda:0'), torch.device('cpu')]
    print(devices)
    data = [torch.ones((1, 2), device=devices[i]) * (i+1) for i in range(2)]
    print("allreduce之前：", "\n", data[0], "\n", data[1])
    allreduce(data)
    print("allreduce之前：", "\n", data[0], "\n", data[1])
    print(len(data))
    print(list(range(1, len(data))))
    print(data[0][:])
    X = torch.ones((6, 2, 3))
    # print(X[0])
    print(len(torch.ones(4)))
    # print(X)
    # print(len(X))
    # 综上所述：len(X)表示X的最外层有多少个元素（嵌套算一个：如果一维张量则为元素数量， 如果是多维则为行数）
    print("-------------------------数据分发-------------------------------")
    data = torch.arange(20).reshape(4, 5)
    split = nn.parallel.scatter(data, devices)
    print(data)
    print(split)
    # print(data)