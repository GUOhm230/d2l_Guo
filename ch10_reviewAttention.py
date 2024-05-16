"""
本部分为注意力机制的复习版
2023.12.13 Created by G.tj
"""
# 问题1：softmax在本阶段的使用。为什么可以把当当做softmax进行计算书写：一个猜测，f(x)中的x是针对所有的x,而不是单个. 实验证明，证实猜想
import torch
from torch import nn
from d2l import torch as d2l

# 生成x,y数据对
n_train = 50 #训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return x * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(x_train)
print(y_train.shape)
x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train)) # [50, 50]
print("(x_repeat - x_train)**2 / 2.shape=", ((x_repeat - x_train)**2 / 2).shape) # [50, 50]
attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1) # [50, 50], 测试数据
print(attention_weights.shape)
y_hat = torch.matmul(attention_weights, y_train)
# 有关softmax需要自己动手实现一下
def softmax(x):
    return torch.exp(x)/torch.exp(x).sum()
st = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print(st)
st_result = nn.functional.softmax(st, dim=1) # 默认为列，按列求，则列消失，就是针对每列
print(st_result)
print(softmax(torch.tensor([1., 2. ,3.])))
print(softmax(torch.tensor([4., 5. ,6.]))) # 法则其实就是等差数列的softmax是相同的
# 所以总结一下：softmax就是一个序列数据，然后针对一个维度的一行或者一列数据进行求解
