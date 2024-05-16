"""
李沐深度学习课程中的内容都要自己手动编写，一年工程
创建目的：从零开始实现线性回归,方便以后直接调用
需要解决的问题：
1. 手动创建数据，主要关注数据维度
2. 手写优化器，主要关注梯度下降以及反向传播的途径
3. 手写train函数，主要包括数据流的走向以及损失函数的书写，以及必要组件
4.
2023.7.12 created by G.tj
===================================
线性回归方程：y = w1x1+w2x2+w3x3+......w4x4+b
"""
import torch
import os
import numpy as np
import random
from torch.utils import data
from torch import nn
from d2l import torch as d2l
# 数据准备, 即合成相关数据
def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w))) # 生成n*d的矩阵，该矩阵的数据符合正态分布
    y = torch.matmul(x, w) + b # tensor矩阵的乘积
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

# 数据预处理test
def dataProcessTest():
    means = torch.arange(1, 11).float()  # 报错RuntimeError: “normal_kernel_cpu“ not implemented for ‘Long‘。是数据类型的问题
    std = torch.arange(1, 0, -0.1).float()
    print(len(means))
    print(len(std))
    a = torch.normal(0, 1, (10, 2))
    b = torch.normal(means, std)
    # print(b)
    # torch.matmul()的计算,相当于矩阵运算，但是维度不同则可以进行自动甄别，并进行自动补全
    c = torch.ones(2, 4, 2)
    d = torch.ones(2, 2, 3)
    e = torch.matmul(c, d)
    list1 = torch.Tensor([1, 2, 3])
    list2 = torch.Tensor([4, 5, 6])
    print(list1)
    print(list2)
    print(list1 * list2)
    print(torch.matmul(list1, list2))


# 数据读取，就是要把这些数据做成迭代数据,可迭代对象
# 其实目的就是处理了这些数据，但是这些数据其实可以用列表或者元组，之所以要用生成器，是因为生成器在生成和调用数据的时候更加快捷
# 当需要关注使用性能的时候，应该首先想到使用将列表换成迭代器。而迭代器的首选则是yield关键字
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 原地打乱list顺序, 两者地址是一样的
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 测试yield关键字
def yieldTest():
    ll = 100
    for i in range(10):
        print("---------------yieldTest测试----------------")
        yield i


# 定义模型,所谓模型，这里简单的包括以后复杂扩展的模型都是指从输入到输出处理的一系列表达式流程，而
def linreg(x, w, b):
    return torch.matmul(x, w) + b

# 定义损失函数.损失函数就是y的预测值与y的真实值之间的差距
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 /2

# 定义优化算法，其中实现当前参数w,b的反向传播即如何更新参数
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params: #其中的参数有w有b针对w而言
            # w以及b的更新策略：w = w-lr/B*loss对于w的偏导，也就是所说的梯度，而该梯度就是param.grad。存在该梯度的时候必须先做loss.sum.backward()
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 梯度清零

# 从零开始实现线性回归代码的缩写
def train(batch_size, n, d):
    # 1.数据合成
    true_w = torch.tensor([2, -3.4]).reshape(d, 1)
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, n)
    # 2. 初始化模型参数
    w = torch.normal(0, 0.01, size=(d, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 3. 构建模型
    net = linreg

    # 4. 创建优化器，前面sgd已经完成
    # 5. 创建损失函数
    loss = squared_loss
    # 6. 设置学习率，设置训练轮数等超参数
    lr = 0.03
    num_epochs = 3
    # 7. 开始训练
    # 一个epoch跑完一次所有训练数据
    for epoch in range(0, num_epochs):
        # 批量梯度下降，作为一次权重更新
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        # 训练完一次批量后，再看看当前已更新完的模型参数针对所有数据1000个的loss
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss{float(train_l.mean()): f}')


# 简单实现线性回归
def load_array(data_arrays, batch_size, is_train=True):
    #构造一个pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays) # 其中的data_array是元组(x, y)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train_linear_simplify(batch_size, n, d):
    true_w = torch.tensor([2, -3.4]).reshape(d, 1)
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, n)
    data_iter = load_array((features, labels), batch_size)
    # 这是一个迭代器,
    for x, y in data_iter:
        print(x, y)
    #定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    print(net)
    #初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0) # 所有值都填充为指定的数
    #定义损失函数
    loss = nn.MSELoss() # 本身就是个类，调用时使用loss(input)调用forward函数
    #定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            print("loss", l) #为一个数
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss{l: f}')
        print(net[0].weight.data, "\n", net[0].bias.data)


if __name__ == "__main__":
    # print(features.shape) # 1000*2
    # print(labels.shape) # 1000*1
    # 数据绘制：
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    # d2l.plt.show()
    # -----------------2.读取数据，使之成为可迭代对象
    # shuffle的测试  *
    # indices = list(range(10))
    # print(indices, id(indices))
    # random.shuffle(indices)
    # print(indices, id(indices))
    # 测试yield关键字
    # for j in yieldTest():
    #     print(j)
    # 1.-------------------------从零开始实现线性回归---------------
    train(10, 1000, 2)
    # 2.-------------------------线性回归的简洁实现
    print("线性回归的简洁实现")
    train_linear_simplify(10, 1000, 2)





