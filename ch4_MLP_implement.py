"""
目标：实现多层感知机的实现
1. LR是多输入单输出，softmax是多输入多输出且只有一层，而多层感知机自然是有多层，目的是实现非线性，那是怎么实现这类非线性的呢？主要是添加隐藏层，而只添加隐藏层，实际是另一种所谓的线性，故而还要增加激活函数
2. 因此多层感知机的实现相对简单许多。而本章主要的难点是怎么做到后面的多项式部分
2023.7.18 created by G.tj
"""
import torch
from torch import nn
from torch.utils import data
from IPython import display # 这个模块有空还是需要搞明白一下是干嘛的
import torchvision
import numpy as np
import os
from d2l import torch as d2l
import ch3_LinearRegression as LR # 即使l...py中包含某个具体模块，而在使用中仍然需要再次导入，不同于c++，include一次就行了，无需重复
import ch3_softmax_implement as sm
import math
# 多层感知机从零开始实现
num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 单元数
# 定义relu激活函数
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a) # 能对矩阵进行逐个比较

# 定义模型
def net(x):
    x = x.reshape(-1, num_inputs)
    H = relu(x@w1+b1) #矩阵乘法，使用torch.mm()与@有什么区别呢？
    return (H@w2+b2)

# w, b权重初始化，且在net内进行初始化，也就是更改net中权重的值
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 对模型进行训练和评估
def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2) #创建2个容器，存储损失的总合，样本数量
    for x, y in data_iter:
        out = net(x)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 可以不设置偏置，全连接模型的输入就等于特征维大小
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, train_labels.reshape(-1, 1)), batch_size, is_train=False)

    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test']) #绘制进度图
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch==0 or (epoch+1)%20==0:
            animator.add(epoch+1, (evaluate_loss(net, train_iter, loss),
                                    evaluate_loss(net, test_iter, loss)))
    print('weight', net[0].weight.data.numpy())


# 实现L2正则化，也就是权重衰减
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True) #本处还是人为按照维度设定大小
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train_L2(lambd):
    w, b = init_params()
    net, loss = lambda X:d2l.linreg(X, w, b), d2l.squared_loss #这个lambda总是忘记， lambda args:expresses其中args是参数，相当于函数的传参
    num_epochs = 500
    lr = 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item()) # torch.norm默认就是求L2范数


# L2正则的简洁实现
def train_concise(wd): # wd表示weight_decay
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # print(type(net.parameters()))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([{"params": net[0].weight, 'weight_decay': wd},{"params": net[0].bias}], lr=lr) # 其中设置的是参数。但是为什么使用net.parameters()的时候效果很差
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            l = loss(net(x), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1, (
                d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)
            ))
    print('w的L2范数是：', net[0].weight.norm().item())  # torch.norm默认就是求L2范数

# dropout是专门加的一个层,当输入进去后，改变输出值
def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1
    # 输入均被丢弃，于是输出为0
    if dropout == 1:
        return torch.zeros_like(x)
    # 输入均被保留，于是为原输出
    if dropout == 0:
        return x
    mask = (torch.rand(x.shape) > dropout).float() #[0,1)的均匀分布中抽取一组数据，数据维度给定
    return mask * x / (1.0-dropout) #给非置0者进行如下的如此的变更

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        super(Net, self).__init__()
if __name__ == "__main__":
    # 1--------------线性激活函数的使用测试===》relu是针对每个向量x中的每个元素进行的因此改变的是输出元素的值，并不改变其维度。深度学习中的维度信息需要格外的注意
    # x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) #取数值范围为-8.0到8.0的左闭右开区间，其中两个数字之间的间隔为0.1
    # y = torch.relu(x)
    # print(x.shape)
    # print(torch.ones_like(x).shape) # ones_like(x)生成与x的形状完全相同的张量
    # y2 = torch.sigmoid(x)
    # # print(y)
    # # d2l.plot(x.detach(), y2.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5)) # x.detach得到的数据其实和原x一样，只是不需要保存其梯度，以后也不计算梯度
    # y2.backward(torch.ones_like(x), retain_graph=True)
    # d2l.plot(x.detach(), x.grad, 'x', 'sigmoid偏导', figsize=(5, 2.5))
    # d2l.plt.show()
    # y3 = torch.tanh(x)
    print("---------------------------------多层感知机从零开始实现----------------------------")
    # 准备数据
    # batch_size, num_epochs, lr = 256, 10, 0.1
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 初始化参数:看隐藏层个数，隐藏层有多少个，也就是有多少个全连接层，则有多少个w,w的维度为特征数*输出个数
    # w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    # w1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True) #这种和上一种的结果是一致的
    # b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad= True))
    # w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
    # b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    # params = [w1, b1, w2, b2]
    # print(w1)
    # print(w1.shape)
    # 设置损失函数：使用交叉熵
    # loss = nn.CrossEntropyLoss(reduction='none')

    # 模型进行训练
    # updater = torch.optim.SGD(params, lr) # softmax中使用net.parameters()。该段部分其实是把参数写进去即可，而本部分的net并不是使用seq中的nn.Module加载的，因此无法使用该语句
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # d2l.plt.show()

    print("------------------------------多层感知机的简洁实现-------------------------------")
    #网络模型的定义
    # net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    # net.apply(init_weights)
    # loss = nn.CrossEntropyLoss(reduction='none')
    # updater = torch.optim.SGD(net.parameters(), lr)
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # d2l.plt.show()
    """
    多层感知机的总结：
    1. 有了前面的铺垫：实现向量以及矩阵的线性乘法，知道怎么进行初始化模型参数，知道如何进行损失函数的创建（MSE以及CrossEntropy）,
    知道如何创建数据集（自己生成的数据集做成迭代器格式，官方数据集如何使用，dataset以及dataloader还需要深入了解），知道如何进行自动微分，如何进行参数更新（updater.step()），
    如何求的训练loss并依据此进行模型的判定。有了这些积累后，该部分的内容就显得十分简单了。
    2. 该部分重点是增加了几层线性知道如何的创建（手动从0实现是怎么做的，运用官方的nn.Sequential()是如何做堆叠的）
    3. 重点明白三个激活函数的使用及其优劣：sigmoid容易引起梯度消失问题（梯度的极限）而被relu所取代。relu的优点包括：函数计算简单，求导方便，不存在sigmoid带来的梯度消失问题；
    而sigmoid使用较早，主要优点是因为其函数平滑且处处可微；而tanhx也比较平滑且处处可微也不存在梯度消失问题，但是计算相对复杂
    还要重点学习的：
    1. 关于w, b初始化的几种方法
    2. 代码优雅简洁的书写方式
    """
    print("过拟合以及欠拟合")
    # 解决过拟合问题：暂退法（dropout）以及权重衰减（L2正则化）
    print("---------------------------多项式回归---------------------------------")
    # 关于np.power的实现
    # 1. 实现简单的幂方运算
    # print("单个幂次方运算", np.power(2, 3)) # 2的三次方
    # print("包含列表幂次方运算", np.power([2, 4], 2)) # 第一项可以是可迭代对象：列表，元组
    # print("幂次方可以列表", np.power([2, 4, 5], [2, 3, 2])) # 第一项可以是可迭代对象：列表，元组,第二项如果也要是可迭代对象，则需要长度对应，要么就一个数
    # print("如果设置第一个参数为矩阵呢？", np.power([[2], [3]], [2, 3, 4])) #两者要么一一对应的长度，要么第一个或者第二个为一维
    # 多项式的计算
    # max_degree = 20 #设置最大阶数
    # n_train, n_test = 100, 100 # 取样本数为200个，因此x的维度为[200, 阶数]，因此取数的时候，两百个样本，只需要取一个x值
    # true_w = np.zeros(max_degree) # 权重的长度自然和特征数数一样, 1也当做一个x
    # true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 维度等于x的特征维，本处也就是多项式的项数
    # x = np.random.normal(0, 0.1, size=(n_test+n_train, 1))# 维度[200, 1]
    # np.random.shuffle(x)
    # print(np.arange(20))
    # print(x.shape)
    # y = np.power(x, np.arange(max_degree).reshape(1, -1))# 维度[200, 20], 后面还需要维度的相加，本处是x，并不是y，我这样命名是有歧义的
    # print(y.shape)
    # print(true_w)
    # # # 对这些数据乘以相关的参数
    # for i in range(max_degree):
    #     y[:,i] /= math.gamma(i+1) # 表示阶乘gamma(n+1)=n!
    # labels = np.dot(y, true_w) # 维度为[200, 1]
    # labels += np.random.normal(0, 0.1, size=labels.shape) # 添加
    # # y = np.power(x, 维度)
    # # print(math.gamma(4)) # math.gamma(n+1)=n!
    # # 数据转换成tensor类型
    # print(x.shape)
    # print(y.shape)
    # print(true_w.shape)
    # print(labels.shape)
    # true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, x, y, labels]]
    # a = [2, 3, 4]
    # b = [[5, 6, 7], [8, 9, 10]]
    # for i in [a, b]:
    #     print(torch.tensor(i))实际上第一个时间只做了恢复性的训
    # # 对模型进行训练和评估, 使用前4项进行运算
    # # train(poly_features[:n_train, :4], poly_features[n_test:, :4], labels[:n_train], labels[n_train:], num_epochs=400)
    # # train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:], num_epochs=400)
    # train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=400) # 容易发生过拟合，因为数据其实还是4项多项式得到的，但是实际上的w有20维
    # d2l.plt.show()
    print("------------------------L2正则化测试-------------------------")
    """
    接下来需要做权重衰减：
    何为权重衰减：就是L2正则化。也就是在损失中添加一个惩罚项：这项是权重的L2范数，而常用的并不是直接的L2范数，却是L2范数的平方
    用L2范数的原因：
    """
    # n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5 # 数据量太少的时候可能产生过拟合，此时其实可以提高数据量或者加大epoch,而增加epoch有限，容易很快到达临界点
    # true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    # train_data = d2l.synthetic_data(true_w, true_b, n_train)
    # train_iter = d2l.load_array(train_data, batch_size)
    # test_data = d2l.synthetic_data(true_w, true_b, n_train)
    # test_iter = d2l.load_array(test_data, batch_size, is_train=False)
    # # train_L2(lambd=3) # 不用正则化。明显的过拟合，而使用正则化后，很快就能收敛
    # # 简单版
    # train_concise(3)
    # d2l.plt.show()
    print("------------------------------dropout暂退法测试------------------------------")
    """
    何为暂退法：就是一些神经元置0。真的忘了啊:按概率置0
    """
    # 从零开始
    print((torch.rand((2, 3))>0.5).float())
    x = torch.arange(16, dtype = torch.float32).reshape((2, 8))
    print(x)
    print(dropout_layer(x, 0)) # 相当于不尽兴该步操作
    print(dropout_layer(x, 1)) # 相当于全部处理完
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5


