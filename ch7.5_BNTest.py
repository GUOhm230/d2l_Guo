"""
主要进行BN层的从零实现
该部分目标：
1. 关于mean(dim=(0, 2, 3))是如何完成计算的，是否如猜测：即所有样本中相同通道维中所有的元素求得均值: 证实如猜测所言
2. 关于gamma*X中的gamma维度=特征维（全连接为特征维d，卷积层为通道维度）
created by G.tj 2023.8.9
"""
import torch
from torch import nn
from d2l import torch as d2l

"""
开始实现batch——norm操作
由于批量规范化是对整个批量样本进行的特征维度上的均值和方差计算
1. 预测时，整个测试样本的均值和方差是确定的
2. 训练时，用批量B的均值和方差去预估整体的均值，方差
针对全连接层和卷积层的操作略有不同
使用BN也可以防止过拟合，主要是加快收敛。因此一般现在卷积后用BN层，而不用nn.Dropout(p=)
"""

# 本来输入的是X，输出的是同维度的Y。现在为什么要有moving_mean和moving_var呢？用于估计全局的均值和方差吗？
# 定义批量规范化：只要完成批量规范化操作即可
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 预测状态
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var)
    # 训练模式下
    else:
        assert len(X.shape) in (2, 4)
        # 全连接层
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        # 卷积层
        elif len(X.shape) == 4:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X-mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0-momentum) * mean
        moving_var = momentum * moving_var + (1.0-momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


# 定义批量规范化层：不仅要完成操作，还要加入nn.Module中，作为Sequential列表中的内容
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape) # 作为分母，不能太小

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to( X.device)
            self.moving_var = self.moving_var.to( X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y


if __name__ == "__main__":
    X = torch.arange(96, dtype=torch.float).reshape(2, 3, 4, 4)
    print(X)
    Y = X.mean(dim=(0, 2, 3))
    print((X[0,0,::,::].sum() + X[1,0,::,::].sum())/32.)
    print(X[0,1,::,::].sum() + X[1,1,::,::].sum())
    print(X[0,2,::,::].sum() + X[1,2,::,::].sum())
    print(Y)
    # print("---------------------------求和--------------------------------")
    # print(Y)
    # print(X[0][0].sum() + X[1][0].sum())
    # print(X[0][1].sum() + X[1][1].sum())
    # print(X[0][2].sum() + X[1][2].sum())
    # print("--------------------------通道维求均值------------------------------")
    # Z = X.mean(dim=(0, 2, 3))
    # print(Z)
    # print((X[0][0].mean() + X[1][0].mean())/2)
    # print((X[0][1].mean() + X[1][1].mean())/2)
    # print((X[0][2].mean() + X[1][2].mean())/2)
    # print("----------------------------gamma计算--------------------------------")
    # gamma = torch.arange(1, 4, 1, dtype=torch.float).reshape(1, 3, 1, 1)
    # # gamma1 = torch.ones((1, 3, 1, 1))
    # print(gamma)
    # V = gamma * X
    # print(V)
    # M = torch.arange(12).reshape(3, 4)
    # print("--------------------------------批量规范化层的应用-----------------------------------")
    # # LeNet
    # net = nn.Sequential(
    #     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2), nn.Flatten(),
    #     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(84, 10)
    # )
    # print(net)
    # # 设定超参数
    # lr, num_epochs, batch_size = 1.0, 10, 256
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # # for x, y in train_iter:
    # #     print(x, y)
    # # d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # # d2l.plt.show()
    # """
    # BN层的总结：
    # 1. BN的诞生是因为想训练深层网络十分困难，要想在较短时间让其收敛.还能起到正则化的作用。至于解释，没有信服的解释
    # 2. BN层在训练时的计算与测试时的计算不一样，主要是体现在测试时样本固定，因此可以算出整体均值与方差。而训练时只针对B个样本而言，故而只能根据小批量样本以估算整体均值和方差
    # 3. 训练时，计算当前批量的均值和方差： 是对特征维进行：卷积层是对通道维进行计算（X.mean(dim=(0, 2, 3))）.全连接层是对特征维进行计算(X.mean(dim=0))
    # 4. BN(X)中添加的gamma和beta都是需要经过训练的，至于如何进行反向传播：其实就是该层加入nn.Module中。构建一个loss,loss和y的预测值以及标签值有关，而预测值是通过前向传播得到
    # 前向传播自然会判定谁是变量？不，是根据是否进行require_grad=True的声明。声明之后，还要确定其为叶张量
    # 基于以下的实验，暂时结束本次测试，得出以下结论：
    # 1. 由于BN层的计算中gamma和beta是叶子节点，且设置其require_grad=True.因此可以在loss中计算梯度，且保留下来
    # 2. 设置require_grad=False的为叶子节点，将非叶子节点剥离成叶子节点用X.detach()
    # 3. 所谓叶子节点，其实就是用户定义，不是中间变量产生，在计算图中为叶子节点者
    # 4. 但是多维矩阵的乘积时，怎么有的时候又是叶子节点，有的时候又不是。使用torch.normal的时候为叶子张量，而使用torch.ones的时候却不是
    # 因此本次其实解决了之前一直的疑问：就是在前向传播之后，怎么做到反向传播，并求出其梯度.其实就是在定义的时候设置require_
    # """
    # print("1. 测试一下自动微分: 一维向量时可以求得")
    # u = torch.arange(4, dtype=torch.float, requires_grad=True)
    # v = torch.ones(4, dtype=torch.float, requires_grad=True)
    # z = torch.arange(-3, 1, 1, dtype=torch.float)
    # print(z)
    # Y = u * v * z
    # print(Y.sum())
    # Y.sum().backward()
    # print("对u求偏导", u.grad==v*z)
    # print(u.is_leaf) # u,v,z为非叶张量，无法进行求导。那如何设置为叶张量呢？让他知道需要更新参数.叶子张量并不是通过设计而得。而是该计算确定下来就存在的。也就是计算图的叶子节点为叶张量
    # print("对v求偏导", v.grad==u*z, u*z, v.grad)
    # print("对z求偏导", z.grad) # 然而z处没有设定偏导计算，故为none
    # print(Y.grad)
    # # x = torch.
    # # Y = x*x
    # print("2. 继续测试自动微分， 当输入向量设置二维")
    # x = torch.arange(12, dtype=torch.float).reshape(2, 6)
    # # z = torch.normal(0, 0.1, size=(6, 2), dtype=torch.float, requires_grad=True)
    # z1 = torch.ones(12, dtype=torch.float).reshape(6, 2)
    # z1.requires_grad_(True)
    # print(z)
    # print(z1)
    # Y = torch.matmul(x, z1)
    # print("Y=", Y)
    # Y.sum().backward() #因此不存在x是二维的，Y也要对应维度才能计算，Y必须是标量
    # print("Y.sum=", Y.sum(dim=1, keepdim=True))
    # print("x的梯度：", x.grad)
    # print("判定x,z1是否为叶子张量", x.is_leaf, z1.is_leaf)
    # print("z1的梯度：", z1.grad)
    # """
    # 反思：但是当计算多输出的线性回归时，W也是多维矩阵，不是一样可以计算吗？而此时的Y也是多维的.为什么此处偏偏不行？
    # """
    # print("3. 继续测试梯度计算， 模拟多输出的线性回归")
    # X = torch.arange(12, dtype=torch.float).reshape(3, 4)
    # W = torch.normal(0, 0.01, size=(4, 3), requires_grad=True)
    # # b = torch.ones(3, dtype=torch.float, requires_grad=True)
    # Y = torch.matmul(X, W)
    # Y.sum().backward()
    # print("W是否为叶子张量：", W.is_leaf)
    # print("W的梯度：", W.grad)
    # # print(b)
    # # print(X)
    # # print(W)
    # # print(Y)
    # # x = torch.arange(12).reshape(3, 4)
    # # y = torch.arange(12).reshape(4, 3)
    # # print(x, y)
    # # print("二维矩阵之间的matmul：", torch.matmul(x, y))  # 二维矩阵乘法，要满足数学规定 3*4 4*3
    # # print(torch.mm(x, y))
    # # print(x@y)