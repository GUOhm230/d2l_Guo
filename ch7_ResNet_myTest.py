"""
由于ResNet网络过于经典，且常用，因此专门拿出来
该网络准备自己写出来，正好是对前面模型的检验
2023.8.27 created by G.tj
"""
# ResNet网络还没复习呀
# BN层的复习不是很清晰到现在为止，我知道哪些要训练，却不知道应该怎么做
# 定义残差块，就是怎么实现f(x)-x
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module): #@save
    def __init__(self, input_channels, num_channels, use_1dConv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1dConv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) # 如果卷积的时候改变了其通道，则需要通过1*1卷积以改变X的通道数，使加法得以匹配
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels) # 这个地方为什么要提供他的输出通道呢，BN层还要深入
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X # 真就是简单的矩阵加法啊。或者做一遍1*1卷积，只改变宽高，并不改变通道数
        return F.relu(Y)

# resnet网络块，其中包含几个残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1dConv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 学习一下稠密网络吧
# 1. 定义卷积块, 先使用了BN
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels): # 2, 3, 10
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels)) # 0+3, 10+3=13

        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 20+3=23
        return X

# 定义过渡层
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

if __name__ == "__main__":
    blk = Residual(3, 4, use_1dConv=True)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
    # 定义一个ResNet模型
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # 从此开始通道增加，高宽减半
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    # 先几个卷积，然后就是连续的4个卷积块
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)), # (1, 512, 1, 1)
                        nn.Flatten(), nn.Linear(512, 10))

    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 模型训练：
    # lr, num_epochs, batch_size = 0.05, 10, 256
    # device = d2l.try_gpu()
    # train_iter, test_iter=d2l.load_data_fashion_mnist(batch_size, resize=96)
    # d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    """
    resnet网络总结：
    1.好像没什么需要总结的
    2.关键点就是如何创建残差实现fx-x。这个实际就是在之后卷积之后做一个Y+=X的操作，如果卷积的时候改变了其通道数和宽高，就要使用1x1卷积以改变其通道数和宽高使得可以正常运算
    3.resnet网络是有块组成，块中嵌套了残差块，而残差块又包含着各种卷积和池化
    4.同样减少了FC全连接的使用，而使用的是全局平均池化层以减少维度
    5.接下来逐层分析下resnet网络模型: 完成在A4纸上
    6.DenseNet
    """
    # 稠密块
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    num_channels, growth_rate=64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels//2))
            num_channels = num_channels//2

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    )
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # DenseNet的一点就是在卷积层前加了BN和ReLU