"""
卷积神经网络
实现目标：
1. 实现卷积运算
2. 实现卷积层
3. 怎么自定义卷积网络
4. 一些经典卷积网络的从零实现
5. 复习先前的训练代码
2023.8.21 created by G.tj
"""
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 1. 如何做互相关运算的呢
# 运算方式：是对应元素的乘积
# 运算位置：Y[i, j] = X[i:i+h, j:j+w] * K
# 注意：因为X可能边缘无法应用到。循环X进行运算麻烦较多。可以先准确计算Y的维度，根据Y的维度查找X的维度。这算是算法的一个思路：就是X的每个元素不一定对应Y中的元素。，但是Y中的元素一定与X的元素对应
# 可以用归纳演绎法：第一个，第i个以及第n个
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1) # 单输入通道，单输出通道时
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 如何定义卷积层
# 1.继承nn.Module模块 2.实现前向传播函数：卷积运算+bias 3.初始化参数
class Conv2D(nn.Module):
    # 进行权重初始化
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1)) # 其维度等于输出通道数

    # 实现前向传播函数
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

# 卷积运算后输出Y的信息
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

# 多输入通道：就是输入通道的所有元素求和
# 多输入通道中X：C1*H*W k: C1*H*W。X,K都是对应的通道维进行运算
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

# 多输出通道：输出通道需要人为设定。对应X：C*H*W。对应的K:CO*C*H*W。两个的H，W不一致，C是相同的
# 运算形式是X对每个输出通道维
# 这个结果是没错的
def my_corr2d_multi_in_out(X, K):
    # 运算形式
    Cout, Cin, KH, KW = K.shape
    Y = torch.zeros(Cout, X.shape[1]-KH+1, X.shape[2]-KW+1)
    for i in range(Cout):
        Y[i] = corr2d_multi_in(X, K[i])
    return Y

def corr2d_multi_in_out(X, K):
    # Y = torch.empty()
    # for k in K:
    #     print("本轮K：", K)
    #     temp = corr2d_multi_in(X, k)
    #     print("结果获取：", corr2d_multi_in(X, k)) # 也是符合的呀，那为什么无法进行torch.stack()
    #     print("维度：", temp.shape)
    # print("到底是啥：", torch.stack(list(corr2d_multi_in(X, k) for k in K), dim=0))
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0) # 逐个计算完后，按照行数堆叠即可啊
    # (corr2d_multi_in(X, k) for k in K)得到的就是一个生成器。此时前面要加list或者tuple也就是list((corr2d_multi_in(X, k) for k in K))或者是tuple((corr2d_multi_in(X, k) for k in K))


# 实现1*1卷积
# 何为1*1卷积：实际计算和其他核卷积类似。只是没有感受野，相当于对通道维进行线性运算
# 1*1卷积主要用于改变输出通道维度，或者根据步幅，填充使得形状加倍或者减半
# 这个写法确实比较巧妙
def corr2d_multi_in_out_1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


# 自己完成一次1*1一维卷积，为什么人家能写的这么巧妙呢？自己完全用了个死办法计算
def corr2d_multi_in_out_1_my(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    Y = torch.zeros(size=(c_o, h, w))
    for i in range(c_o):
        for j in range(h):
            for p in range(w):
        # Y每个通道是X与K中每个通道进行卷积获得
                Y[i, j, p] = torch.matmul(X[::,j, p].flatten(), K[i,::, ::, ::].flatten()) #其中哪个维度确定了，则该维度自然消失
    return Y

# 实现一下pooling运算： 假设输入是二维.如果是三维，也就是对每个通道进行同样的处理，四维也就是加上batch_size。同理：对每个样本进行这样的操作
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i+p_h, j: j+p_w].max() # 其实所在的位置以及计算
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

if __name__ == "__main__":
    print("---------------------1.互相关运算-------------------------")
    X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    K = torch.arange(4, dtype=torch.float32).reshape(2, -1)
    Y = corr2d(X, K)
    print(Y)
    print("--------------------2. 卷积运算之实现黑白边缘检测-------------")
    X = torch.ones(6, 8)
    X[:, 2:6] = 0
    print("X=", X.T==X.t()) # 两者都是转置
    K = torch.tensor([[1.0, -1.0]]) # K必须是个二维矩阵，才能进行二维卷积运算，因此不能是向量
    Y = corr2d(X, K)
    # Y = corr2d(X.t(), K.t())
    # print("边缘检测Y=", Y)
    print("----------------------3. 卷积层的实现及应用------------------")
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = X.reshape(1, 1, 6, 8) # N C H W
    Y = Y.reshape(1, 1, 6, 7) # N的维度随X而定，H W的维度随X，K而定， 而C为输出通道数：自定义
    lr = 3e-2
    print("conv2d:", conv2d.weight.grad)
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat-Y) ** 2
        # conv2d.zero_grad() # zero_grad()是对多元函数变量使用的
        # conv2d.weight.grad.zero_()
        l.sum().backward()
        # print("叶子张量：", conv2d.weight.grad.zero_())
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        conv2d.zero_grad()
        # conv2d.weight.grad.zero_() # 如果对X进行梯度，摆放位置需要合理
        if (i+1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')
    print("权重：", conv2d.weight.data.reshape(2, 1))
    """
    卷积运算的阶段总结：
    1. 卷积运算实际是互相关运算，要实现卷积运算，主要是搞清楚输入输出维度的对应关系
    2. 要实现卷积层就有三要素：1. 初始化权重参数，权重参数就是K 2.继承nn.Module 3.调用卷积运算，实现前向传播
    3. 模型参数是K，可以对该模型进行训练，则需要有3要素：1. 模型，也就是第二步定义的卷积层 2. 需要有loss=(net(X)-Y)**2 3. 需要进行参数更新，参数梯度初始化为0:net.zero_grad()
    4. 模型训练步骤依然如之前所述，但是数据并没有使用dataLoader因为只有一个数据。关于模型训练的具体步骤，其实没有固定死。我需要的是把X送进去运算，然后得到的Y进行loss计算
    """
    print("------------------------4. 卷积的复杂化：多输入，多输出---------------------------------")
    # 填充与步幅
    # 这里要做的事其实就是填充和步幅怎么影响输出维度,，并在官方api中的应用
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1)) # padding=(2, 1)指的是X左右两边各填充了2个，上下两边各填充1个
    X = torch.rand(size=(8, 8), dtype=torch.float32)
    print("X.shape=", X.shape)
    # X = X.reshape((1, 1) + X.shape) # 加号是连接符，不是计算符
    print(X.shape)
    # print([1, 2] + [3, 4]) # [1, 2, 3, 4]
    print(comp_conv2d(conv2d, X).shape)
    # 添加步幅
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)
    # 多输入通道的卷积运算
    print("=====zip的使用=====")
    a = [1, 2, 3]
    b = [-1, -2, -3]
    # zip是对其中的元素按照对应的位置打包成元组
    # *zip则可以对其中的元素进行
    # print(zip(*zip(a, b)))
    for i, j in zip(a, b):
        print(i, "\t", j)
    X = torch.arange(9, dtype=torch.float32).reshape(3, -1)
    K = torch.arange(4, dtype=torch.float32).reshape(2, -1)
    X = torch.stack((X, X+1), dim=0)
    K = torch.stack((K, K+1), dim=0)
    Y = corr2d_multi_in(X, K)
    print(Y)
    # 多输入，多输出通道的测试
    K = torch.stack((K, K+1, K+2), dim=0)
    print(my_corr2d_multi_in_out(X, K))
    # print(K.shape)
    print("------------------------------5.1*1卷积---------------------------------")
    X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    K = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1)
    Y1 = corr2d_multi_in_out_1(X, K)
    Y2 = corr2d_multi_in_out_1_my(X, K)
    print("使用书本提供的方法获得一维卷积：", Y1)
    print("使用自定义方法获得一维卷积：", Y2)
    # print("假如创建空的tensor:", torch.empty(size=(2,3)))
    Y3 = corr2d_multi_in_out(X, K)
    print("通用多维卷积：", Y3)
    print("两种结果进行比较：", Y1==Y2)
    print("两种结果进行比较：", Y1==Y3)
    # 总结：自己实现的方法和官方实现的方法结果一样，均和之前的通用方法效果一致。其实不必专门实现一个1*1卷积
    # 事实证明，只要愿意好好做，就能做好这件事
    # 实现一下书本上的例子，待会扩充至三维数据
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    print(Y1)
    print(Y2)
    print(Y1==Y2)
    # Y[i, j, p] = torch.matmul(X[::, j, p], K[i, ::, ::, ::])
    # print(X[::, 1, 1])
    # print(K[1, ::, ::, ::].flatten())
    # print(torch.matmul(X[::, 1, 1], K[1, ::, ::, ::].flatten()))
    print("-----------------------------------6. 汇聚层--------------------------")
    # 汇聚层与卷积层有着同样的填充，步幅，步长，宽高的计算遵从同样的公式
    # 但是，汇聚层不同于卷积层的一个特征是，汇聚层不是当前通道与对应维K进行互相关运算后对通道维进行求和。
    # 汇聚层在处理多通道输入时，是在单个通道上单独运算。因此X：C * H1 * W1，K：C * H2 * W2。而输出通道维也是C。注意这里我犯错的地方：K并不是一个多维矩阵，而是指定一种操作, 指定其size, 么有C
    X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    Y = pool2d(X, (2, 2), 'avg')
    print(Y)
    # 看看如何调用官方运算
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)
    pool2d = nn.MaxPool2d(3) # 官方也是：指定操作以及维度即可。后面其实还要涉及步幅和填充
    Y1 = pool2d(X)
    print(Y1)
    # 手动调整一下填充和步幅
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    Y3 = pool2d(X)
    print(Y3)
    # 其中维度以及步幅和填充都可以左右与上下不一致的
    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    Y4 = pool2d(X)
    print(Y4)
    # 如果X是多个通道呢
    X = torch.cat((X, X+1), 1) #
    # X = torch.stack((X, X+1), dim=0) # stack会增加维度。而cat则会加在外面。具体还需要学习一下，并做好笔记
    print(X)
    print(X.shape)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    Y5 = pool2d(X)
    print(Y5)
    """
    卷积阶段性总结2--卷积基础知识总结：
    1. 在卷积层中的卷积运算实际上是互相关运算
    2. 卷积层也就是添加几大要素：1）继承nn.Module 2）初始化参数（使用nn.Parameter(size=())），不仅是对参数的初始化，更重要的是指明谁是参数，设置该参数为叶子张量，
    加入nn.Module中作为需要求自动微分的参数，这样的话，在设定loss之后，调用自autograd可以求梯度以更新参数 3）实现前向计算互相关运算+偏置
    3. 多输入通道的卷积：X中每个通道与对应的K进行卷积运算，然后通道维相加，最终得到的输出是一个通道
    4. 多输入多输出通道的卷积：此时的K是四维，X与每个K[i]进行卷积运算（3中的多输入通道的卷积运算），这样计算c_o次，得到输出为c_o通道
    5. torch.stack()以及torch.cat的使用需要学习，并做好笔记。而且，通过该方法进行张量的创建好像更加方便
    6. pooling层（池化或者汇聚层）的计算与卷积层类似，但是并不用通道维的累加，而是通道单独运算
    7. 卷积调用官方API: nn.Conv2d(输入维度， 输出维度，Kernel_Size, padding, stride)。卷积需要指明输出维度，pooling不用，因为pooling的输出维度=输入维度
    8. pooling的官方API: nn.MaxPool2d(kernel_size, padding, stride)即可。
    """


