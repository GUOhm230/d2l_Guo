"""
前7章的复习。一些关键方法和网络自己动手复习
闭卷默写
2023.8.28 Created by G.tj
总：
1. 前七章学习了深度学习中涉及矩阵的处理方式
2. 学习了如何对多元函数进行自动微分
3. 如何实现线性计算
4. 关于线性连接的权重维度
5. 如何实现模型的训练：设置损失函数，进行自动微分，进行权重更新
6. softmax如何进行计算
7. softmax中出现的上溢和下溢，当怎么解决？就是在实际处理的时候，并不对softmax进行计算，而是直接计算
8. MLP计算
9. 参数管理
10. 卷积
11. 经典卷积网络
"""
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.utils import data
import random

# 实现线性计算
def linear(X, output_shape):
    # 初始化权重参数
    x_h, x_w = X.shape
    w = torch.normal(0, 1, size=(x_w, output_shape))
    b = torch.zeros(output_shape)
    Y = torch.matmul(X, w)
    Y += b
    return Y

# 自定义的线性全连接的网络层
class myLinear(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(myLinear, self).__init__()
        self.weight = nn.Parameter(torch.normal(0, 0.1, (input_shape, output_shape)))
        self.bias = nn.Parameter(torch.zeros(output_shape)) # 不能使用torch.empty().你这个罪魁祸首，我就说怎么可能线性回归怎么可能导致梯度巨变，生成的是一个任意类型的数据，任意大小
    def forward(self, X):
        # print("X.shape=", X.shape)
        # print("torch.matmul=", torch.matmul(X, self.weight).shape)
        return torch.matmul(X, self.weight) + self.bias

def synthetic_data(w, b, num_examples): #@save
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1)) #目的是维度对应

# 十分重要的点，需要记下来
# 其实数据迭代器，就是把一个batch_size的X，Y数据打包在一起。这里其实主要是list的灵活运用，还有就是yield有关python生成器和迭代器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] # yield就是跑到这里的时候运行，接下来等下一次next

# 定义损失函数: 1/2(y_hat-y)2。这里算法其实是小事，关键是维度如何
def squared_loss(y_hat, y):
    return 1/2*(y_hat-y.reshape(y_hat.shape))**2

# 进行参数更新 param -= lr.param.grad/B
def sgd(params, lr, batch_size):
    with torch.no_grad():#该句会将param中的requires_grad设置为False。官方：self.prev = torch.is_grad_enabled() self.prev=False
        for param in params:
            param -= lr*param.grad / batch_size
            param.grad.zero_() # 参数更新完需要将梯度置0

#开始训练
def train(net, X, Y, batch_size, lr, num_epochs):
    for epoch in range(num_epochs):
        for x, y in data_iter(batch_size, X, Y):
            l = squared_loss(net(x), y)
            # print("当前权重", net[0].weight.data, net[0].bias.data)
            # print("当前轮训练数据：", x, y, net(x), l)
            # if l=="inf":
            #     print("测试输出：", l)
            l.sum().backward()
            # sgd([net.weight, net.bias], lr, batch_size)
            sgd(net.parameters(), lr, batch_size) #和上面结果一样
        with torch.no_grad():
            train_l = squared_loss(net(X), Y)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# softmax计算公式, 以及softmax网络的前向计算
def softmax(X):
    return torch.exp(X)/torch.exp(X).sum(dim=1, keepdim=True)

def softmax_net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W)+b)

# 交叉熵计算
def crossEntropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 官方的dataLoader怎么用呢
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == "__main__":
    print("-------------------------第二章-----------------------------")
    # 主要是矩阵计算：torch.dot(), torch.matmul(), torch.mm(), torch.normal(), torch.sum()(包含维度上的求和与求平均), torch.mean(), a@b
    X = torch.arange(6, dtype=torch.float32)
    Y = torch.ones(6, dtype=torch.float32)
    print(torch.dot(X, Y))
    # 向量内积
    # torch.matmul()可以做向量内积，可以矩阵与向量运算：矩阵的每一行与向量做内积
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    W = torch.ones(4, dtype=torch.float32)
    Y = torch.matmul(X, W) #注意维度变化，矩阵与向量运算，变成一维
    print(Y)
    # 按维度进行求和和平均：按照哪个维度，则哪个维度消失dim=1则列消失
    print(X.sum(dim=0, keepdim=True)) # dim=0则行消失，keep之后等于1
    print("-------------------------第三章--------------------------------")
    Y = linear(X, 2)
    # print(Y.shape)
    # 如何定义网络模型：继承nn.Module, 初始化权重参数, 前向传播
    net = myLinear(2, 1)
    # net = nn.Sequential(ll)#构建模型的话，还需要和其他一起放着
    # 如何训练呢：这个问题搞明白了，自己重写一遍，其实今天的工作也就差不多完成了呀
    # 1. 数据准备设置仿真数据：y = ax1+bx2+b.设置这样的数据50个
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    # 数据打包dataset和dataLoad.今天的关键所在
    batch_size = 10 # 为什么偶尔出现了inf的情况
    features, labels = synthetic_data(true_w, true_b, 1000)
    print("特征：", features.shape)
    print("标签：", labels.shape)
    # 看看做好的数据包怎么用
    for x, y in data_iter(batch_size, features, labels):
        print(x.shape, '\n', y.shape)
        break
    # 测试下tensor切片的灵活运用
    A = torch.arange(12).reshape(6, 2)
    print(A)
    b = torch.tensor([3, 4])
    c = [3, 4]
    print(A[b, 0]) #可以这样嵌套灵活使用，
    print(A[c]) #
    # print()
    # 开始训练
    lr = 0.03
    num_epochs = 3
    train(net, features, labels, batch_size, lr, num_epochs)
    # print(net[0].weight.data)
    # 怎么用官方的dataload和dataset
    print("---------------softmax------------------")
    # 实现softmax运算.主要是官方的实现版
    # 实现损失函数的计算
    # LSE问题继续
    X = torch.arange(72).reshape(-1, 3, 4)
    print(X)
    print(softmax(X))
    y = torch.tensor([0, 2]) # y的标签值对应的独热编码：[[1, 0, 0], [0, 0, 1]].也即：标签值是个一维向量，每个元素的含义是当前样本属于第几类
    y_hat = torch.tensor([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]])
    print(len(X)) #len(X)=X.shape[0]
    print(y_hat[range(len(y_hat)), y])










