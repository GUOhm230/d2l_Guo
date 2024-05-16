"""
深度学习计算
该部分主要完成
自定义块，自定义层
怎么自己定义一个网络，然后和官方层实现反向传播
2023.8.7 created by G.tj
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch._jit_internal import _copy_to_script_wrapper

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from collections import OrderedDict, abc as container_abcs

"""
自定义块：
需要继承nn.Module模块，方便添加到nn.Sequential
需要输入数据
需要初始化参数，如果用的是官方组件，则不用初始化参数
需要前向传播的一组表达式
需要损失函数
需要反向传播，反向传播知道把谁当做多元函数的变量呢
"""
# 定义一个实现前向传播的MLP模型
class MLP(nn.Module):
    # 定义层
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    # 设置前向传播函数
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
        # 所以真就是自己做的时候才知道问题。之前定义模型的时候是用的sequential列表添加，其实是定义这样一个类型，该类中包含forward方法，可以直接对象(forward形参调用)


# 自定义顺序快，自定义Sequential()
class MySequential(nn.Module):
    # 初始化并维持一个列表，用于储存module类成员
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    # 定义前向计算
    def forward(self, X):
        for block in self._modules.values():
            print("网络层：", block)
            X = block(X)
        return X

# 从嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# 将所有参数初始化为给定的常量
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

# 使用Xavier初始化
def init_xavier(m):
    print("看看怎么进去初始化的：", m)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# 进行自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()[0]])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # 定义前向运算，输入输出维度应当一致
    def forward(self, X):
        return X-X.mean()


# 自定义带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units)) # 这个用法需要搞清楚啊: 输入维度，输出维度
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear) # 这些方法的灵活应用要重新看一下，应该也就20分钟

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X))) # 注意：要执行并使用relu函数，该函数是在

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus(): #@save
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

if __name__ == "__main__":
    # 应用模型定义
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    """
    该列表在运行时，指定：
    def forward(self, input):
        for module in self:
            input = module(input)
        return input
    """
    X = torch.rand(2, 20)
    # for m in net:
    #     print(m)
    #     X = m(X)
    #     print(X.shape)
    y = net(X)
    mlp = MLP()
    print("MLP模型参数：", mlp.state_dict())
    # mlp(X)
    # print(net)
    print(y)
    # 其前向推断方法完全与官方方法一样,但是由于初始化的参数不一样，因此最终前向传播的结果其实也可能不一样
    # net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # print(net(X))requires_grad
    # for w in net.parameters():
    #     print(w)
    print("------------------------------模型参数的读取--------------------------------") # 正是该章接下来参数管理部分
    # 1. 按照列表迭代的方式读取，net.state_dict是一个字典。某一层也存在这个东西？是的，只要继承了nn.Module都有这个东西，有这么一个方法,所以，Sequential有，其下的某一层当然也会有
    # print(net.state_dict())
    # print(net[0].state_dict())
    print(net[0])
    print(net[0].weight) # nn.Linear中有：self.weight=，故weight以及bias为其中属性
    print(net[0].bias) # nn.Linear中有：self.weight=，故weight以及bias为其中属性
    print("数据所在设备：", X.device)
    # 一次性访问所有参数
    l = []
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()]) # 在这类代码中Sequential内的每个操作都算单独一层进行编号
    print(net._modules["0"]) #module中是有这么一个属性的。add_module用于在net中添加层或块
    a = nn.Linear(3, 4)
    a.add_module("zidingyi", nn.ReLU())
    print("直接对单个层添加同档次的网络层，是否可以呢？", a) #实际不报错，但是计算好像无意义。暂时未知啊
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print("这个又是怎么完成的计算呢：", x)
    print("这个又是怎么完成的计算呢：", a.state_dict())
    print("a.weight.data：", a.weight.data)
    y = torch.matmul(x, a.weight.data.T) + a.bias.data # 官方
    print(y)
    print(a(x))
    print("---------------------设计嵌套块并获取其中参数----------------------------")
    # 嵌套读取即可.其实本处更多的是关于如何自定义块状的模型，嵌套块，放入其中
    regnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(regnet)
    print(regnet[0][1][0].bias.data) # block2中第一个block1块的第一层也就是Linear(4, 8),该层的w参数就是[4, 8],b参数则是一个一维8个数的向量。
    print(regnet[0][1][0].weight.data) # block2中第一个block1块的第一层也就是Linear(4, 8),该层的w参数就是[4, 8],b参数则是一个一维8个数的向量。
    print("------------------------------参数初始化-------------------------------------")
    # 函数初始化可以分为内置初始化和自定义的初始化
    # 本处主要学习两个地方：1. 关于apply的使用，在何处可以使用
    net.apply(init_normal) # 递归的运用到net的每个子模块中
    net[0].apply(init_xavier) # 或者其实可以对单个层应用上述方法
    print(nn.Linear(in_features=20, out_features=256, bias=True).weight) # 这里.weight也可以是网络层的名字
    # print(net.state_dict())
    # 进行自定义初始化
    print("参数设置：", net.named_parameters())
    for name, param in net.named_parameters():
        print(name, param.shape)
    # 对某个数据位置进行
    net[0].weight.data +=1
    print(net[0].weight.data)
    print("--------------------如何做好参数绑定--------------------------")
    # 进行参数共享，需要提前设定模块，不然w初始化是不一样的
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    # print(net(X))
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    print(net[4].weight.data[0, 0])
    print("------------------------延后初始化------------------------------")
    # 所谓延后初始化就是，输入维度不确定的时候
    print("----------------------------进入本章的重点内容：如何自定义层----------------------------------")
    # 自定义层有两种形式：1. 不带参数的层；2. 带参数的层
    # 1. 定义不带参数的层：
    # 应当如何定义呢：1. 继承nn.Module. 2. 实现前向传播，也就是定义一个表达式就行了
    # 应当怎么应用呢？
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    y = net(torch.rand(4, 8))
    print(y.shape)
    # 2. 定义一个带参数的层
    # 应当如何定义呢：1. 继承nn.Module，2. 参数初始化，因为nn.Module中不会带有参数初始化。事实证明，多读读官方代码确实有好处3.实现前向传播算法
    # 那如何使用呢：其实和之前的一模一样
    linear = MyLinear(5, 3)
    X = torch.rand(4, 5)
    print(X)
    print("输出结果", linear(X))
    print(linear.weight.grad)
    print(linear.weight.data)
    # 要看看官方定义的模型是在哪实现的模型参数初始化
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))
    print("------------------------读写文件------------------------------")
    # 1. 加载和保存张量：1个张量的保存与读写 2.整个模型权重张量的保存与读写
    X = torch.arange(4)
    torch.save(X, "singleTensor.pt") # 张量保存
    X2 = torch.load("singleTensor.pt") # 张量读取
    print(X2)
    Y = torch.zeros(4)
    print("[X, Y]=", [X, Y])
    torch.save((X, Y), "doubleTensor.pt")
    xy = torch.load("doubleTensor.pt") # 读取的形式和保存的形式是一致的,[], ()
    print("xy=", xy)
    mydict = {"X": X, "Y":Y}
    torch.save(mydict, "mydict.pth")
    print(torch.load("mydict.pth"))
    # 1. 保存模型参数， 2. 读取模型参数，3. 加载模型参数
    net = MLP()
    for para in net.parameters(): #
        print(para.shape)
    # print(net.state_dict())
    for name, para in net.named_parameters(): #不再
        print(name, para.shape)
    # print(net.state_dict())
    net.hidden.weight.data = torch.ones_like(net.hidden.weight.data)
    print("hidden.weight=", net.hidden.weight.data) # net是一个对象，该对象中有hidden这个属性，而这个属性同样是个对象，该对象nn.Linear中也有weight这个属性。因此可以如此调用
    # print(net[0].state)
    torch.save(net.state_dict(), "net.pth")
    net.load_state_dict(torch.load("net.pth"))
    print(net.state_dict())
    print("------------------------GPU-----------------------------")
    # 1. 怎么查看GPU 2. 怎么将数据， 模型转到某张GPU上 3. 怎么在GPU上实现运算
    # 1. 查看GPU状态，数量
    print("本机显卡数量：", torch.cuda.device_count())
    print("本机显卡是否存在：", torch.cuda.is_available())
    # 2. 怎么指定GPU使用
    # 一个问题：torch.device("cuda")是怎么应用到模型和数据的？
    print("device设备名", try_gpu())
    X = torch.tensor([1, 2, 3])
    print("查看X所在的位置：", X.device)
    # 将数据存储在GPU上
    X = torch.ones(2, 3, device=try_gpu())
    Y = torch.zeros_like(X, device="cpu")
    print("关注Y内存位置：", id(X))
    print(X, "\n", Y)
    Y1 = X.cuda(try_gpu()) # 括号内用device, RuntimeError: Invalid device, must be cuda device
    Y = Y.to(try_gpu()) # 括号内用device,
    print("测试数据转储方式2：", Y)
    """
    官方解释：
    If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.
    """
    print("新Y位置：", id(Y)) # 如果移动后的位置与原不在同一显卡，则需要复制数据。如果所在位置是一样的，则地址是一样的，意味着原路返回
    print(X.cuda() is X)
    print(X+Y)
    # 如何对模型指定存储，运算位置
    print("net=", net)
    # net = net.cuda() # 和数据转存方式类似
    net = net.to(try_gpu())
    print(net)
    X = torch.rand(4, 20, device=try_gpu())
    print(net(X)) # 参数和模型需要在一个设备上才能进行模型运算

    """
    第五章内容总结：
    本章主要学习深度学习的计算的相关组件：模型构建，参数访问，自定义层和块，GPU对于计算的加速
    主要学到了以下内容
    1. 如何从层定义网络模型到由块定义。如何创建块：块由层组成，只是块可以最后的输出也用relu函数.而层的定义需要：1. 继承nn.Module 2. 定义前向计算方法 3. 初始化模型权重 3. 反向传递是根据loss自动微分，自动识别叶子张量为多元函数的变量的
    2. 由块定义网络时候，使用块的循环时，则需要保证首尾能相接。将块加入net的方法是net.add_Module().其中net需要是nn.Sequential对象，该对象才维持一个_modules列表，才有add_Module方法
    构建模型时，最外层是nn.Sequential()类，其中的组件可以是块，可以是层，可以是自定义的层和块，也可以是nn.Sequential().理论上可以无限嵌套下去
    3. 学习了参数的访问：net.state_dict()一次性输出所有参数，net[i].state_dict()，根据自己对模型的熟悉程度，可以根据嵌套取出其中模型参数，net[i].weight.data可以得到w权重或者b
    for name, para in net.named_parameter():可以取得每一层的名字和参数。for para in net.parameters()可以取得每一层参数而无法得到名字。
    也能根据层名：m.weight.data直接访问权重m可以.
    注意:参数的访问一定要到层
    4. 学习了如何进行参数初始化：def init_weight(m): if type(m) == nn.Linear:nn.init.xavier_uniform_(m.weight).使用net.apply(init_weight)进行递归调用net中的模块直到层
    学习了如何自定义的初始化，可以根据参数访问的位置进行精准修改参数。使用矩阵运算操作
    5. 学习了如何绑定参数，共享权重：需要在模型构建前建立该层，否则模型创建即初始化。此时即使w维度一样，初始化也是随机的，因此要shared=nn.Linear()。之后在nn.SEQUENTIAL中调用shared
    6. 知道了为什么要延后初始化：因为样本个数未知，或者输入维度不定，而且卷积中也是不确定输入维度的。因此要数据通过一次才进行模型初始化
    7. 知道怎么自定义层和块：带参数和不带参数。几大要素：1. 初始化权重参数，2. 继承nn.Module, 3. 定义前向传播 4. 权重初始化的时候注意标明required_grad=True。但是实际使用nn.Parameter是不需要这个的
    8. 知道了如何保存和读取并加载模型参数。参数可以单个保存，可以组成列表，元组，字典的形式保存
    9. 知道GPU的查找，设定，怎么将数据和模型转到GPU上：X.cuda(), net.to(device):模型和数据能用同样的方法.这个device=torch.device(设备)而不是device=设备.好像直接设备也没问题
    """
    block1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(),
                           nn.Linear(8, 4), nn.ReLU(),
                           nn.Linear(4, 16), nn.ReLU()).cuda()
    X = torch.rand(4, 16, device=try_gpu())
    print("X.shape", X.shape)
    Y = block1(X)
    print("自己测试一下：", Y)
    # 根据刚才定义的块来定义网络
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1)
    print(block1[0].weight.data)
    # net.add_module(nn.Linear(16, 2))
    print(net(X))
    print("------------为什么是做梯度相加，这个部分有点不清楚------------")
    lnet = nn.Linear(4, 2)
    X = torch.randn((3, 4))
    print(X)
    print(lnet.state_dict())
    loss = nn.CrossEntropyLoss(reduction='none')
    # l = torch
    




#

