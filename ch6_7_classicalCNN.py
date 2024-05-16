"""
实现一些经典的CNN模型
2023.8.23 created by G.tj
熟悉几大经典网络
1. LeNet
2. AlexNet
3. VGG
4. GoogLeNet
5. Nin
6. ResNet（放到后面）
7. BN层的复习
8. DenseNet
其实之前把网络层都搞清楚了
"""
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# LeNet网络
# 自己复现版:要实现的是网络模型，而实现层的时候需要满足三要素：1）定义类并继承nn.Module 2) 初始化权重参数(使用nn.Parameter(size=)比较方便)，该权重需要加入loss的多元函数变量列表中进行梯度计算，权重更新 3）实现前向计算
# 模型的实现涉及很多python方法，需要自己手动去做，不然始终会出现这种麻烦
# 官方定义模型并且要训练的其实还因为A的mro表中不包含B。好的，我们处理了第一个问题是
class LeNet(nn.Module):
    def __init__(self, ):
        super().__init__(self)

# 准确率统计
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device # 找到权重所在的位置

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 模型训练,后面几大经典网络其实训练路径和这个是一样的
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 其实模型本身已经初始化一遍了，这是重新初始化参数了吗？待会测试下，把数据改成自己
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print("训练设备：", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train_loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*x.shape[0], d2l.accuracy(y_hat, y), x.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i+1)/num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l: .3f}, train acc {train_acc: .3f}, '
          f'test_acc {test_acc: .3f}')
    print(f'{metric[2] * num_epochs / timer.sum(): .1f} examples/sec '
          f' on {str(device)}')

# 定义VGG网络块：自己换一种做法吧
def vgg_block(num_convs, in_channels, out_channels):
    layers = [] # # 如果改成layers=nn.Sequential(),则不用append，用的是layer.add_Module()
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels # 要保证无论多少个
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))# 因此每个卷积块还是实现了宽高的减半
    return nn.Sequential(*layers)

# 通过网络块定义网络层
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10))

# 定义NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

# 定义GooLeNet 以定义Inception块。本处有关python的程序设计方法需要加强
# 本处模型的定义不再是顺序块了，因此不能用nn.Sequential()就囊括一整个模型
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs) # 添加可以认为是强绑定
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        return torch.cat((p1, p2, p3, p4), dim=1)

if __name__ == "__main__":
    print("---------------------------1. LeNet模型的实现----------------------------")
    # 先用简单的方法实现一遍
    LeNet = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2), # 这里的结果是80*5
                        nn.Flatten(), #卷积到FC层需要展平
                        nn.Linear(400, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid(),
                        nn.Linear(84, 10)
                        ).to(f"cuda:0")
    # print("模型架构：", LeNet)
    # print("模型所在设备: ", LeNet.device)
    # print(LeNet[0].weight.device) # 由于输入通道是1, 输出通道是6 故W的维度，也就是K的维度是[out_channel, in_channel, H, w]
    # print(LeNet.state_dict()["0.weight"]) # 和上面结果是一样的，可以直接取出其中的权重参数
    # X = torch.rand(1 * 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28).cuda() # 为什么要四个维度：nn.Flatten()的展开有关
    # print("X:", X.device)
    # Y = LeNet(X)
    # print("LeNet结果：", Y)
    # for name, para in LeNet.named_parameters():
    #     print(name, "\t", para.shape) # 获取的是w的维度，但是实际是转置

    # 每层名字获取并获取该维度的输出结果
    # for layer in LeNet:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)
    # 模型训练
    # 1.数据集获取
    # batch_size = 256
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    # lr, num_epochs = 0.9, 10
    # train_ch6(LeNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # 训练结果有问题，留待后改: 发现准确率接近0.1.首先反应是不是因为没有更新权重参数，一查果然是。
    # 但是当我想到要检查其中参数时，却生出一丝害怕。直觉不会骗我。我对这个地方还是不熟。参数获取，继续练习一下
    """
    这些内容的难点并不是网络和模型的构建
    1. 而是创建训练模式
    2. 训练流程包括：模型数据获取并制作，模型创建，loss创建，使用自动微分进行参数更新
    3. 如何设定停止条件，一般是训练多少轮就停滞了
    4. 如何保存信息，进行训练控制。这个涉及内容太多了，也包括很多python知识
    思考如何完成反向传播：1. 继承nn.Module, 初始化参数的时候调用nn.parameter() 2. 设置loss 3. 使用自动微分，optim.step()
    """
    print("---------------------------2. AlexNet------------------------------")
    # 本处设置网络要较为专业一点,应用class。至于网络如何，我本处不需要细究
    # AlexNet的作用是添加了新的网络层，DropOut
    # 1. 模型构建
    AlexNet = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.5),
                        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Flatten(),
                        nn.Linear(6400, 4096), nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096, 4096), nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096, 10))

    # print("初始化权重：", AlexNet[0].weight.data)
    # print("该层权重形状：", AlexNet[0].weight.data.shape) # 预计维度：96, 1, 11, 11
    # 2. 模型运行测试， 获取层名以及
    # X = torch.randn(1, 1, 224, 224)
    # for layer in AlexNet:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)
    # # 2. 数据获取，模型训练
    # batch_size = 128
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # lr, num_epochs = 0.01, 10
    # train_ch6(AlexNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # print("训练之后权重：", AlexNet[0].weight.data)
    # print("该层权重形状：", AlexNet[0].weight.data.shape) # 预计维度：96, 1, 11, 11 实际：torch.Size([96, 1, 11, 11])
    """
    AlexNet：
    该网络是深度学习模型相对于传统机器学习展现出来的第一次优势def
    主要是使用了nn.Dropout()以及nn.ReLU
    将nn.AvgPool2d()改成nn.MaxPool2d()效果被证明也是更好
    使用了更大的卷积核，以及更深的网络
    其余对于我的学习来说则是教会我模型设计思路
    关于dropout需要复习一下
    1. 该方法的目的是为了让复杂的模型简单化，因为A的mro表中不包含B。好的，我们处理了第一个问题这样不仅能减少过拟合，还能加快收敛
    2. 该方法的实现过程是每个神经元的以p概率置为0。而每个神经元都是进行一次线性运算的单元
    3. 神经元置为0的意思是，该神经元的输出为0。并不是说这个神经元下的d个权重全为0。权重是仍然保持的
    4. 一般只针对FC而言，针对卷积则大可不必，因为卷积本身权重就比较少，维度很低，且怎么计算呢？哪个作为神经元？
    5. 实验证明，dropout确实可以加在任意一层，但是在卷积层加没有意义
    """
    print("---------------------------3. VGG网络--------------------------------")
    # 该方法的意义是，自己定义卷积块
    # 下午为什么到现在才开始：午饭1点，做完2点，吃完，收拾好近3点今天的游戏没有在学习期间玩，但是玩了个无用的地方
    # 1. 创建模型
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    VGGNet = vgg(conv_arch)
    # 2. 随机数字测试
    X = torch.randn(size=(1, 1, 224, 224))
    # for blk in VGGNet:
    #     X = blk(X)
    #     print(blk.__class__.__name__, 'output shape： \t', blk)

    # 训练仍然和之前是一样的
    # batch_size = 128
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # lr, epochs = 0.01, 10
    # train_ch6(VGGNet, train_iter, test_iter, epochs, lr, d2l.try_gpu())
    # 数据较大，我要怎么把网络减半呢？无非就是减少输入输出通道以及降低宽高
    # ratio = 4
    # small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch] # 将输出通道全部降低4倍
    # print(small_conv_arch)
    """
    VGG网络模型中学会的是如何构建网络块
    并通过网络块构建网络模型
    VGG中对K进行新的尝试：K的核数较少，而网络层数较高时的效果更好
    注意一点：使用for layer in net：时，layer是net的第一层嵌套，是块就是块，非块是层。并不像net.apply()那样，逐级遍历层
    """
    print("----------------------------------------------4. NiN网络----------------------------------------------------")
    # 引入1*1卷积以改进全连接,：在每个像素位置应用全连接层，也就是对通道维进行线性以及激活函数的操作
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    # 2. 引入测试一下
    # X = torch.rand(size=(1, 1, 224, 224))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape： \t', X.shape)
    # 全局平均汇聚层是什么？怎么在卷积之后使用这个将数据转成对数几率。知道怎么用，知道维度是什么。其他的就不清楚了。自己设置的核以及步长，无需人为设定
    # m = nn.AdaptiveAvgPool1d(5) # 后面宽高位置按照指定生成
    # input = torch.randn(1, 2, 8)
    # output = m(input)
    # print("input:", input)
    # print("output:", output)
    # print(output.shape)
    # 模型训练部分，其实和别的地方是一样的
    # lr, num_epochs, batch_size = 0.1, 10, 128
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    """
    NiN网络好像没啥好的新东西。主要就是一个全局自适应卷积的使用
    """
    print("---------------------------------5. GoogLeNet-----------------------------")
    # 主要是Inception块的设立, 用class去创建模型。因此本处的内容才是重点中的重点
    # 块创建完成之后进行模型构建
    # 块的探索相当于滤波器，我用各种滤波器的组合去探索图像，这样可以拿到更多的图像细节
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),

                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)), #沿用了NiN网络的设计指定宽高输出
                       nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 模型训练
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    device = d2l.try_gpu()
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    """
    总结一下GoogleNet:
    1. 该网络的创新点是使用了多个卷积网络作为并行通道，然后按照通道维进行拼接，这样的话可以有效识别不同范围的图像细节
    2. 使用了NiN的思路：用1*1卷积代替了全连接。但是他后面还是有全连接
    3. 接下来就是关于代码方面的进展了：
        super(type, obj).__init__(**kwargs)。知道这个是用来进行强绑定的，但是不是很清楚什么是强绑定，什么不是
        使用class怎么定义网络的部分也是从这里开始的
        torch.cat()需要进一步的学习，留待后做吧。后面已经做完并做好了笔记
    """
    """
    总结：
    本轮涉及的模型包括：LeNet，AlexNet，VGG，NiN，GoogLeNet
    后面的ResNet单独一列
    1.主要难点还不是在模型构建，而是在训练的时候，训练流程也比较清楚，难得是一些前后处理。这种主要还是考研python以及编程功力
    2.主要学习卷积，汇聚（池化层）等的官方API：nn.Conv2d(), nn.MaxPool2d()
    3.学习了网络模型构建时如何组织：nn.Conv2d()后要加激活函数，卷积后加池化，最后用全连接进行预测。总之就是：卷积学习特征，池化汇聚信息改变宽高，调整模型复杂程度，1*1卷积相当于通道维上的全连接，可以替代FC以大幅减少模型复杂度
    4.学习了网络模型创建的一些思路：卷积核大小的设置，可以考虑组合模块，用块的方式构建网络，用1*1卷积代替FC
    5.接下来就是一些python知识了
    """

