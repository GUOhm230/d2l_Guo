"""
目标：从零开始实现softmax
待解决的问题：
1.softmax的原理及其向量化
2.使用的数据集较大了，怎么调用并读取数据集，关于pytorch中
created by 郭辉铭 2023.7.15
"""
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l
from ch3_LinearRegression import sgd

# 获取图片文件名
def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]

# 绘制图片
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 为tensor张量
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# mnifashion数据读取整合
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) #注意这个用法，竟然可以这样拆分，如此灵活
    mnist_train = torchvision.datasets.FashionMNIST("../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST("../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

# 定义softmax操作
def softmax(x):#针对每个图片来说，x是一个d维列向量
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)# 按列求和，则列消失，同一行中所有元素想加维度[行*1]
    # 为什么要保持这个维度呢，方便后面做广播机制,若不加对一维是可以的，但是多维，则报错：RuntimeError: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 1
    return x_exp / partition

# 定义模型 softmax也是类似于线性回归的模型，只是从一个输出变成了多个输出，其运算形式并没有改变
def net(x):
    xr = x.reshape((-1, w.shape[0]))
    print("xr:", xr.shape)
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)

# 定义损失函数:交叉熵损失函数
def cross_entropy(y_hat, y): #交叉熵的函数为：y*log(y_hat)的求和,而只对预测正确的求导，故而，y=1,所以此处无有y*,因此，在标签中其实无需对y进行独热编码，只需要使用当前分类序号作为标签即可，因此y的维度=[batch_size, ]，y的意义是指示y_hat哪个位置的值应当被取用
    return -torch.log(y_hat[range(len(y_hat)), y]) #这个用法确实很妙，list的用法居然如此灵活

# 分类精度：针对tensor数据集，注意维度对应
def accuracy(y_hat, y):
    # 计算预测正确的数量，y_hat是一个n*分类数的矩阵，真实标签当然只有一个分类，但是作为损失函数，则维度也和y_hat一样，而本函数内，y的维度则为原始维度为n*1,其中每个样本只有一个分类，用数字代表分类，比如十分类就是0-9表示
    if len(y_hat.shape) > 1 and y_hat.shape[1]>1: #表示二维样本，意思就是样本数>1,分类数>1
        y_hat = y_hat.argmax(axis=1)# 获取的索引即为预测的类别，这个不难理解，因为使用的是独热编码的形式,这之后的y_hat的维度为[样本数]，argmax是求最大值
    cmp = y_hat.type(y.dtype) == y
    # print("y_hat", y_hat)
    # print("y", y)
    return float(cmp.type(y.dtype).sum())

# 该类的具体功能留待后了解,用于对预测变量的累加，因为在实际做数据时，我们都是针对一个batch进行的
class Accumlator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a,  b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 针对迭代器，以及模型输出结果进行比较，常用于eval中
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumlator(2)#构造保存数据的容器
    with torch.no_grad():
        for x, y in data_iter: # x维度为[b, 3, 28, 28], y为一个具体的数字，也就是[256, 1]， net[x]得到的结果是[256, 10]
            # print("net[x].shape=", net(x).shape)
            # print("y.shape:", y.shape)
            # print("y.numel()=", y.numel()) # y.numel()=y中包含的元素个数
            metric.add(accuracy(net(x), y), y.numel()) # acc返回预测正确的类型个数，y.numel()返回预测总数
            return metric[0] / metric[1]

# softmax从零开始实现的训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一轮"""
    if isinstance(net, torch.nn.Module):
        net.train()
    # 储存训练损失总和，训练准确度总和以及样本数
    metric = Accumlator(3)
    for x, y in train_iter:
        print("模型输入x维度：", x.shape)
        print("标签y维度：", y.shape)
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): # 使用了官方指定的优化器
            updater.zero_grad()
            l.mean().backward() # 计算梯度
            updater.step() # 对计算的梯度值进行更新
        else: # 使用自己定制的优化器，输入输出为啥呢？需要复习一下
            l.sum().backward()
            updater(x.shape[0]) # batch_size
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]


# 为了对训练数据进行可视化，一次编写，终身可用
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend=[]
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):#查看x中是否有该属性
            x = [x]*n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# 自定义优化器，也就是小批量梯度下降
def updater(batch_size):
    return sgd([w, b], lr, batch_size)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """模型训练"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc, ))
        train_loss, train_acc = train_metrics
        print(f'epoch{epoch + 1}, loss{train_loss: f}, acc{train_acc: f}')

    # train_loss, train_acc = train_metrics
    # print("train_loss", train_loss)
    # assert  train_loss < 0.5, train_loss
    # assert  train_acc <=1 and train_acc > 0.7, train_acc
    # assert  test_acc <= 1 and test_acc > 0.7, test_acc

# 模型预测
def predict_ch3(net, test_iter, n=6):
    pass

# softmax的简单实现：优化器，模型创建，初始化模型等都用官方的即可
# 权重初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    import cv2
    # 有关图像分类数据集的使用
    trans = transforms.ToTensor() #对下载的数据需要做的预处理：数据格式转换，resize, 归一化等：transform=transforms.Compose([transforms.ToTensor()])
    # mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    # mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    # print(len(mnist_test), len(mnist_train))
    # 可视化这些图片, 针对迭代器的使用
    # x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # a = np.transpose(mnist_test[0][0])
    # show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    # d2l.plt.show()
    # print(a.shape)
    # cv2.imshow("min", a.numpy())
    # cv2.waitKey(0)
    # 数据已经准备好
    train_iter, test_iter = load_data_fashion_mnist(256)
    # 开始从零开始实现softmax
    # ---------2. 初始化模型参数
    num_inputs = 784
    num_outputs = 10
    w = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # x.sum(0, keepdim=True)
    # 测试一下softmax算子, softmax算子并不改变当前特征的维度，而改变的是当前特征的数值
    x = torch.normal(0, 0.01, size=(2, 5))
    print(x)
    x_prob = softmax(x)
    print(x_prob)
    # 定义模型
    # 定义损失函数，并单独测试之
    y = torch.tensor([0, 2]) # 要搞清楚，目的是对一项物体进行分类，0,1,2表示物体所属类别，第一个样本的标签是0,也就是第一个样本实际为[1, 0, 0], 第二个样本标签是2也就是[0, 0, 1]
    y_hat = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y])
    print(np.log(0.1), np.log(0.5))
    print(cross_entropy(y_hat, y))
    print("-----------精度计算------------")
    # print("精度计算：", accuracy(y_hat, y))
    # tensor.type()， tensor.dtype以及type(a)的用法
    t1 = torch.tensor([1., 2., 3.])
    t2 = torch.tensor([1, 2, 3])
    t3 = torch.tensor([True, False, False])
    print("dtype:", t2.dtype, "\t", t1.dtype, type(t3)) #torch.int64 	 torch.float32
    print(t1.type())# 当不指定dtype时返回的是类型，也就是t1的数据类型为torch.FloatTensor
    print(t1.type(t2.dtype))# 当指定dtype后，返回类型转换后的数据，符合要求，则不做处理，直接返回
    print(t3.type(type(t2)))
    print(np.array([1, 2, 3]).dtype)
    print(type(accuracy))
    c = t1.tolist()==t2.tolist()
    print(t1.tolist())
    print(c)
    print("-------------eval下的精度--------------")
    print(evaluate_accuracy(net, test_iter))
    print("----------开始训练-------------")#重中之重
    lr = 0.1
    # num_epochs = 3
    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # d2l.plt.show()
    print("softmax的总结")
    """"
    重新梳理下softmax的从零开始实现：
    1. 搞清楚fashion-mnist数据的下载以及读取和迭代器的创建，而后由于图片是三维数据，因此需要将这些图片展平：x.reshape((-1, w.shape[0]))
    2. w以及b的创建。其中w的维度为特征数*输出维度: W = torch.normal(0, 0.01, size=(num_examples, num_outputs), requires_grad=True)
    3. 关于softmax的计算：exp(x)/exp(x).sum(axis=1, keepdim=True).
    4. loss的创建处的简要表达：针对label，无需使用独热编码，因为ylogy_hat中针对y=0的点是省略的，而y的唯一作用是指示分类的位置。关于列表的技巧需要掌握：-torch.log(y_hat[range(len(y_hat)), y])
    5. 自定义优化器，所谓优化器主要是进行参数更新，参数更新是通过梯度实现：param = param - lr*param.grad/batch_size.使用官方定义的优化器则使用loss.mean().backward()获取当前梯度后updater.step()进行梯度更新
    6. 训练步骤则是定义模型-->以batch_size为单位进行运算-->求损失-->计算梯度-->参数更新-->计算损失以及训练准确率（y_hat.argmax(axis=1)使之与y.shape相等）-->画图，绘制走向
    """
    aa = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bb = [1, 2]
    print(list(range(len(aa))))
    print(aa[[0, 2], bb])
    print("---------------reshape[-1]-------------------")
    rr = np.arange(16).reshape(-1, 4)
    print(rr)
    print("----------------------------softmax的简单实现-----------------------------------------")
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    print(net)
    net.apply(init_weights) #会递归的将init_weights应用到net的子模块中，有关初始化参数的部分，这些函数要整理下
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(net.parameters(), lr = 0.1)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.plt.show()

