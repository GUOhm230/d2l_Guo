"""
学习检验阶段。本处的代码优先自己写
参考之前章节的内容
目标：第13章，2个kaggle实验
2023.11.5 Created by G.tj
13.13: 图像分类，基于数据CIFAR-10数据
 
"""
from d2l import torch as d2l
import torch
import torchvision
import collections
from torchvision import transforms
from torch.utils import data
import math
import random
import os
import shutil
import cv2 as cv
import matplotlib.pyplot as plt
# 数据获取本是去kaggle那边下载。
# 得到5万张训练图片，30万张测试图片,32*32。测试图片好像不太好。数据怎么使用呢》labels怎么和数据存储在一起？
# 出现了第一个问题， 数据制作的dataloader的使用，我不太熟啊。主要是dataset

data_path = r"/home/gtj/SSDProject/job-hopping/data/cifar-10/train"
labels_path = r"/home/gtj/SSDProject/job-hopping/data/cifar-10/trainLabels.csv" # labels第一行的名字对应于名字
classes = []

# 先来看个例子：minist中数据是如何制作，怎么表现的
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans)
X, y = next(iter(data.DataLoader(mnist_train, batch_size=2)))
print(X.shape, y) # 生成的dataloader其中X的组织形式为[batch_size, 图片通道，h, w]
# datasets的使用，回顾一下
# 输入的应该是什么呢？
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

#这个features应该是什么呢？是指定的数据就是X，其长度就是他的维度
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
])
trainDataSet = torchvision.datasets.ImageFolder(root="../data/classImage", transform=trans)
# print(trainDataSet.classes)
# print(trainDataSet.class_to_idx)
# print(trainDataSet.targets)
# print("图片数量：", len(trainDataSet))
# print("图片路径：", trainDataSet.imgs)
# print("查看图片维度：", trainDataSet[0][0].shape)
# plt.imshow(trainDataSet[0][0])
# plt.show()
train_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=3, shuffle=True)
# for x, y in train_loader:
#     print(x.shape)
#     print("应该是标签", y)
# ImageFolder知道是干嘛的，但是为什么会出现这种情况。
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip', '2068874e4b9a9f0fb07ebe0ab2b29754449ccacd')

demo=True
if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = d2l.download_extract('cifar10_tiny')

print("-----------------------------------官方代码------------------------------------------------")
# 读取.csv文件，并存储成labels。文件名: 类别
def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
        # print(lines)
    tokens = [l.rstrip().split(',') for l in lines] # l.rstrip()删除末尾指定字符，空格.
    print("tokens:", tokens)
    return dict(((name, label) for name, label in tokens))

# 将图片复制到相应的路径
def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

# 训练数据集整理
def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1] # 包含最少图片的类别的数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels,valid_ratio)
    reorg_test(data_dir)

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    net = torch.nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():1f}' f'examples/sec on {str(devices)}')

if __name__ == "__main__":
    # 1.数据整理：分类任务的数据用torchvision.datasets.ImageFolder()
    print("-------------------------1. 数据制作--------------------------")
    batch_size = 32 if demo else 128
    valid_ratio = 0.1

    # 暂时不管，这步的作用其实就是把train中数据按照label中分类重新构建文件夹
    reorg_cifar10_data(data_dir, valid_ratio)
    # # 接下来使用ImageFolder()
    # transform_train = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(40),
    #     torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
    #                                      [0.2023, 0.1994, 0.2010])
    # ])
    # transform_test = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
    #                                      [0.2023, 0.1994, 0.2010])
    # ])
    #
    # # 设置训练数据。测试数据暂且不管. 看看
    # train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
    #                                             transform=transform_train) for folder in ['train', 'train_valid']]
    #
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
                                                transform=transform_train) for folder in ['valid', 'test']]
    #
    # train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True) for dataset in (train_ds, train_valid_ds)]
    #
    # valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    # test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
    #
    #
    # # 2. 构建模型，因为数据不够，所以我这里用一个于预训练模型
    # net = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # # print("net:", net)
    # net.fc = torch.nn.Linear(net.fc.in_features, 10)
    # print("net:", net)
    # # 初始化该层网络模型
    # torch.nn.init.xavier_uniform_(net.fc.weight)
    #
    # # 3. 模型损失,并定义优化器
    # loss = torch.nn.CrossEntropyLoss(reduction='none')
    #
    # # 4. 模型超参数设置
    # lr = 2e-4 # 设置太大不好
    # num_epochs = 50
    #
    # # 5. 开始训练。评价指标
    # # trainer = torch.optim.SGD(net.parameters(), lr, momentum=0.9) # 还需要添加weight_decay， 但是数据一般取多少呢？
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # net = net.to(device)
    # acc = 0
    # for i in range(num_epochs):
    #     net.train()
    #     for features, labels in train_iter:
    #         features = features.to(device)
    #         labels = labels.to(device)
    #         trainer.zero_grad()
    #         pred = net(features)
    #         l = loss(pred, labels)
    #         l.sum().backward()
    #         trainer.step()
    #         train_acc_sum = d2l.accuracy(pred, labels)
    #         y_hat = torch.tensor([torch.argmax(p) for p in pred])
    #         # cmp = y_hat==labels
    #         # print("pred:", y_hat)
    #         # print("labels:", labels)
    #         print("epoch: {}/{}, loss: {}, acc:{}".format(i, num_epochs, l.mean(), train_acc_sum/32.))

    # 还有关键几步：1.怎么做评价指标，训练完之后，如何提取其中的损失
    """
    这里其实很多地方是我的问题
    要了解一点：书本你是看了，但是缺乏自己动手练习。
    1. 对于数据的创建，尤其是dataset，dataloader这两个，很不明确怎么做，做目标分类的时候。这里需要重新开始。除了数据，其实其他的都是我自己做的
    2. 模型超参数的设置，为什么设置0.1和0.01差别居然这么大，L1正则权重的设置居然这么小。用到的优化算法是动量法
    3. 关于loss的使用居然也不是很熟悉。
    4. 图片后处理过程中，有个地方的代码需要重新做起来
    5. 预测的图片，需要对预测图片做测试，这个点其实我也不是很清楚。主要是代码怎么写。也很简单的。就是把我用训练数据做的那部分，换成测试数据就好了
    5. device的使用，并不是直接赋值
    """
    # x = torch.tensor([1, 2, 3]).to(device)
    # x = torch.tensor([1, 2, 3]).to("cuda") # 以上两者效果其实一致
    # print(x)
    # d2l.evaluate_accuracy_gpu()
