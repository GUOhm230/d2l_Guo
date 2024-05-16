"""
本专题主要解决之前一直迷惑的数据处理问题
涉及DataLoader和DataSet

"""
import random
import torch
from collections.abc import Iterable, Iterator
from torch.utils import data
import os
from itertools import accumulate
import cv2 as cv
from torchvision import transforms
from PIL import Image
print("---------------------------1. 先看个之前的例子--------------------------------")
# 自己序列数据处理。定义生成器函数
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 原地打乱list顺序, 两者地址是一样的
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

w = torch.tensor([2, -3.4]).reshape(2, 1)
b = 4.2
batch_size = 32
features = torch.normal(0, 1, (1000, 2)) # [1000, 2]的输入序列，1000是样本数量，2是特征数量
labels = torch.matmul(features, w) + b # tensor矩阵的乘积
labels += torch.normal(0, 0.01, labels.shape).reshape(-1, 1)
train_iter = data_iter(32, features, labels)
print(isinstance(train_iter, Iterable)) #True ，该生成器对象就是一个可迭代对象
# for x, y in train_iter:
#     print(x.shape)

# 使用官方提供的API接口
def load_array(data_arrays, batch_size, is_train=True):
    #构造一个pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays) # 其中的data_array是元组(x, y)
    print(isinstance(dataset, Iterable)) #False
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 其实这个方法是常有的。那关键是怎么包装成DataSet类呢

data_iter = load_array((features, labels), batch_size)
# data_iter是迭代器，iter方法返回的是一个迭代器对象
for x, y in data_iter:
    print(x.shape)

print("---------------------------2. dataset-------------------------------------")
# dataset就是把数据打包成dataset
# 1. torch.utils.data.TensorDataset().该类继承自Dataset
dataset = data.TensorDataset(features, labels)
print(dataset[0])
i=0
# 这里明明是两个数据。就是一个features，一个labels
for tensor in (features, labels):
    i += 1
    print(i)

tt = tuple(tensor[0] for tensor in (features, labels)) # 相当于tuple(features[0], labels[0])
print(tt)
# t = tuple((1, 2))
# print(t[1])
# 总结一句：该对象的作用是把分离的x,y按照样本序号进行组合，并使得可以按索引存取

# 2. data.Dataset的使用：import torch
# 官网给的一个实际例子
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data2",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data2",
    train=False,
    download=True,
    transform=ToTensor()
)
# d2 = data.Dataset()
# 要实现dataset需要自己重写，我来尝试一下.
# 1.给features， labels是序列的数据写一个dataset集成
# 首先明确该类的目的：做数据集成，让x, y可以一一对应
class MyDataset(data.Dataset):
    def __init__(self, *tensors, device=torch.device('cpu')):
        # 根据官方提示，新增一条
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors) # 判断输入数据的每个维度
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple((tensor[index] for tensor in self.tensors))

x = torch.arange(24).reshape(6, -1)
y = torch.rand(6).reshape(-1, 1)
# print(x)
# print(y)
print("-----------------------------------")
datasetTest = MyDataset(*(x, y))
print(any(x.size(0) == tensor.size(0) for tensor in (x, y)))
print(datasetTest[0])
# 3. 如果要实现自己给图片进行提取操作呢？假如要给RGB实现图片的读取。(imgs, labels)
# 所以要给定两个之后数据之后读取rgb图像和labels
def read_file(root:str):
    file = os.listdir(root)
    dir_list = []
    for fileName in file:
        path = os.path.join(root, fileName)
        if os.path.isdir(path):
            dir_list.append(fileName)
    classes = [0, 1, 2, 3, 4]
    file_classes = dict(zip(dir_list, classes))
    # 储存训练图片的文件路径
    train_img_path = []
    # 储存训练图片的分类数
    train_img_labels = []
    for img_fileName, labels in file_classes.items():
        img_dir = os.path.join(root, img_fileName)
        curr_file_list = os.listdir(img_dir)
        for curr_file in curr_file_list:
            train_img_path.append(os.path.join(img_dir, curr_file))
        train_img_labels += [labels] * len(os.listdir(img_dir))
    print("图片设置：", len(train_img_labels))
    print("图片设置：", len(train_img_path))

    # img_num = list(accumulate([len(os.listdir(os.path.join(root, fileName))) for fileName in file_classes]))
    # print("图片", img_num)
    # # 要把所有的图片文件读取一下
    # for k in file_classes:
    #     print(k)
    return train_img_path, train_img_labels
    # for i, j in file_classes:
    #     print(i, j)

# 定义函数，读取文件夹内的文件路径. 其实还要加上一步，就是训练集的预处理方法
class MyImageDataset(data.Dataset):
    def __init__(self, img_path, transforms=None):
        self.root = r"/home/gtj/SSDProject/job-hopping/flower_photos"
        # 获取文件夹并分类
        self.train_img_path, self.train_img_labels = read_file(root)
        self.transforms = transforms
        # self.img_file = [os.listdir(os.path.join(root, fileName)) for fileName in self.file_classes]
        # self.img_num = list(accumulate([len(os.listdir(os.path.join(root, fileName))) for fileName in self.file_classes]))

    def __len__(self):
        # print("计算长度：", self.)
        assert len(self.train_img_labels) == len(self.train_img_path)
        return len(self.train_img_labels)

    def __getitem__(self, index):
        img = Image.open(self.train_img_path[index])
        if self.transforms:
            img = self.transforms(img)
        return tuple((img, self.train_img_labels[index]))


root = r"/home/gtj/SSDProject/job-hopping/flower_photos"
# torchvision下有很多官方数据集的获取dataset.cifar10
import torchvision
datasets = torchvision.datasets.CIFAR10(root="../data", train=True, transform=None, download=True)
# print(sum([23, 45]))
# read_file(root)
# arr = [1, 2, 3, 4]
# print(list(accumulate(arr
trans = transforms.Compose([transforms.Resize([240, 240]),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4832, 0.4856],
                                                 std=[0.2023, 0.2013, 0.2111])])

my_image_dataset = MyImageDataset(root, transforms=trans)
print(isinstance(my_image_dataset, Iterable))
img, labels = my_image_dataset[0]
print(img.shape)
tensor_dataloader = data.DataLoader(my_image_dataset,   # 封装的对象
                               batch_size=32,     # 输出的batch size
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程
#
# # 以for循环形式输出
for data, target in tensor_dataloader:
    print(data.shape, target)
