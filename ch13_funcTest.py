"""
这里主要是关于第13章的一些函数方法的学习和应用
"""
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
import cv2

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update): # num_updates又是根据什么而变化的呢？
        return self.lr * pow(num_update + 1.0, -0.5)

if __name__ == "__main__":
    # # 获取数据
    # batch_size, crop_size = 32, (320, 480)
    # train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size) # 这个数据是dataloader数据，我到要看看这个数据怎么获取
    # # # 对预测图片做归一化
    # # # X = test_iter.dataset.normalize_image(img)
    # # voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    # # test_images, test_labels = d2l.read_voc_images(voc_dir, False)
    # # # for i in range(10):
    # # #     print(test_images[i].shape)
    # # n, imgs = 1, []
    # # for i in range(n):
    # #     crop_rect = (0, 0, 320, 480)
    # #     X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    # #     # print(X)
    # #     # print(type(X))
    # #     # print(X.shape)
    # #     X1 = test_iter.dataset.normalize_image(X) # 对每个通道做归一化
    # #     # print(X1)
    # #     norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 怎么做的，先归一化，再
    # #     Y = norm(X.float()/255)
    # #     print(Y)
    # #     print(X1==Y)
    # #
    # # """
    # # 该部分需要学的东西：
    # # 1. 图片的裁剪怎么进行。使用.crop()
    # # 2. 对通道进行归一化。归一化的计算：/255, 然后(X-mean)/std
    # # """
    # # # print("测试数据的读取：", test_images[0].shape)
    # # # x, y = next(iter(train_iter)) # iter()是先将数据转为迭代器类型，再用next可以取其中一个
    # # # # print(iter(train_iter)[0])
    # #
    # # # print(x.shape) # [32, 3, 320, 480] 图像数据
    # # # img = x[0].permute(1, 2, 0).unsqueeze(0)
    # # # print(img[::2].shape)
    # # # d2l.show_images(img, 2, 2)
    # # # d2l.plt.show()
    # # # plt.imshow(img, cmap="gray")
    # # # plt.show()
    # # # img1 = cv2.imread("/home/gtj/Pictures/personal photo/2017.jpg")
    # # # print(img1.shape)
    # # # print(type(img1))
    # # # print(cuda.torch.version)
    # # # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # # # cv2.imshow("img", img1)
    # # # cv2.waitKey()
    # # # print(img.shape)
    # # # print(type(img))
    # # # print(img.type())
    # # # print(y.shape) # [32, 320, 480] 该数据是标签。每个数字表示像素的类别，因为语义分割是针对每个数字来讲的
    # # # print(train_iter.num_workers)
    # # # 尝试显示一下
    # # a = torch.tensor([0.1, 0.2, 3])
    # # print(a.mul(255)) # tensor元素逐个乘以255
    # # # 学习率调度器
    # # """
    # # 所谓学习率调度器其实就是有关学习率的的衰减和变化
    # # 在不同阶段，学习率根据步数进行变化
    # # """
    # # scheduler = SquareRootScheduler(lr=0.1)
    # # # 实现一个官方的试试。多因子调度器，也就是给一些学习率时间，其实就是学习率的步数，随着这个步数，学习率进行一定的调整
    # # net = nn.Sequential(
    # #     nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    # #     nn.MaxPool2d(kernel_size=2, stride=2),
    # #     nn.Conv2d(6, 16, kernel_size=5), nn.ReLU()
    # # )
    # # trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
    # # trainDataSet = torchvision.datasets.ImageFolder()
    # # d2l.train_batch_ch13()
    # # d2l.evaluate_accuracy_gpu()
    # # a = torch.tensor([0, 1, 2, 3])
    # # b = torch.tensor([1, 2, 2, 4])
    # # c = a==b
    # # print(float(d2l.reduce_sum(d2l.astype(c, a.dtype)))) # 给
    # # d = float(d2l.reduce_sum(d2l.astype(c, a.dtype)))
    # # print(c)
    # # d = dict((("name", "guo")))
    # str1 = "12guo1aa21"
    # print(len(str1))
    # # str2 = str1.rstrip("a1")
    # # print(len(str1))
    # # print(str1.rstrip())
    # # print(len(str2))
    # # print(str2)
    # str2 = str1.strip('12')
    # print(str2)
    # # name = "guo"
    # # age = "18"
    # # d = dict(name='a', age=18, t='t')
    # print("-----------------dict()的使用------------------")
    #
    # # 1. 创建空字典
    # dd = dict()
    # # 2. 直接用元组元素创建字典
    # d1 = dict(name="guo", age=18, sex="male")
    # d = {'name': 'guo', 'age': 18, 'sex': 'male'}
    # d11 = dict(d)
    # d111 = dict(**d)
    # print("d1=", d1)
    # print("d11=", d11)
    # # 3. 使用zip()元组形式，映射函数方式
    # str1 = zip(["name", "age"], ["guo", 18])
    # for name, age in str1:
    #     print(name, age)
    # d2 = dict(((name, age) for name, age in str1))
    # print("d2=", d2)
    # # 4. 使用嵌套列表方式
    # str = [["name", "guo"], ["age", 18]]
    # d3 = dict(str)
    # print("d3=", d3)
    # # 5. 使用可迭代对象方式.
    # iter = [("name", "guo"), ("age", 18), ("sex", "male")]
    # d4 = dict()
    # for k, v in iter:
    #     d4[k] = v
    # d5 = dict(iter)
    # print("d4=", d4)
    # print("d5=", d5)
    # # for key in d5: # 得到的是键
    # #     print(d5[key])
    # # print("d[1]=", d5[1])
    # print("-------------------------可迭代对象---------------------------")
    # # 含有__iter__()方法的类就是可迭代对象，比如list, tuple, dict, set
    # from collections.abc import Iterable
    # l = [1, 2, 3, 4]
    # # if isinstance(d5, Iterable):
    # #     print("为可迭代对象")
    # # print(l.__iter__())
    # # for i in d5.__iter__():
    # #     print(i)
    # # print(hasattr(l, 'iter'))
    # # print(hasattr(l, 'iter'))
    # print(next(d5.__iter__()))
    # # a = iter(l)
    # # print(a)
    print("--------------------------ch13_4 锚框部分确实很多-----------------")
    print("+++++++++++++torch.meshgrid+++++++++++++++++")
    # torch.meshgrid()网格部分吧
    a = torch.arange(6)
    print(a)
    b = torch.tensor([7, 8, 9, 10])
    c = torch.tensor([11, 12])
    d = torch.tensor([13, 14, 15, 16, 17])
    e, f, h, i = torch.meshgrid(a, b, c, d, indexing='xy') # indexing有两个选项，ij和xy
    # tensor([[0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5]])
    # tensor([[7, 7, 7, 7, 7, 7],
    #         [8, 8, 8, 8, 8, 8],
    #         [9, 9, 9, 9, 9, 9],
    #         [10, 10, 10, 10, 10, 10]])
    print(e.shape) # shape都是[6, 4].第一个做的是a上的重复，。那这个怎么能叫网格呢
    print(f.shape)
    print(h.shape)
    print(i.shape)
    # 2. torch.cat(),torch.stack()。
    print("+++++++++++++++++cat, stack()++++++++++++++++++++++++")
    a2 = torch.arange(3)
    b2 = torch.arange(3, 6)
    aa2 = torch.arange(7,9)
    print(a2)
    print(b2)
    c2 = torch.cat([a2, b2, aa2], dim=0)
    print("c2", c2)
    d2 = torch.stack((a2, b2), dim=0) # 要求两个张量维度必须一致
    e2 = torch.stack((a2, b2), dim=1) # dim=1 dim=0是按行堆叠
    print("d2=", d2)
    print("e2=", e2)
    # 二维张量
    a22 = torch.arange(24).reshape(4, 6)
    b22 = torch.ones(24).reshape(4, 6)
    print(a22)
    print(b22)
    c22 = torch.cat((a22, b22))
    c222 = torch.cat((a22, b22),dim=1)
    print("c22=", c22.shape)
    print("c222=", c222.shape)
    d22 = torch.stack((a22, b22))
    e22 = torch.stack((a22, b22), dim=1)
    f22 = torch.stack((a22, b22), dim=2)
    print("d22=", d22)
    print("e22=", e22)
    print(d22.shape)
    print(e22.shape)
    print(f22.shape)
    # dim=0表示按行拼接，则行增加维度。按dim=1拼接，则列增加维度。默认dim=0
    # repeat重复
    rt = torch.arange(8).reshape(2, 4)
    zz = rt.repeat((3, 4))
    rr = rt.repeat((2, 3, 4)) # 行数为3.后面的维度数量等于输入维度的数量。行数乘3, 列数乘4。因此形状为[2, 6, 116]
    print(zz)
    print(rr)
    print(zz.repeat((2, 1, 1)) == rr)
    # print(r.shape)
    # a = torch.tensor([1, 2, 3])
    # print(a.repeat((3, 2)))
    # torch.clamp()
    print("+++++++++++++clamp++++++++++++++++")
    # clamp的意思是对tensor进行限幅。取值在[min, max]之间，如果大于max则替换为max,如果小于min则取值为min
    cl = torch.randint(2, 7, [6])
    print(cl)
    acl = cl.clamp(max=4) # 大于4者变成4,小于4的时候不变
    print(acl)
    acl2 = cl.clamp(min=4) # 小于4的变成4,大于4的不变
    print(acl2)
    acl3 = cl.clamp(3, 5) # 值只能在3-5之间（闭），大于5置5,小于3置3
    print(acl3)
    print("+++++++++++++++max++++++++++++++++++")
    mt = torch.randint(2, 48, [6])
    print(mt)
    # 不指定维度，那么只返回最大值
    values = torch.max(mt)
    print(values)
    # 指定维度
    values, index = torch.max(mt, dim=0) # 一维，则只能指定dim=0
    print("一维", "\n", "values={}, index={}".format(values, index))
    # 二维
    mt = torch.randint(0, 48, [4, 6], dtype=torch.float32)
    print("二维矩阵：", mt)
    print("不指定维度", mt.max())
    values, index = torch.max(mt, dim=0, keepdim=True)# 按行求最大值，则行消失，也就是说对没列元素中所有行取最大值.和以下结果类似啊
    print(torch.sum(mt, dim=0))
    print(torch.mean(mt, dim=0))
    print("二维", "\n", "values={}, index={}".format(values, index))
    print("++++++++++++++torch.full()++++++++++++++++++")
    f = torch.full((2, 3), -1, dtype=torch.float32)
    print(f)
    print("++++++++++++++++torch.nonzero()+++++++++++++")
    mat = torch.rand([3, 4])
    sc = torch.rand(7)
    print(mat)
    print(sc)
    # print(mat >= 0.5)
    sn = torch.nonzero(sc >= 0.5)
    mn = torch.nonzero(mat >= 0.5)
    print(mat >= 0)
    print("返回非零元素", mat[mat>=0])
    print(sn)
    print(mn)
    print("+++++++++++++++三维linear()+++++++++++++++++++++++++++")
    X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    linear = torch.nn.Linear(4, 5)
    Y1 = linear(X[0])
    Y2 = linear(X[1])
    Y = linear(X)
    print(Y1)
    print(Y2)
    print(Y)
    # print(Y.shape)








