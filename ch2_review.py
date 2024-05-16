"""
主要用于复习，并做好相应的笔记
2023.7.31
本处的复习阶段，不再照抄代码，而是自己写
"""
import torch
import numpy as np

if __name__ == "__main__":
    X = torch.arange(48).reshape( 3, 4, -1)
    print(X.shape)
    print(len(X))
    print("维度", X.dim())
    A = torch.arange(12, dtype=torch.float32).reshape(3, -1) # 要加dtype才行，不然出错的
    B = A.sum(axis=1, keepdim=True) # 按列求和，消列维度，则之后维度为[3,]， 其实为一维张量，故而无法进行广播机制
    A_l2 = torch.linalg.norm(A)
    print("求范数：", A_l2)
    print(B)
    print(len(B))
    print(B.shape) #维度为torch.size([3])
    C = torch.ones(1, 3) #注意向量和矩阵的区别
    print(C)
    print(C.dim())
    print("----------------------8.1 关于梯度形状---------------------------")
    x = torch.arange(4, dtype=torch.float, requires_grad=True)
    print(x)
    y = x * x # 其导数为2x=[0, 2, 4, 6]
    z = torch.dot(x, x)
    print("z=", z)
    print(y)
    y.sum().backward()
    print(x.grad) # 对吧梯度明明是行向量，书上咋说是个列向量===>torch上显示为行向量，但是数学上向量默认为列向量。
    print("--------------------------8.1 向量运算：dot, matmual以及广播机制-------------------------------------")
    # 本段其实搞不明白的是广播机制，以及dot，matmual计算时对维度的要求
    # dot是元素的乘积，然后累加：测试一下
    # 两个向量：列向量以及行向量
    x = torch.arange(4)
    print("x尝试转置：", x.reshape(-1, 1)) # 但是是做了升维工作
    y = torch.arange(5)
    mm = torch.arange(12).reshape(3, 4)
    print("行向量点积：", torch.dot(x, x)) # 两个行向量，相同位置乘积再相加
    print("一维张量(向量)x的形状={}，x.dim={}".format(x.shape, x.dim())) # torch.Size([4])
    # print("向量反转：", x.T==x) # 一维张量做这个操作是无用的，X.T==X
    # print("列向量点积：", torch.dot(x.T, x.T)) # 实则并非列向量，仍然是一维。也就是向量无法使用转置
    # print("列向量以及行向量之间做点积：", torch.dot(x.T, x))
    # print("矩阵点积：", torch.dot(mm, mm)) # 该项操作实际上是不成立的，也就是说对于矩阵是不支持使用点积操作，会报维度错误
    # 因此，真正要实现列向量，那就必须使用二维的
    x = torch.arange(4).reshape(4, -1)
    print("二维张量(矩阵)x的形状={}，x.dim={}".format(x.shape, x.dim())) # torch.Size([4，1])
    # print("能否做点积，我猜测是不行的：", torch.dot(x, x)) #果然报错
    """
    dot总结：
    torch.dot只针对一维向量，是相同位置的乘积，而不针对矩阵，矩阵是不行的
    在torch中一维向量是无法转置的，表示形式就是行相信，X.T没有转置功能
    该功能只对二维矩阵有效
    """
    print("--------------------torch.matmul()操作-------------------------")
    # 看看torch.matmul()的操作
    # torch.matmul矩阵运算
    x = torch.arange(4)
    y = torch.arange(4)
    print(torch.dot(x, y).sum())
    print("1. 向量之间的matmul：", torch.matmul(x, y)) # 可以做向量内积
    x = torch.arange(12).reshape(3, 4)
    y = torch.arange(12).reshape(4, 3)
    print("二维矩阵之间的matmul：", torch.matmul(x, y)) # 二维矩阵乘法，要满足数学规定 3*4 4*3
    print(torch.mm(x, y))
    z = torch.arange(4).reshape(-1, 1)
    print("二维矩阵之间的matmul：", torch.matmul(x, z), torch.matmul(x, z).shape) # 二维矩阵乘法，要满足数学规定 3*4 4*1
    x_1d = torch.arange(4)
    print("矩阵与向量之间的matmul:", torch.matmul(x, x_1d), torch.matmul(x, x_1d).shape) # 二维矩阵与向量乘法，维度为3*4 4.而不能向量在前
    yy = torch.tensor([torch.matmul(x1, x_1d) for x1 in x])
    print("yy:", yy==torch.matmul(x, x_1d))
    x_3d = torch.arange(24).reshape(2, 3, 4)
    y_3d = torch.arange(24).reshape(2, 4, 3)
    print(x_3d)
    print(y_3d)
    print("多维矩阵的matmul", torch.matmul(x_3d, y_3d), torch.matmul(x_3d, y_3d).shape) # 2*3*4 2*4*3==2*3*3.
    x_3d_2 = torch.arange(12, 24).reshape(3, 4)
    y_3d_2 = torch.arange(0, 12).reshape(4, 3)
    # print(x_3d_2)
    # print(x_3d[1, :, :])
    # print(y_3d_2)
    # print(y_3d[0])
    print("三维测试第二个维度：", torch.matmul(y_3d_2, x_3d_2))
    y_3d_differentShape0 = torch.arange(24).reshape(-1, 4, 3) # 第一维要么相等，要么为1,使用广播机制，不然都会报维度不匹配的错误，该维度当做batchsize.维度看2,3维
    print("三维维度测试：", x_3d.shape, y_3d_differentShape0.shape)
    print("三维矩阵matmul，同时第一个维度形状不同：", torch.matmul(y_3d_differentShape0, x_3d), torch.matmul(y_3d_differentShape0, x_3d).shape)
    print("三维矩阵matmul，同时第一个维度形状不同：", torch.matmul(x_3d, y_3d_differentShape0), torch.matmul(x_3d, y_3d_differentShape0).shape)
    print("三维矩阵与二维矩阵的matmul：", torch.matmul(x_3d, y_3d_2), torch.matmul(x_3d, y_3d_2).shape)

    # 接下来测试关于线性方程计算的广播机制 y = Xw + b
    x = torch.normal(0, 0.1, size=(5, 2))
    w = torch.tensor([2, -3.4, ])
    b = torch.tensor([4.2])
    print(b)
    y = torch.matmul(x, w) + b# torch中依然是把y当做行向量。因为是个一维张量，也就是向量，不是矩阵.所以标量和一维向量都能与向量做广播机制，可以不用同样的维度
    print(y) #
    print(y.shape)


