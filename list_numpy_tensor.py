"""
Numpy, torch.tensor, list数据类型转换，总是忘记
2023.9.1 Created by G.tj
"""
import torch
import numpy as np

if __name__ == "__main__":
    print("-----------------------------1.list转numpy.array-------------------")
    x = list(range(10))
    print(x)
    y = np.array(x)
    print(y)
    print(type(y))
    # 如果是嵌套列表呢？能做到吗
    # x2 = [x, x*10] # x*10并不是对其中元素做乘法，而是元素的扩充
    x2 = [x, [a*10 for a in x]]
    y2 = np.array(x2)
    print(type(y2), type(y2[0])) #内外都是array数组
    print(y2)
    print("-----------------------------2.numpy.array转list--------------------")
    x1 = np.arange(10)
    # x1 = np.ndarray([2, 3, 4]) #这是创建了一个这个形状的数组啊
    # x1 = np.array([2, 3, 4]) #这是才是创建[1 2 3]的矩阵
    print(x1.shape)
    print(type(x1))
    print(x1.tolist()) #转成list
    x2 = np.random.randint(0, 20, (2, 10))
    # Return random integers from `low` (inclusive) to `high` (exclusive).左闭右开，随机整数，并不需要每个数字都要兼顾
    print(x2)
    print("----------------------------3.list转torch.tensor--------------------")
    x1 = [x for x in range(10)]
    y1 = torch.tensor(x1)
    print(y1)
    print(type(y1))
    print("----------------------------4.torch.tensor转list--------------------")
    x2 = torch.arange(10)
    # x21 = torch.range(0, 10)
    # UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).
    # range是双闭的，不符合python主流，故建议用torch.arange(start, end)而range()则常用于创建迭代对象
    print(x2)
    # print(x21)
    print(x2.tolist()) #转换
    # 尝试下二维
    x22 = torch.arange(12, dtype=torch.float32).reshape(-1, 2)
    print(x22)
    y22 = x22.tolist()
    print(x22.tolist())
    print(type(y22[0][0]))
    print("---------------------------5.numpy转torch.tensor--------------------------")
    x5 = np.random.randint(0, 20, (2, 3, 10))
    print(x5.shape)
    print(x5.size) #元素个数
    print(x5.ndim) #有几个维矩阵
    y5 = torch.from_numpy(x5)
    print(y5)
    print(y5[0].type()) # torch.LongTensor
    print("--------------------------6.torch.tensor转numpy----------------------------")
    x6 = torch.arange(6, dtype=torch.int8)
    print(x6)
    print(x6.type()) # torch.CharTensor
    y6 = x6.numpy()
    print(y6)
    print(y6.dtype) #int8
    print(type(y6)) #<class 'numpy.ndarray'>
    print(torch.arange(0, 10, 2))
