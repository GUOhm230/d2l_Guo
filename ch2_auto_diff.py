"""
李沐动手学深度学习---自动微分
2023.7.11
主要问题：
1.
"""
import math

import torch
import time
import numpy as np
from d2l import torch as d2l
# 自动微分实现的一些例子
def auto_diff():
    x = torch.arange(4.0)
    x.requires_grad_(True) #其后的计算都会被追踪，且能保存其梯度
    print(x)
    y = 2*torch.dot(x, x)#
    print(y)
    y.backward() #计算当前x下的所有梯度值，并保存在auto_
    print(x.grad)


# 时间计算的封装
class TimerGuo:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器，并将时间记录在列表中，返回当前耗时"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times)/len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist() #按列累加


# 向量化加速
def max_speedUp():
    n  = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    timer = TimerGuo() # 该类创建即包含初始时间
    start = time.time()
    for i in range(n):
        c[i] = a[i] + b[i]
    end_cur = time.time()
    print("使用循环耗时：", end_cur-start)
    print("使用循环耗时：", timer.stop())
    c = a+b
    end_mat = time.time()
    print("使用矩阵计算耗时：", end_mat-end_cur) # 时间减少1000倍


# 计算正态分布, mu是均值，sigma是方差
def normal(x, mu, sigma):
    p = 1/math.sqrt(2* math.pi * sigma*2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

if __name__ == "__main__":
    # auto_diff()
    # max_speedUp()
    # a = [1, 2, 3]
    # b = np.array(a)
    # print(type(b))
    # c = b.cumsum() # 按列累加 c = [b[0], b[0]+b[1], b[0]+b[1]+b[2]]
    # d = c.tolist()
    # print(type(c))
    # print(type(d))
    # print(d)
    # 可视化正态分布
    x = np.arange(-7, 7, 0.01)
    print(x)
    print(len(x))
    params = [(0, 1), (0, 2), (3, 1)]
    # 关于图像的绘制也会是以后的重点，但不是现在
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel="x", ylabel="p(x)", figsize=(4.5, 2.5),
             legend=[f"mean{mu}, std{sigma}" for mu, sigma in params])
    d2l.plt.show()