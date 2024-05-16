"""
优化计算
1. 梯度下降
2. 随机梯度下降
3. 小批量梯度下降
4. 动量法
5. AdaGrad算法
6. AdaGrad变体之：RMSprop
7. AdaGrad变体之：Adadelta
8. 以上算法综合之：Adam
主要是把上面这些优化算法用代码给实现了，包括从零开始实现，以及简洁实现。
各类方法有什么优缺点，为什么要这么做替换
2023.10.9 Created by G.tj
"""
import torch
from torch import nn
import numpy as np
from d2l import torch as d2l
import math
def train_2d(trainer, steps=20, f_grad=None):
    # 返回迭代步之后的权重参数,输入：更新算法也就是优化器，迭代次数（相当于学习率函数中的t）, f_grad是当前参数下的梯度值计算函数
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    return results

def show_trace_2d(f, results):
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), color='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.xlabel('x2')

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n , 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

# 目标函数，深度学习中一般是损失函数
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 **2

# 输入变量值，返回当前变量值下的梯度
def f_2d_grad(x1, x2):
    return (2 * x1, 4 * x2)

# 参数更新，输入当前里变量值，返回更新后的参数值
def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

# 牛顿法:以双曲余弦函数举例
def f(x):
    return torch.cosh(c * x) # 双曲余弦函数

def f_grad(x):
    return c * torch.sinh(c * x)

def f_hess(x):# 目标函数的黑塞矩阵
    return c**2 * torch.cosh(c*x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)/f_hess(x) #黑塞矩阵：为目标函数的二阶导数
        results.append(float(x))
    return results

# 随机梯度下降
# 目标函数
def f(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1, ))
    g2 += torch.normal(0.0, 1, (1, ))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

# 常数学习率
def constant_lr():
    return 1

# 动态学习率: 分段恒定常数以及指数衰减和多项式衰减.这里的t是迭代次数
# 指数衰减
def exponential_lr():
    global t
    t += 1
    return math.exp(-0.1 * t)

# 多项式衰减
def polynomial_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return (1+0.1*t) ** (-0.5)

# 小批量随机梯度下降
# 数据获取
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'), dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0)) #
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True) #数据打包成可迭代对象, 用于训练
    return data_iter, data.shape[1]-1

def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

# 梯度下降从零开始实现
def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    # 初始化模型参数
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss # 这个用法创建模型还是挺好玩的
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter), (d2l.evaluate_loss(net, data_iter, loss), ))
                timer.start()
    return timer.cumsum(), animator.Y[0]

# 梯度下降的简洁实现
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 创建模型，其中包含了参数的初始化，也能自己再做初始化.
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if  type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # 创建解释器
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter), (d2l.evaluate_loss(net, data_iter, loss) / 2, ))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(sgd, None, {'lr':lr}, data_iter, feature_dim, num_epochs)

# 动量法
def f_2d(x1, x2):
    return 0.1 * x1**2 + 2 * x2**2

# 梯度下降更新参数
def gd_2d(x1, x2, s1, s2):
    return(x1-eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# 动量法更新参数:本处没有用矩阵法
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 0.2 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

def momentum_mat(x, v):
    v = beta * v + x.grad
    return x -eta * v

# 从零开始实现动量法,
# vt = beta*vt-1 + gt（其中v的初始值为0, 故能从v1开始算起是没错的）。由此可知V.shape=p.grad.shape=p.shape.因此初始化状态的时候注意对准维度
def init_momentum_state(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

# 动量法实现参数更新，更新的时候先计算v再计算p,之所以用v[:]目的是不开辟新的内存
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_state(feature_dim), {'lr': lr, 'momentum':momentum}, data_iter, feature_dim, num_epochs)

def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

# 从零实现AdaGrad算法
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

# 从零开始实现RMSProp算法
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma *s1 + (1-gamma) * g1 ** 2
    s2 = gamma *s2 + (1-gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def init_rmsprpop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1-gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s+eps)
        p.grad.data.zero_()

# adadelta算法的从零实现
def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            s[:] = rho * s + (1-rho) * torch.square(p.grad) # square()返回的是张量的平方
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1-rho) * g *g
        p.grad.data.zero_()

# 从零开始实现Adam算法
def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w)), ((v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1-beta1) * p.grad
            s[:] = beta2 * s + (1-beta2) * torch.square(p.grad)
            v_bias_corr = v / (1-beta1 ** hyperparams['t'])
            s_bias_corr = s / (1-beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr' ] * v_bias_corr / (torch.sqrt(s_bias_corr + eps))
        p.grad.data.zero_()
    hyperparams['t'] += 1

# Yogi改进思路：其实就是增加了一种思路
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)+eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

# 学习率调度器的使用
# 定义网络模型
def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    return model

def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

if __name__ == "__main__":
    print("------------------------1. 梯度下降-----------------------------")
    """
    做法：梯度下降的主要计算在于当前多元函数的梯度， 所有数据求的梯度后做平均即为本次迭代的梯度
    参数更新：x = x-学习率*x.grad
    什么时候用：任何时候都能用， 因为负梯度是函数值下降最快的方向。
    缺陷：计算量太大，每次更新都要对所有数据做操作，而且可能数据量大小不符合GPU计算的批次规律，但是总得来说一次更新时间较短
    """
    # eta = 0.1
    # show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
    # d2l.plt.show()
    # 自适应方法，也就是可以自动调整学习率的方法
    # 牛顿法，在原公式下除以一个黑塞矩阵即可
    c = torch.tensor(0.5)
    # show_trace(newton(), f)
    # d2l.plt.show()
    # 一种改进牛顿法的思路是不取黑塞矩阵，而是取黑塞矩阵的对角线，计算减了很多
    print("--------------------------------2. 随机梯度下降------------------------------------")
    lr = constant_lr
    # d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad)) # 本处的sgd是上一个
    # d2l.plt.show()
    print("--------------------------------3. 小批量梯度随机梯度下降-----------------------------")
    # 优势就是比随机梯度下降计算更快，有可能在时间上还优于梯度下降。且不像随机梯度下降那样导致瞬时梯度而左右摇摆，不能收敛，或者收敛很慢。
    # 有关torch.sub()操作
    # a = torch.tensor((1, 2))
    # b = torch.tensor((0, 1))
    # c = torch.sub(a, b, alpha=2)
    # # 上面这个代码，输入是a和b,输出是c：其中c=a-alpha*b
    # print(c)
    # # c = a.sub(b, alpha=2)
    # d = a.sub_(b, alpha=2)
    # # a.sub_(b, alpha=2) # 在a的基础上进行修改
    # print(d)
    # print(a)
    # 小批量随机梯度下降
    # 设置batch_size=1500 也就是全数据梯度下降
    # gd_res = train_sgd(1, 1500, 10)
    # d2l.plt.show()
    # mini1_res = train_sgd(0.4, 100, 10)
    # d2l.plt.show()
    # # 梯度下降的简洁实现
    # data_iter, _ = get_data_ch11(10)
    # trainer = torch.optim.SGD
    # train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
    print("--------------------------------3. 动量法---------------------------")
    # 动量法用过去梯度的加权平均值来代替当前迭代下的梯度，更新方法是一致的
    # eta, beta = 0.6, 0.25
    # # 测试一下动量法
    # # d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
    # # d2l.plt.show()
    # print(0.9 ** 11)
    # # 内存开辟与否
    # a = torch.arange(12)
    # print(a)
    # print(id(a))
    # a = eta * a
    # print(a)
    # print(id(a))
    # a[:] = eta * a
    # print(id(a))
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # train_momentum(0.02, 0.9)
    # d2l.plt.show()
    # # 简洁实现
    # trainer = torch.optim.SGD
    # d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
    print("-------------------------------4. AdaGrad------------------------------")
    # 该优化算法适用于稀疏特征。因为常见特征会较快的收敛，而不常见特征则收敛较慢。因此加入了特征功能项约束
    # 从零实现
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # d2l.train_ch11(adagrad, init_adagrad_states(feature_dim), {'lr': 0.1}, data_iter, feature_dim)
    #
    # # 简洁实现
    # trainer = torch.optim.Adagrad
    # d2l.train_concise_ch11(trainer, {'lr':0.1}, data_iter)
    # d2l.plt.show()
    print("----------------------5. RMSProp算法----------------------------")
    # 该算法其实是上一个的变体:添加了类似动量法的泄露平均值，又沿用了adagrad中以gt平方对下降较为缓慢的稀疏特征进行约束，使其学习率可以相应增大
    # 从零开始实现之
    # eta, gamma = 0.4, 0.9
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # d2l.train_ch11(rmsprop, init_rmsprpop_states(feature_dim), {'lr':0.01, 'gamma':0.9}, data_iter, feature_dim)
    # # 简洁实现
    # trainer = torch.optim.RMSprop
    # d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)
    # d2l.plt.show()
    print("------------------------------6. Adadelta算法-----------------------------------")
    # 该算法也是adagrad的一种变体：该算法的主要特征是没有学习率参数
    # 从零开始实现
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # d2l.train_ch11(adadelta, init_adadelta_states(feature_dim), {'rho': 0.9}, data_iter, feature_dim)
    # # 简洁实现
    # trainer = torch.optim.Adadelta
    # d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
    print("--------------------------------7. Adam算法-------------------------------------")
    # 该算法其实综合了多种算法的优势：泄露平均值，取gt平方
    # 从零实现
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # d2l.train_ch11(adam, init_adam_states(feature_dim), {'lr': 0.01, 't':1}, data_iter, feature_dim)
    # # 简洁实现
    # trainer = torch.optim.Adam
    # d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
    #
    # data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    # d2l.train_ch11(yogi, init_adam_states(feature_dim), {'lr': 0.01, 't': 1}, data_iter, feature_dim)

    print("--------------------------------8. 学习率调度器----------------------------------------")
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    loss = nn.CrossEntropyLoss()
    device = d2l.try_gpu()

    lr, num_epochs = 0.3, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
    # d2l.plt.show()

    # 设置学习率调度器‘
    lr = 0.1
    trainer.param_groups[0]["lr"] = lr # optimizer.param_groups[0]：是一个字典，一般包括[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]等参数（不同的优化器包含的可能略有不同，而且还可以额外人为添加键值对

    print("trainer.param_groups：", trainer.param_groups[0]["lr"])
    """
    本阶段还有两个需要完成：
    一： 优化算法的总结
    1. 优化算法就是参数更新的算法。其主要理论是凸函数优化理论，极小值朝着负梯度方向行进
    2. 梯度下降，随机梯度下降和小批量梯度下降是常见的下降算法。但是梯度下降计算量大，随机梯度下降耗费时间太长，且随机性太强。小批量梯度下降则综合了两者的优点：
    optimizer = torch.optim.SGD(net.parameters(), **hyperparams),其中hyperparams包含了学习率以及其他算法中用到的超参数
    3. 针对一些嘈杂梯度，也就是参数之间相差过大，导致一个降低缓慢，一个降低较快。此时用小批量梯度下降则会发生震荡。用动量法效果更好，且不增加额外的时间复杂度。
    但是要增加保存梯度和动量（状态）的内存，从而会增加空间复杂度。动量法用的不是当前迭代时的梯度，而是过去至当前所有梯度的加权平均值（这个权重就是人为设置的超参数）
    官方API：optimizer = torch.optim.SGD(net.parameters(), **hyperparams)其中hyperparams中包含：学习率以及动量momentum
    4. 针对一些稀疏特征，就是说常见特征迭代较慢而稀疏特征迭代较快的变量在更新参数时，步骤不一致。因此需要加上观察到的特征数。而在实际使用的时候，
    则用平方和来约束稀疏特征，使其更新加快，而对于梯度本身较小者会更加平滑。故而AdaGrad算法能在单个层面降低学习率，他利用了梯度的大小作为调整进度速率的手段
    官方API：optimizer = torch.optim.Adagrad(net.parameters(), **hyperparams),超参数中只有学习率
    5. 作为AdaGrad的变体之一，RMSProp增加了泄露平均值，以解决AdaGrad中st也就是状态量持续增长的问题。于是应用了泄露平均值。同样的，该算法使用的是梯度平方，不增加额外的计算。
    但是要增加保存状态量和历史梯度的内存空间
    官方API：optimizer = torch.optim.RMSprop(net.parameters(), **hyperparams),其中超参数包括：学习率和alpha作为泄露平均值的参数
    6. Adadelta是Adagrad的另一个变体。该算法减少了adagrad减少了学习率适应坐标的数量。该算法同样维持一个st状态量，用于计算过去梯度的加权平方值。该算法还增加了一个梯度的重新缩放，
    同时维持了一个梯度缩放的泄露平均值的状态delta.
    要注意的是，这些状态是对于每个权重都有的，故而w, b各要建立一个矩阵用于存储
    该算法是没有学习率的，而是用delta，st以及eps进行缩放
    官方API：optimizer = torch.optim.Adadelta(net.parameters(), **hyperparams),其中超参数只有一个参数：rho,delta以及xt两个状态量用同一个
    7. 作为集大成者的adam算法几乎集中了前几种算法的所有公式。几乎集合了所有的优势
    官方API：optimizer = torch.optim.Adam(net.parameters(), **hyperparams), 该超参数只有学习率一项
    二次矩可能爆炸导致无法收敛，因此Yogi算法进行了一次更新。其实就是改变了st状态量的计算
    8. 关于学习率的调整有几种方式：常用的是恒定学习率，还可以做动态学习率：分段恒定常熟，指数衰减和多项式衰减。其实比如adagrad, rmsprop, adadelta这种算法也相当于改变了学习率
    
    二： 还有学习率调度器还没有完成
    
    """
