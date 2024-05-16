"""
第八章：循环神经网络
本章了解序列模型
了解怎么预处理数据
2023.7.22 created by G.TJ
"""
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import collections
import re
import random
import math

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) #

def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 使用全连接层进行模型训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            l = loss(net(x), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch{epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

# 8.1当中的测试代码，搞清楚序列模型和几步预测
def ser81Test():
    # 1. 序列模型的简单实现：数据x维度[600, 4]其中600为数据个数，相当于样本数，4为特征数。而这些数据则均来自于一个时间序列.因此可以实现一个全连接网络，来得到该自回归模型
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.1, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()
    tau = 4  # 马尔可夫模型指示的是需要几个时间序列点进行预测下一个时间序列
    features = torch.zeros(T - tau, tau)  # 其中维度为[996, 4],从时间序列提取而得
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    batch_size, n_train = 16, 600
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    loss = nn.MSELoss(reduction='none')
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)
    # 2. 有关时间步
    # 使用单步预测：前4个预测后一个，因此预测后面的每一个时间点时，都要前面的所有时间序列均预测完毕方可
    onestep_preds = net(features)  # 用600的数据训练得到的参数，来预测后面400个时间序列点，事实证明结果是不错的,而且预测的时候使用的是原始的时间序列数据
    d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
             legend=['data', '1-step preds'], xlim=[1, 1000],
             figsize=(6, 3))
    d2l.plt.show()
    # 使用k步预测，而且使用的训练数据是自己预测的.
    # 使用自己的预测数据来预测接下来的数据
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]  # 前604个用原始序列,后面根据预测值得到
    print()
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(multistep_preds[i - tau: i].reshape((1, -1)))
    print(multistep_preds.shape)
    d2l.plot([time, time[tau:], time[n_train + tau:]], [x.detach().numpy(), onestep_preds.detach().numpy(),
                                                        multistep_preds[n_train + tau:].detach().numpy()], 'time', 'x',
             legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000],
             figsize=(6, 3))
    d2l.plt.show()

    # 基于k=1, 4, 16, 64
    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))  # 维度1000-4-64+1, 68.用68个数去预测后一个？
    for i in range(tau):
        features[:, i] = x[i:i + T - tau - max_steps + 1]
        # tensor的切片，即使是某一列，实际也表示为行,因此这样赋值是没有问题的。该处功能就是给0, 1, 2, 3列进行赋值
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
             [features[:, (tau + i - 1)].detach().numpy() for i in steps],
             'time', 'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
    d2l.plt.show()


# 8.2文本数据预处理
# 文本下载
def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] # re应该是正则表达式.正则模式为括号内，表示非这些字母则替换为空格，要替换的是line这些内容

# 然后需要对这些文档进行词元化：可以以单词为词元，也能以字符为词元
def tokenize(lines, token='word'):
    if token == 'word':
        print("当前行内容：", lines[0])
        return [line.split() for line in lines] # 每行维持一个嵌套列表，列表内每个元素代表每一行的内容，每行内容又是个列表，子列表内每个元素是个单词，用空格分开
    elif token == 'char':
        print("当前行：", lines[0])
        return [list(line) for line in lines] #每行都维持一个嵌套列表，第二层列表为改行的所有字符, 没有以空格分开，因此还会出现空格在的位置
    else:
        print("错误：未知词元类型：" + token)


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line] # 在列表中的两层嵌套能这样使用的呀。学到了
        print("换成单层嵌套tokens：", tokens)
    return collections.Counter(tokens) # 返回一个字典，得到某个词元出现的频数

# 做词表
class Vocab:
    # 文本词表
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 统计文本集中词元出现频率并按之排序。这个地方的主要学习点其实就是一些API的使用
        counter = count_corpus(tokens) # Counter({'the': 2261, 'i': 1267, 'and': 1245, 'of': 1155, 'a': 816, 'to': 695, 'was': 552, 'in': 541, 'that': 443, 'my': 440, 'it': 437, 'had': 354,

        print("词的排序counter：", counter, "\n", counter.items())
        self._token_freqs = sorted(counter.items(), key=lambda x:x [1], reverse=True) # 按照值进行排序， 从大到小
        print("排序后_token_freqs：", self._token_freqs)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        print("初始idx_:", self.idx_to_token)
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}
        print("初始token_:", self.token_to_idx)
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token) # 为列表，元素应该是不重复的字符吧
                self.token_to_idx[token] = len(self.idx_to_token)-1 # 同样的是个字典, 词元：索引。词元就是语料库中的词，索引就是根据频率排序获得
        print("该类中的主要内容：", self.token_to_idx)

    def __len__(self): # 就是语料库的长度.
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk) # 如果token只是一个值，则获取对应的值，也就是索引即可
        return [self.__getitem__(token) for token in tokens]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

# 以上功能整合
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# 8.3节：搞清楚下怎么做的抽样检查和顺序分区
# 随机抽样
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 获取
    corpus = corpus[random.randint(0, num_steps-1):]
    num_subseqs = (len(corpus)-1) // num_steps # 计算能有多少个子序列
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    print("initial_indices:", initial_indices)
    random.shuffle(initial_indices)
    print(initial_indices)

    #获取时间步长为5的连续子序列
    def data(pos):
        return corpus[pos : pos+num_steps]

    num_batches = num_subseqs // batch_size # 计算能有几个小批量样本
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 顺序分区：这里的代码有时间还需要自己读一下
def seq_data_iter_sequential(corpus, batch_size, num_steps):

    offset = random.randint(0, num_steps) #设置初始偏移量
    num_tokens = ((len(corpus)-offset-1) // batch_size) * batch_size #需要用到的词元数
    Xs = torch.tensor(corpus[offset: offset+num_tokens]) # 设置了batch_size=2 指的是每个样本中包含两个长度为5的子序列。设置小批量大小为2指的是以2个2*5的矩阵为一组
    Ys = torch.tensor(corpus[offset+1: offset+1+num_tokens])

    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps # 批量数
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i:i+num_steps]
        yield X, Y

# 模型参数初始化。也就是步骤1里面的那几个
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01 # 这样会不会不太稳定

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q] #这个写法主要是方便后面做梯度下降，参数更新
    for param in params:
        param.requires_grad_(True)
    return params

# 设置初始隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh)+torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq)  + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )

# 组装一下
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn #分别为初始隐状态函数名以及rnn前向计算方法名

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params) # 此时调用的run函数

    def begin_state(self, batch_size, device):# 返回初始状态
        return self.init_state(batch_size, self.num_hiddens, device)

# 字符预测
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_snum_hiddenstate(batch_size=1, device=device) # 构造初始隐状态
    outputs = [vocab[prefix[0]]] # prefix是待预测的前几个字符
    # print("vocab[]:", prefix[0], vocab[prefix[0]])
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state) # state就是下一个时间步的H，也就是隐状态
        outputs.append(vocab[y]) # outputs记录的其实就是当前的时间步对应的数据，这个时候不是做预测
    for _ in range(num_preds): # 预测num_preds时间步
        y, state = net(get_input(), state) # 此处进入预测，以及下一步预测的隐状态计算了
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 梯度截断
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和，词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:# 随机初始化，针对每个batch_size的数据都进行一次初始化
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else: # state已经有了初始值，且为顺序分区.此时下一个小批量的第一个样本的隐状态是上一个小批量最后一个样本的隐状态。此时。因此该隐状态需要做梯度分离，不然会累积梯度
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater,  torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size)
        metric.add(l*y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", ylabel="perplexity", legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl: .1f}, {speed:.1f} 词元/秒 {str(device)}')
    predict('time traveller')
    predict('traveller')

# 借用nn.Module来定义模型
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init_forward_fn_(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) #只有隐藏层，没有输出层
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))


# 初始化模型参数
def get_params_gru(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three() # 更新门参数
    W_xr, W_hr, b_r = three() # 重置门参数
    W_xh, W_hh, b_h = three() # 候选隐状态
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 定义gru模型
# 定义初始隐状态
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 定义模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1-Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 初始化参数——lstm
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three() # 输入门参数
    W_xf, W_hf, b_f = three() # 遗忘门参数
    W_xo, W_ho, b_o = three() # 输出门参数
    W_xc, W_hc, b_c = three() # 候选记忆元参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 设置初始隐状态
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

# 前向计算，其实和之前的RNN以及GRU方法是一模一样的，只是多一点内容
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X@W_xi) + (H@W_hi) + b_i)
        F = torch.sigmoid((X@W_xf) + (H@W_hf) + b_f)
        O = torch.sigmoid((X@W_xo) + (H@W_ho) + b_o)
        C_tilda = torch.tanh((X@W_xc) + (H@W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

if __name__=="__main__":
    print("------------------------------1. 序列模型的测试--------------------------------------")
    # 该处主要是搞清楚序列模型中，数据是什么？搞清楚时间步是什么，何为单步预测，何为多步预测。时间步X1,X2,...,Xt也相当于随机过程，
    # 其中的每个Xt都是一种分布，包含很多样本空间。而之前的MLP还是其他的，都只满足一种分布，相当于MLP中的X与Xt相同
    """
    我了个去，终于是搞定了这个问题
    明白了k步预测到底是个啥玩意
    k步预测，当前时间序列的第K步之后，而这K步，是需要用自己的预测数据去做预测，在做模型训练的时候，仍然和上面类似
    本来是一个简单的问题，这本书却给了障眼法。差别就是用原始数据预测还是用预测的数据去做预测。如果是一步预测，就是用原始的数据去预测下一个
    如果是K步预测，那么后面就是用自己预测的数据去做预vocab测，就这么简单，理解了半天
    """
    # ser81Test()
    print("--------------------------------2. 文本预处理----------------------------------------")
    # 该处主要是知道语言模型中，数据的组织形式，如何处理这些数据：获取文本--文本拆分成词元--文本数字化--随机抽样以及顺序分区--以及为什么要是这样，其实和后面的RNN训练有关
    # 1. 获取文档数据
    # # 下载文本
    # d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55d93cb0a180b9691891a')
    # lines = read_time_machine() # 按行存取
    # print(f'#文本总行数：{len(lines)}')
    # print(lines[0])
    # print(lines[10])
    # print(type(lines))
    # 2. 词元化
    # 所谓词元化，就是把文本按照设定的词元进行切分，并数字化，然后独热编码传入模型中
    # tokens = tokenize(lines, 'char')
    # print(tokens)
    # for i in range(11):
    #     print(tokens[i])
    # print(len(tokens)) # 文本行数为3221
    # # ss = ["guohuiming hhaha", "hkajkajakjl"]
    # # print([list(ss1) for ss1 in ss])
    # print("建立词表")
    # 3. 建立词表：即统计文本集中的词频，并按照词频高低进行排序
    # vocab = Vocab(tokens)
    # print(vocab.token_to_idx)
    # print(vocab.idx_to_token)
    # print(len(vocab.idx_to_token))# 语料库中有4580个字
    # print(type(vocab))
    # print(vocab["the"])
    # print("功能整合")
    # corpus, vocab = load_corpus_time_machine()
    # print(len(corpus), len(vocab))
    # print(vocab.idx_to_token) # 28个：多了空格和未知词元
    # # print(len([vocab[token] for line in tokens for token in line])) # 怎么会有类似于字典这种用法呢？而vocab只是个对象啊.__getitem这个方法啊，很有效的
    """
    该部分的要学习的内容其实更多是列表，字典的处理
    接下来进行第本章第三节的学习，学习语言模型和数据集
    主要搞清楚下什么是随机抽样和顺序分区.搞清楚了
    """
    # lines = read_time_machine()
    # tokens = tokenize(lines)
    # corpus = [token for line in tokens for token in line]
    # vocab = Vocab(corpus)    net = RNNModel(rnn_layer, )

    # print(vocab)
    print("---------------------------------8.3 随机抽样和顺序分区-----------------------------------------")
    # my_seq = list(range(35))
    # print(my_seq)
    # seq_data_iter_random(my_seq, batch_size=2, num_steps=5)
    # for x, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    #     print(x, y)
    # seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5)
    # for x, y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    #     print(x, y)
    print("------------------------------------8.4 循环神经网络-------------------------------------------------")
    # 该处主要还是学会怎么处理循环神经网络.该处的重点并不是一些python常用API。内容其实不难，因此要快
    # 1. 非常简单的实现隐状态的一次计算
    # X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
    # H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
    # Y = torch.matmul(X, W_xh) + torch.matmul(H, W_hh) # Ht = f(X*W_Xh + Ht-1*W_hh + b)
    # print(Y)
    # Y1 = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
    # # 按照列拼接，则列想加的维度相加，行的维度相等。同理，按照行拼接，则要保证列相等，增加的是行
    # print(Y1)
    #
    #
    # # 2. 数据准备
    # batch_size, num_steps = 32, 35
    # train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    #
    # # 以上其实只是做了数据抽样一起数字化，接下来做独热编码
    # print("vocab的长度：", len(vocab)) # 长度为28
    # print("就是想看看vocab中的内容：", vocab.token_to_idx) # 维持一个字典，该字典为该语料库。知道28咋来的了，一个是unk未知字符，一个是空格
    # print("就是想看看vocab中的内容：", vocab.idx_to_token) # 语料库内容。28个字符
    # F.one_hot(torch.tensor([0, 2]), len(vocab))
    # X = torch.arange(10).reshape(2, 5) # 设置batch_Size=2, num_steps=5.每个时间步上的数据为5.然后利用独热编码，将一个数字扩展成len(vocab）
    # # print(X)
    # X_onehot = F.one_hot(X.T, 28)
    # # print(X_onehot) # 维度：5, 2, 28
    # # 3. 初始化模型参数
    # # 4. 设置初始隐状态
    # # 5. 定义rnn运算
    # # 6. class整合
    # # 测试一下输出，以上其实只完成了前向计算
    # num_hiddens = 512
    # net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    # state = net.begin_state(X.shape[0], d2l.try_gpu())
    # Y, new_state = net(X.to(d2l.try_gpu()), state)
    # print(Y.shape, len(new_state), new_state[0].shape) #其实这里有个重点，还是关于维度的计算，笔记中搞得比较清楚
    # print("vocab:", vocab[1])
    # print(predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu()))
    # 接下来是要搞清楚是怎么完成预测的
    """
    循环神经网络的预测需要求得两个东西才行：
    1）当前时间步的隐状态
    2）根据上一时间步的隐状态以及当前时间步的隐状态求的当前时间步的预测值
    而给定的例子中，由于没有经过训练，于是之前所以时间步的隐状态都不是很清楚因此要进入预热期求的所有的中间时间步的隐状态、
    代码中迭代使用了lambda表达式确实也很巧妙：get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    预热期结束后就是根据要预测的时间步进行逐一计算。其实计算结果还是和之前是一样的
    针对每一个输入，都有两个输出：当前时间步的隐状态以及当前时间步的输出。获得隐状态以用于下一个时间步的预测，得到Y进行字符转化。获取结果
    """
    # 接下来是如何进行训练
    # 7. 梯度截断，防止梯度消失以及梯度爆炸
    # 8. 开始训练,先抄了代码，再解析
    # num_epochs, lr = 500, 1
    # train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu()) #该处的代码不是很熟。但是先这样过来吧，毕竟非重点了
    print("-------------------------------------8.6 简洁实现------------------------------------------")
    # # 1. 数据
    # batch_size, num_steps = 32, 35
    # train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # # 2. 定义模型
    # num_hiddens = 256
    # rnn_layer = nn.RNN(len(vocab), num_hiddens) # 定义了一个循环网络层
    # state = torch.zeros((1, batch_size, num_hiddens)) # 单隐藏层，故而前面数字为1
    # print(state.shape) #其维度：[隐藏层数，批量大小，隐藏单元数]
    # X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    # Y, state_new = rnn_layer(X, state)
    # print("state_new: ", state_new.shape)
    # # print(Y.shape, state_new.shape)
    # device = d2l.try_gpu()
    # net = RNNModel(rnn_layer, vocab_size=len(vocab))
    # net = net.to(device)
    # d2l.predict_ch8('time traveller', 10, net, vocab, device)
    # num_epochs, lr = 500, 1
    # d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    # """关于循环神经网络的知识，不需要细究， 如果要的话，那就第二遍再说"""
    print("----------------------9.1 看看现代循环神经网络-----------------")
    # GRU的从零实现
    # 1. 数据准备
    # batch_size, num_steps = 32, 35
    # train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # # 4. 训练
    # vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    # num_epochs, lr = 500, 1
    # model = d2l.RNNModelScratch(len(vocab),num_hiddens, device, get_params_gru, init_gru_state, gru)
    # d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    print("-------------------------9.2 LSTM-----------------------------")
    # lstm的从零实现
    # 1. 数据获取
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # # 4. 开始训练以及预测
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    # model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
    # d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    # lstm的简洁实现
    num_inputs = vocab_size
    # lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    # model = d2l.RNNModel(lstm_layer, len(vocab))
    # d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    print("------------------------9.3 深度循环神经网络-------------------------")
    # 1. 数据准备同上
    # 直接调用官方API。自己从零实现，太麻烦了。而且也就是多了几层中间隐状态的计算就是。本质是一样的
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    # lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers) # 多层就是加了一个层数，这里只有RNN的隐藏循环层，并没有输出层。输出要加上nn.Linear()
    # model = d2l.RNNModel(lstm_layer, len(vocab))
    # model = model.to(device)
    # d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    print("--------------------------9.4 双向循环神经网络-------------------------")
    # 其实同样，也就是改了一个lstm的创建
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    """
    循环神经网络的总结：
    1. 有关序列模型，其中序列数据如何使用，并使用线性模型进行训练
    2. 有关条件概率，分布函数，全概率公式，马尔可夫模型，马尔可夫链。
    3. 有关时间步的预测，单步预测，多步预测。数据读取方式：随机抽样或者顺序分区
    4. 循环神经网络的计算公式：
        1）计算隐状态：H = nn.tanh(X@W_xh + Ht-1 @ W_hh) + b_h
        2）计算当前时间步的输出：Y = H @ W_hq +b_q
        注意：X的维度为(num_steps, batch_size, N)。时间步放前面是方便预热期的时候求出每个时间步的隐状态
    5. 训练的时候其实主要也就是。
        1）数据制作：获取文本--->文本词元化--->够造语料库--->词元数字化--->独热编码
        2）初始化隐状态：[隐藏层数， batch_size, 隐藏单元数]:torch.zeros(shape, device)
        3）模型参数初始化.就是公式里面的W，b进行赋值。注意设置梯度可求
        4）循环神经网络的前向计算
        5）组合成类，方便计算。其实就是包含三个方面：隐状态初始化，前向计算
        6）梯度截断以防止梯度消失和梯度爆炸
        7）进行时间步预测：预热期求出当前隐状态。然后循环之后时间步进行预测
        8）模型训练：区分随机抽样和顺序分区时计算梯度的差别。随机抽样针对每个小批量都要进行随机初始化隐状态。而顺序分区则不用，由于其相邻小批量上数据上的连续性
        接着训练流程就和之前的内容一致了。我觉得难点就是在这个地方。初始隐状态的分区
    6. 关于GRU，LSTM以及深度RNN和双向RNN则不做过多总结
        1）GRU涉及更新门，重置门，以及之后候选隐状态。因此他的数据量较大。参数初始化较多
        2）LSTM及其他同理
        3）使用官方API的时候注意rnn_layer=nn.gru()（nn.rnn(), nn.lstm()）等都是计算隐藏层信息。要后面输出信息，还需要加入nn.Linear()
    7. 本章内容还有不熟悉的地方：比如训练的时候，为什么对初始隐状态要进行这样的区分。训练中的一些语法，困惑度是什么？如何计算困惑度。具体内容需要深入的话后续继续
    """









