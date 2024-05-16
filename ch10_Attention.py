"""
来到非常非常非常的重点，也是自己一直不知道的内容----->注意力机制
这个内容好像是针对序列化时间数据的，现在搬到图像领域
主要问题：
1. 我现在都不知道怎么做的注意力汇聚
2. 数据如何
2023.9.18 Created by G.tj
"""
import torch
from d2l import torch as d2l
from torch import nn
import math

# 注意力的可视化，可视化注意力权重.该处主要是绘图工具
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1] # batch_size, 通道？
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    # 针对每个batch_size进行循环
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        # 当前batch_size的通道
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            print("pcm", pcm)
            if i == num_rows-1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def f(x):
    return 2 * torch.sin(x) + x**0.8

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True)) # 设置的权重，就是个标量，如此设置是为了可以做广播

    def forward(self, queries, keys, values):
        print("训练中：", queries.shape, keys.shape, values.shape) # q:[50, ], keys: [50, 49], values:[50, 49]
        # 对输入的单个查询值，其实输入肯定是矩阵，也肯定是批量处理的
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1])) #[50, 49]
        print("变化后的Q：", queries)
        print("keys:", keys)
        self.attentions_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1) # 按照哪个求，哪个维度消失，因为在行上是相同的x与不同的
        print("self.attention", self.attentions_weights.shape)
        print("values:s", values.shape)
        return torch.bmm(self.attentions_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# 掩蔽softmax操作. 其实输入是一个权重矩阵和该权重矩阵对应的有效长度
# 实际为什么做的，也不清楚啊
def masked_softmax(X, valid_lens):
    # 也就是没有指定无用词元，故而直接运算
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1: # valid_lens是个一维向量
            valid_lens = torch.repeat_interleave(valid_lens, shape[1]) # 如此重复，那么到底是针对哪个维度进行判定呢？针对的是第三个维度，因为第三个维度的信息是查询数为i的词元与键数为j的词元所做的评分函数，如果是某个词元无校，也就是该词元是无效的
            print("mask中的valid_lens:", valid_lens)
        else:
            valid_lens = valid_lens.reshape(-1)
        # print("X.reshape():", X.reshape(-1, shape[-1]))
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 输入的X一般都是评分函数，要使最后的注意力权重为0，则要设置一个很小的数字才行
        print("mask中的X：", X)
        print("mask2:", nn.functional.softmax(X.reshape(shape), dim=2))
        return nn.functional.softmax(X.reshape(shape), dim=-1) # dim=-1就是最后一个维度了，这里的维度

# 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys) #由于需要的维度不一样，所以需要线性回归进行维度扩展
        print("queries.shape, keys.shape=", queries.shape, keys.shape)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        print("features.shape=", features.shape)
        print("self.W_v(features).shape=", self.W_v(features).shape)
        scores = self.W_v(features).squeeze(-1)
        print("scores.shape=", scores.shape)
        self.attention_weights = masked_softmax(scores, valid_lens)
        print("attention.shape=", self.attention_weights.shape)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 缩放点积注意力。输入的是查询，键值对以及有效长度三项。
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d) #
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 定义通用注意力机制解码器
class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# Bahdanau注意力：定义序列到序列的注意力机制解码器:
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        # 因为普通的解码器是比较简单的，
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # 当前解码器层的上下文变量。之前的解码器层的上下文变量是编码器最后一层隐状态，而其中用注意力机制取代，是没错的
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

# 将q,k,v按照头数进行拆分重组：输入维度：[batch_size, 查询数或者键值对数， 特征长度（经过线性变换后的特征长度numn_hiddens）]。所用到的方法无非就是reshape之后permute做转置，然后再reshape合起来
# 输出维度：[batch_size*num_heads, 查询数或者键值对数，特征长度（num_hiddens）/头数]
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 将输出之后的注意力机制结果按照头数拆分，再回到特征长度上来：输入维度：[batch_size*num_heads, 查询数，特征长度（num_hiddens）/头数]：注意力之后的结果，第二个维度永远是查询数
# 输出维度：[batch_size, 查询数， 特征长度（经过线性变换后的特征长度numn_hiddens）]
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 流程明显：对输入的qkv做线性变换，然后做数据变换，做成num_heads等分的查询，键值数据，接着做缩放点积注意力，明显的缩放点积注意力较为简单，没有这么多复杂的参数
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 有关这个复制的还不是很清楚：现在很清楚了: 因为要按照头数拆分出来，然后把头数乘到batch_size上，因此batch_size扩大，而len(valid_lens)=batch_size, 由于是从相同的查询，键值对上按照头数等份截断的，因此该头的有效与否与原查询是相同的，故而valid_lens需要复制
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            print("valid_lens:", valid_lens)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

#位置编码
class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000): # 设置个最大值进行操作，后面直接取值的方式好像做过好几次了
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

if __name__ == "__main__":
    # 书本上是说显示一个简单的例子，我反正感觉是没有意义的，无非就是给这个单位矩阵画出来了而已. 9.18补充：但是实际上这个矩阵就是注意力权重
    # at
    print("-------------------1. 汇聚的实例-------------------------")
    # 1. 创建数据集（x, y）
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5) #看样子第二个参数是返回其索引的.和list的sort()
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)
    x_test = torch.arange(0, 5, 0.1)
    # 1. 平均汇聚：f(x) = y_train.mean(), 这是一个标量
    y_hat = torch.repeat_interleave(y_train.mean(), n_test) # y(x)的值是一样的,该函数的意思就是把y_train.mean()这个标量展平
    # print(y_hat)
    # print(y_train.mean())reshape
    # plot_kernel_reg(y_hat)
    # d2l.plt.show()
    # print(x_test.shape)
    # x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # print(x_repeat.shape)
    # 测试一下repeat_interleave这个函数的意思：官方没看到文档
    ri = torch.tensor([[1, 2], [3, 4]])
    print(ri.repeat_interleave(2, 0))
    print(torch.repeat_interleave(ri, 2, 0)) # dim=0:不展平，按行重复，则是行数增加，将该矩阵中每行元素看做一个整体，新的行=原的行*2
    print(torch.repeat_interleave(ri, torch.tensor([2, 1]), 1)) # dim=0:不展平，按列重复，则是列数增加，将该矩阵中每行元素看做一个整体，新的列=原的列*2
    # 2. 非参注意力汇聚：加入x_train中的数据
    x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train)) #
    print("x_repeat:", x_repeat)
    print("x_train:", x_train)
    # 按照列求，则列消失
    attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1) # x_repeat是待预测的新x表示查询值，x_train是键.x的要与x_train中每一项都计算评分函数，
    print(attention_weights.shape) # x_test中有50个数据，其中每个数据都要与x_train中每个数据一一求得权重，因此attention_weights维度=[50, 50].其中每行是单个x对所有x_train的权重
    y_hat = torch.matmul(attention_weights, y_train)

    # plot_kernel_reg(y_hat) #相对平滑一些，但是还是差距有点大
    # d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted train inputs',
    #                   ylabel='Sorted testing inputs')
    # d2l.plt.show()
    # 接下来一个比较重要的点，就是unsqueeze是啥意思，一直不是很清楚
    print("------------------------torch.unsqueeze()和squeeze--------------------------")
    # unsqueze()是添加维度的，squeeze是减维度的
    X = torch.tensor([1, 2, 3, 4]).reshape(2, 2)
    Y = X.unsqueeze(1)
    Y1 = torch.unsqueeze(X, -2)

    Z = Y1.squeeze() #
    print(Y.shape)
    print(Y1.shape)
    print(Z.shape)
    print("--------------------------带参数的注意力汇聚-------------------------------")
    # 1. 批量矩阵乘法的计算
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))

    Z = torch.bmm(X, Y)
    print(Z.shape)
    print(Z)
    print(Y.type())
    print(type(Y))
    # 测试，X， Y为三维
    X = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    print(torch.bmm(X, Y))
    # print(torch.arange(8).reshape(2, 1, 4))
    # 测试X，Y为四维
    # X = torch.arange(8).reshape(2, 4)
    # Y = torch.ones(4, 6)
    # torch.bmm(X, Y)
    # print(X, Y)
    # 计算一下, 其实就是最后一步测试一下，使用
    weights = torch.ones((2, 10)) * 0.1
    values = torch.arange(20.0).reshape((2, 10))
    fx = torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
    # 整体走一下
    print(x_train.shape)
    X_tile = x_train.repeat((n_train, 1))
    print("X_tile:", X_tile)
    # 如果输入的是一个tensor向量，这里repeat依然把他当做[1, len(a)]处理，因此行数*3,就是添加[1, 2, 3]两行， 列数*2, 就是列上再加1, 2, 3
    # a = torch.tensor([1, 2, 3])
    # print(a.repeat((3, 2)))
    # print(x_tile.shape)
    # rt = torch.arange(8).reshape(2, 4)
    # print(rt)
    # z = rt.repeat((2))
    # zz = rt.repeat((2, 2, 2))]
    # print(z)
    # print(zz)
    Y_tile = y_train.repeat((n_train, 1))
    # 去除和本身进行的计算，故而每个train应该对应49个
    keys = X_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # [50, 49] 因为要训练，故而X_train中数据也要作为查询的
    print("keys:", keys)
    values = Y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) #同理呀，values也做了同样的处理
    # print((1-torch.eye(n_train)).type(torch.bool)) #这个可以直接做数据类型转换呢
    # print(values.shape)
    # X = torch.arange(16).reshape(4, -1)
    # print(X)
    # print(X[torch.tensor([[1, 2], [1, 2]])])
    # v = X[(1-torch.eye(X.shape[0])).type(torch.bool)]
    # print(v)
    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
    # 训练流程并没有差别
    # for epoch in range(5):
    #     trainer.zero_grad()
    #     l = loss(net(x_train, keys, values), y_train)
    #     l.sum().backward()
    #     trainer.step()
    #     print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    #     animator.add(epoch+1, float(l.sum()))
    # keys = x_train.repeat((n_test, 1))
    # values = y_train.repeat((n_test, 1))
    # y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    # plot_kernel_reg(y_hat)
    # d2l.show_heatmaps(net.attentions_weights.unsqueeze(0).unsqueeze(0),
    #                   xlabel='Sorted training inputs',
    #                   ylabel='Sorted testing inputs')
    # d2l.plt.show()
    # d2l.plt.show()
    # 带参数的好像确实不错
    """
    NW核的阶段总结：
    1. 本处由于是注意力机制的起始，因此也有关注的必要。看完后面的，其实也就是抛砖引玉。只是思路的借鉴，后面做注意力机制时候，并没有用到注意力权重后的参数
    2. 本处从预测入手，提出三种根据数据对（x, y）预测新x的方法:1) 只根据y值得平均值进行预测，2）加入x引入NW核方法，3）引入参数
    3. 但是本处的关键是计算权重参数。他把一个新的待预测的x同训练集中的每个x做核计算，得到权重参数，然后这些权重参数与y做线性加权运算得到最后的结果。
    4. 其实本处还需要学的就是如何进行矩阵运算：包括torch.repeat, torch.repeat_interleave()的使用，以及tensor.type()的使用
    5. 其实该处还是很多不是很清楚，看过后面回头再来瞧瞧吧
    """
    print("----------------------------看看注意力评分函数-------------------------")
    # 掩蔽softmax操作：有些词元没有意义, 因此要把相关的键值对取0，也就是其对应的注意力评分函数相应位置置0.也就是输入的是注意力评分函数，输出的是
    Y = masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
    print("softmax掩蔽结果：", Y)
    # 测试一下softmax函数的维度，就是dim是如何选定的
    X = torch.arange(3, dtype=torch.float32).reshape(-1, 3)
    Y = nn.functional.softmax(X, dim=1)
    print(Y)
    print("维度求和：", X.sum())
    # 一维的向量
    # def softmax(X):
    #     # print(torch.exp(X).sum()) # 他也是张量，只要用的还是torch去计算的
    #     # print(torch.exp(X).sum().shape)
    #     return torch.exp(X)/torch.exp(X).sum(dim=1, keepdim=True) # 设置的维度必须要两个维度，dim
    # print(softmax(X))
    # 如果是二维向量

    # X = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    # print(X)
    # Y = nn.functional.softmax(X, dim=1) #自定义是按列进行
    # print("二维：", Y)
    # print("自定义：", softmax(X))
    # # 维度到底是咋回事
    # X1 = torch.arange(3, dtype=torch.float32)
    # X12 = torch.arange(3, 6, dtype=torch.float32)
    # print(softmax(X1))
    # print(softmax(X12))
    # 例1：使用tensor向量
    # Z = torch.arange(12)
    # print(Z[None, :].shape) # 果然用于增加维度了，但是实际没有作为复制
    # print(Z[:, None].shape)
    # print(Z[None, :, None].shape)
    # 试试大于号，小于号的广播机制。同样适用于
    # X = torch.arange(4)[None, :]
    # Y = torch.tensor([0, 2, 0, 3])[:, None]
    # mask = X < Y
    # print(mask)
    
    # 加性注意力，主要关注其维度，知道原理，却不知道数据是怎样构成的。数据构成已经知道了
    # queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    # valid_lens = torch.tensor([2, 6])
    # attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    # attention.eval()
    # attention(queries, keys, values, valid_lens)
    # 测试一下squeeze(-1)
    # z = torch.zeros((2, 1, 2))
    # print(z.squeeze().shape)
    # # 测试一下三维进行全连接的计算. 证明了我的猜想, nn.Linear() =
    # X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # X2 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # X3 = torch.arange(12, 24, dtype=torch.float32).reshape(3, 4)
    # net = nn.Linear(4, 2)
    # print(net.weight.data)
    # Y = net(X)
    # Y2 = net(X2)
    # Y3 = net(X3)
    # print("Y=", Y)
    # print("Y2=", Y2)
    # print("Y3=", Y3)
    # Z = torch.matmul(X, net.weight.data.T) + net.bias.data # 该方法的运算
    # print(Z==Y)
    # 有关tensor的转置，permute, transpose
    X = torch.arange(12).reshape(3, 4)
    print(X.permute(0,1))
    print(X.transpose(1,0)) # 0, 1和1, 0的结果是一致的, 给出两个维度，就是在哪两个维度上完成转置
    #  尝试下三个维度的, 最后也是只能两个维度
    X = torch.arange(24).reshape(-1, 3, 4)

    print(X.transpose(0, 1).shape) #不能有三个数字，只能是两个，X.permute(里面则是维度多少个就多少个)
    print(X.transpose(1, 2).shape)
    """
    评分函数部分总结：
    1. 本阶段知道给定一组数据之后，怎么求的注意力权重，并由注意力权重求得最后的预测值
    2. 给定的数据为查询，键，值。这三个数据的来源可以多种多样，不过后面的例子来自于词元，来自于编码或者解码层的输出。其维度是统一的为：[Batch_size, 查询数/键值对数， 特征长度]
    3. 如果查询以及键值对长度不同，则要经过线性回归进行维度变化，这就是加性注意力
    4. 流程为：给定q, k, v--->求的计算评分函数--->根据有效长度进行掩蔽softmax计算注意力权重--->
计算f(x)就是torch.bmm操作
    5. 评分函数就是计算q与k之间的关系，分为加性注意力和缩放点积注意力评分函数后的维度为：[小批量大小，查询数，键值对数]
    6. 加性注意力需要进行线性回归，再做加法，然后再进行线性回归。对查询和键的线性回归是为了将新的查询和键统一维度，方便做连接。具体分为：线性回归，维度加1扩展，然后矩阵加法，再使用线性回归，然后掩蔽softmax计算注意力权重。最后torch.bmm()
    7. 对注意力权重使用了dropout：就是对输入矩阵进行按概率置0的操作，输入维度是什么，输出维度就是什么
    8. 缩放点积注意力就更简单了：torch.bmm(queries, keys.transpose(1, 2))*math.sqrt(queries.shape[2]).之后的操作同上
    """
    # 看看加法的广播机制
    X = torch.arange(3).reshape(-1, 3) # 此处其实dim维度是1和2结果都是一样的
    Y = torch.arange(3).reshape(3, -1)
    Z = torch.arange(3)
    print(X+Y) # 做加法的时候果然自动补全，最后维度变成了(3, 3)
    print(X.dim())
    print(Z.dim())
    print("--------------------Bahdanau注意力--------------------")
    # 这是注意力应用的一个例子，应用于编码器解码器中。
    # 先回顾下：编码器用于将词元序列变成隐状态，再变成上下文变量C。编码器的输出是各层隐状态和最后一层隐状态
    # 解码器是相同长度序列的隐状态生成词元序列。解码器的输入是(X, Y)中的Y,Y与隐状态编码器输出的最后一层隐状态做cat连接，作为GRU网络的输入，然后对输出进行线性回归计算
    # B注意力则为了改变上下文变量的值
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()

    X = torch.zeros((4, 7), dtype=torch.long)
    # print(encoder(X)[1].shape)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
    # 怎么做训练呢？参数来自哪里
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    for  key, value in net.state_dict().items():
        print(key)
    print(net.parameters())
    # d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    print("-----------------------------5.多头注意力---------------------------------")
    # 多头就是把同样的查询，键值对进行线性回归后按照头数进行拆分，然后分别做注意力机制
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    Z = attention(X, Y, Y, valid_lens)
    print(Z.shape)
    # 有关valid_Lens的复制再复习一下，这个知识点本来不是那么重要的，关键是要把python学习的知识点弄清楚一下
    # 有效词元自然是针对词元而设置的，主要是针对键值对，表明当前batch_size数据中有效数据长度是多少，也就是前几个词元是有效的
    print("---------------------------------自注意力--------------------------------")
    # 所谓自注意力机制，其实也能结合多头注意力，用不同的评分函数进行操作。只是输入的东西来自于一组数据
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Z = attention(X, X, X, valid_lens)
    print(Z)
    # 其实本处最难得不是自注意力如何计算，而是卷积神经网络，循环神经网络和自注意力的时间复杂度和空间复杂度
    # 卷积神经网络和自注意力都能做并行计算，也就是不考虑时间步，因为不是根据时间步逐个计算的，即使是按照时间步逐个计算，也会像循环神经网络那样进行隐状态的传递
    # 因此自注意力和卷积并没有利用了词元的时间步信息，故而需要进行位置编码
    encoding_dim, num_steps = 32, 60
    # 果然第十章没有完成
    pos_encoding = PositionEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    # 有关pow的计算，怎么实现的i,j的
    pp = torch.arange(3, dtype=torch.float32).reshape(-1, 1) / torch.pow(2, torch.arange(0, 10, 2, dtype=torch.float32)) # torch.pow(i, j)为i的j次方
    print(torch.pow(2, torch.arange(0, 10, 2, dtype=torch.float32)))
    print(torch.arange(3, dtype=torch.float32).reshape(-1, 1))
    print(pp) #[3, 5]
    print("++++++++++++++++++++++层规范化+++++++++++++++++++++++++")
    # 层规范化和批量归一化
    ln = nn.LayerNorm(2)
    bn = nn.BatchNorm1d(2)
    X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    print(ln(X))
    print(bn(X))






