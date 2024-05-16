"""
有关attention的内容过于重要，再次重复一遍
2023.12.19 Created by G.tj.2024年1.2号才完成。一章看了近一个月，13章也看了近一个月。近3个月效率十分低下。
主要问题：
1. 什么是注意力机制，到底在做一件什么事
2. 注意力计算有哪些常见的方法
3. 输入，输出以及维度变化
4. 怎么做到的并行计算，不区分词源的先后顺序
5. Bahandu注意力
6. 多头注意力
7. self attention。自注意力
8. transformer
"""
import torch
from torch import nn
from d2l import torch as d2l
import math

print("-------------------1. 注意力相关要素---------------------------")
# 注意力三大要素：查询，键，值。注意力机制的主要内容就是做预测，给定键值对(keys, values)，然后预测f(query)的值。
# 那么预测就有几种方法：1. 去values的均值f(query) = mean(values). 这样的话，使用的只有values这个值，因此会缺乏可靠性
# 更进一步则是非参注意力机制了：也就是NW核。该方法把(q, k)的相似度作为断定values的一个标准
# 创建数据: 就是创建q, k, v.q:x_test, k: x_train, v:y_train. q的真实预测值：y_truth
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)

# 做NW核回归计算
x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
print(x_repeat)
# a = torch.tensor([1, 2, 3])
# print(a.repeat_interleave(2))
print(x_train)
print(x_repeat-x_train)
attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1)
# st = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# print(nn.functional.softmax(st, dim=1))
# print(nn.functional.softmax(torch.tensor([0, 1, 2, 3], dtype=torch.float32)))
print(y_train.shape)
print(attention_weights.shape)
y_hat = torch.matmul(attention_weights, y_train)

# 带参数的注意力机制
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, )), requires_grad=True)

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries-keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# 进行模型的训练
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # 这里其实很巧妙啊。由于训练数据的查询和键都是相同数据，因此要除去与自己做相似度计算的部分。
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # 同样的，要删除与自己对应的键的值
net = NWKernelRegression()
print("w:", net.w.data)
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch{epoch + 1}, loss {float(l.sum()) :.6f}')
print("训练后的参数：", net.w.data)
print("x_train:", x_train)
keys = x_train.repeat((n_test, 1))
print("keys:", keys)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
print(y_hat.shape)
"""
第一问总结：
1. 三大要素：查询，键，值
2. 注意力机制的目的就是根据查询预测相应的值，也就是求f(q)的结果。方法就是NW核回归。
3. NW核回归通俗的说就是通过计算查询与键的相似度，然后与对应的值进行加权求和。而这个相似度计算用的是softmax(-1/2(x-xo)^2)
4. NW核回归计算的时候学习一下重复和matmul()的计算形式.softmax()
5. 带参数的NW核回归需要建立的模型参数，该模型参数是一个标量，即使是标量，也能做模型训练。训练时注意数据
6. torch.matmul(x, y)中x,y可以是向量，矩阵或者多维张量。torch.bmm()则必须为三维矩阵
7. 本处简单的，注意输入三项：查询：维度[查询数]，键值对：[键值对数]
8. 相关输出：注意力评分函数，注意力权重：[查询数，键值对数];注意力输出：[查询数]
"""
# x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# attention_weights = nn.functional.softmax(-(x_repeat-x_train)**2, dim=1)
# y_hat = torch.matmul(attention_weights, y_train)
# a = torch.tensor([1, 2, 3])
# b = a.repeat_interleave(2)
# c = a.repeat(2)
# print(b)
# print(c)
# a = torch.arange(12).reshape(3, 4)
# b = torch.tensor([1, 2, 3], dtype=torch.float32)
# print(torch.bmm(a, b))
print("---------------------2. 注意力评分函数------------------------------")
# 本处开始使用词元数据
# 掩蔽softmax操作
# 注意力分数：缩放点积注意力和加性注意力
# 加性注意力：输入的查询和键值对维度可以不同，但是要通过全连接使之相同
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # 输入：queries:[batch_size, 查询数也就是查询词元数，词元特征长度], keys: [batch_size, 键值对数，键值对词元特征长度]
        queries = self.w_q(queries)
        keys = self.w_k(keys) # 将特征长度替换为num_hiddens
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 维度:[batch_size, 查询数，键值对数，num_hiddens]
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weight = nn.functional.softmax(scores, dim=1) # 计算权重啊，。权重维度：[batch_size, 查询数，键值对数]
        print("self.attention_weight:", self.attention_weight.shape)
        return torch.bmm(self.dropout(self.attention_weight), values) #[batch_size, 查询数，键值对] @ [batch_size, 键值对, 值得特征长度] = [batch_size, 查询数, 值的特征长度]

# 测试一下
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# w_k = nn.Linear(2, 8, bias=False)
# w_q = nn.Linear(20, 8, bias=False)
# w_v = nn.Linear(8, 1, bias=False)
# queries, keys = w_q(queries), w_k(keys) # 将特征长度替换为num_hiddens
# features = queries.unsqueeze(2) + keys.unsqueeze(1) # 维度:[batch_size, 查询数，键值对数，num_hiddens]
# features = torch.tanh(features) # 维度保持不变
# print(features.shape)
# scores = w_v(features).squeeze(-1) # 维度：[batch_size, 查询数，键值对数，1],squeeze(-1)之后压缩了最后一个维度为1的维度，最后[batch_size, 查询数，键值对数]。懂了为什么要变成1了，因为评分函数得到的结果就是这样的
# print(scores.shape)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1) # 维度[2, 10, 4]
# print(values.shape)
addAttention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
add_y_hat = addAttention(queries, keys, values)
print("加性注意力输出结果：", add_y_hat)
# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=1)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 测试一下
queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
dot_y_hat = attention(queries, keys, values)
print(dot_y_hat.shape)

# 掩蔽softmax操作搞一下， 所谓掩蔽softmax操作，就是当一些键盘的词元无效时，把这个评分函数置0
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape # [batch_size, 查询数， 键值对]
        if valid_lens.dim() == 1: # [batch_size]
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

X = torch.arange(24)
print(X)
print(X.dim())
"""
注意力评分函数：
一句话概括就是：求得查询与键值对中键的相似度系数。故而维度为[batch_size, 查询数, 键值对]
两种方法：加性注意力，缩放点积注意力
区别：加性注意力不要求查询与键值对的特征长度可以不一样。但是缩放点积注意力则要求两者的特征长度相同
输出：[batch_size，查询数，值的特征长度]
"""

print("----------------------------3. 不同的注意力机制-----------------------------------")
print("++++++++++++1. Bahdanhu++++++++++")
# Bahdanhu注意力，目的就是用注意力机制改进编码-解码机制改进编码解码问题进行改进
class AttentionEncoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionEncoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# 这个代码还是要仔细分析一下
class Seq2SeqAttentionDecoder(AttentionEncoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0., **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        # print("输入X：", X.shape)
        # print("hidden_state：", hidden_state[-1].shape)
        outputs, self._attention_weights = [], []
        # 确实是按照时间步进行的循环.但是时间循环
        for x in X:
            # print(x.shape)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            # print("out:", out)
            # print("out:", hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property # 以属性的形式输出
    def attention_weights(self):
        return self._attention_weights

# 测试一下Bahdanau注意力
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long) # batch_size, num_steps
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
# 本段时间，主要是完全搞明白Bahdanau注意力的流程和代码。其实之前弄清楚好几遍了，都忘了.一个绝对时间过去，已经完全明白
"""
Bahdanau注意力
step1: 通过编码器计算结果，而编码器则是一个rnn网络。输出为RNN网络的隐状态，作为解码器的初始隐状态也是解码器的初始查询（有关rnn网络需要后面复习一下还）。
step2: 编码器中所有时间步最后一层的隐状态值作为初始的键值对。编码器最后一个时间步最终层隐状态作为初始的查询。也就是key, value = encoder(X)[0]，query = encoder(X)[1]
step3: 注意力输出和输入嵌入（也就是embeding(X)）的连接作为rnn也就是解码器的输入
step4: 解码器的输入是以时间步作为循环的，也就是RNN的计算是以时间步作为循环的。第一个输出为：当前时间步最后一个隐藏层的输出，第二个输出为当前时间步所有隐藏层的输出=hidden_state作为下次时间步的查询
"""
# 接下来放入整体中训练进行结果查看
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
# net = d2l.EncoderDecoder(encoder, decoder)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 本段时间，主要是搞清楚训练流程以及多头注意力机制。
print("+++++++++++++++2. 多头注意力++++++++++++++++")
# 多头注意力机制比较简单。就是把数据按照特征长度，分配成不同的组合，然后用一个线性
# 对qu
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 确实是截断最里面的数据，也就是说截断维度最深的
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])
X = torch.arange(36).reshape(2, 3, 6)
Y = X.reshape(X.shape[0], X.shape[1], 3, -1)
Z = transpose_qkv(X, 2) # 维度[4, 3, 3]
print(X)
# print(Y)
print(Z)

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
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
"""
多头注意力的总结
何为多头: 就是把原来的qkv按照特征长度分成不同的头，然后对这些头进行注意力机制求解
在注意力机制之前使用全连接以保证数据查询和键的特征长度相似，同时又能变换不同头的查询和键值，使得计算结果可靠
"""

print("++++++++++++++++3. 自注意力机制与位置编码+++++++++++++++++++++")
# 本段时间把自注意力机制搞定，位置编码。以及transformer搞定
# 所谓自注意力就是查询，键值都来源于一组数据。至于注意力机制的方式可以是多头注意力，也可以是普通的注意力机制。而评分函数可以使用加性注意力也可以使用点积注意力
# 至于代码是前两天写过的，就暂时不做了
print("---------------------------------4. Transformer-----------------------------------")
# 关于transformer很不清楚。主要有以下问题需要解决
#1. 基于位置的前馈网络不就是两个全连接嘛？为什么叫基于位置的前馈网络？
#2. 位置编码不是应该在编码器解码器之前使用嘛？怎么之前没有看到相应的使用信息呢？最终到底是怎么使用的？:使用方法是没错的
#3. transformer是通过注意力机制完成编码解码的过程。那有隐状态嘛？输入输出是啥？实际意义是什么呢？
# 还有问题需要一个一个解决。后面要放到面试里面的
# 上午一个时间主要搞清楚第一个问题
# 1. 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 编写一下编码器，这个编码器中使用了基于位置的前馈网络.此处为一层编码器
# 输入-->多头注意力-->残差层和层规范化-->基于位置的前馈网络-->残差连接和层规范化
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# blks = nn.Sequential()
# #
# for i in range(2):
#     blks.add_module("block"+str(i), EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, False))
# blk = blks[0]
# print("blk:", blk)
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # print("输入：", X[0]) # [64, 10]: batch_size, num_steps
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens)) # 这一步是为啥呢？为啥要乘以嵌入维度的平方。
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights # blk是一个EncoderBlock，这个EncoderBlock.attention是多头注意力，blk.attention.attention是缩放点积注意力，blk.attention.attention.attention_weights是缩放点积注意力的注意力权重
        return X

# 一个解码器：是一个解码器层
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout) # 掩蔽多头注意力
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens) # 基于位置的前馈网络
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            print("None")
            key_values = X
        else:
            print("非None")
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        # print("内部：", state[0])
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 掩蔽多头自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        print("掩蔽多头自注意力：", X2.shape)
        Y = self.addnorm1(X, X2)
        # 编码器--解码器注意力。其中query=Y, keys=enc_outputs, values=enc_outputs
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        print("编码器-解码器注意力：", Y2.shape)
        Z = self.addnorm2(Y, Y2)
        print("逐位前馈网络：", self.ffn(Z).shape)
        return self.addnorm3(Z, self.ffn(Z)), state

# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
# valid_lens = torch.tensor([3, 2])
# decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
# decoder_blk.eval()
# X = torch.ones((2, 100, 24)) # [batch_size, ]
# state = [encoder_blk(X, valid_lens), valid_lens, [None]] # 这么来看，state确实没有变化。state不变
# dec_output, state = decoder_blk(X, state)
# print("dec_output:", dec_output.shape) # [2, 100, 24]
# 构建Transformer的解码器
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        print("解码器输出：", X.shape)
        print("最终输出：", self.dense(X).shape)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# 训练
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32,  32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 本段时间主要学习任务就是搞清楚编码器的隐状态是啥？就是注意力输出嘛？因为没有用RNN呀。关于rnn需要进一步学习一下
# 本段时间主要搞清楚transformer的每个阶段的输入与输出
# 本段时间完全在线，从transformer全局搞清楚其中的每个输入与输出：确实搞清楚了
# 本段时间，transformer的收尾工作
# 本段时间，transformer最后收尾。并开始做笔记。不看任何无关的东西了
# 本段时间，已经完成了Transformer的收尾工作。继续啊。总结，写面试经验
# 又一天了，昨天忙的时候还是很不错的。但是我妈去医院检查，还是担心的。晚上又和肖瑞琪聊了好久西游记。
"""
Transformer总结：
1. Transformer主要是编码器-解码器模型的实例。用于文本序列的学习。输入序列，输出的也是预测的序列。后面的应用，其实还要看论文
2. Transformer的主要工具件就是注意力机制。而这个注意力包括多头注意力和自注意力
3. 数据包装后还要经过嵌入维度修整和位置编码。
4. 编码器的输入是[batch_size, num_steps],虽然是时间步，但是却不像RNN那样根据时间步，依次执行。而注意力机制可以并行计算。因为只用的是矩阵计算
5. 编码器的输出是[batch_size, num_steps, 第二个注意力机制的value_size].由于后面需要做加法和层规范化，因此这个value_size等于编码器
输入做embedding后的第三维度。
6. 编码器的输出作为解码器的输入，这个输入用于解码器中的第二个多头注意力机制中的kv键值对
7. 解码器的输入有两个：1. 训练数据中的真实输出词元（翻译任务中英语-法语对中的英语作为编码器的输入，法语作为解码器的输入
也相当于训练数据对（X，Y），X作为编码器的输入，Y作为编码器的输入）。2. 编码器层的输出与掩蔽信息作为初始隐状态输入到解码器中
8. 解码器中组件包含：注意力机制包括：掩蔽多头注意力和编码器-解码器注意力，逐位前馈网络，加法与层规范化
9. 由于有加法，因此解码器输出：[batch_size, num_steps, embedding_size].
10. 最后通过一个全连接网络，输出为[batch_size, num_steps, vocab_size]。这样得到的每个元素就是对应词元，对应位置的编码
最后通过前面的语料库进行重组，得到预测的词元
Transformer完结
"""


