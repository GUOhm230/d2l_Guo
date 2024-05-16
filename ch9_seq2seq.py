"""
本章的点其实很重要：序列到序列的学习
本章目标：
1. 什么是序列到序列的深度学习
2. 如何做到呢
3. 怎么实现的编码器和解码器：先是怎么做到通用实现
4. 待定
Created by G.tj 2023.9.15
"""
from torch import nn
import collections
import math
import torch
from d2l import torch as d2l

# 创建通用编码器：实现从输入序列转化成包含输入序列的状态
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

# 解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# 合并编码器和解码器
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# 编码器:
class Seq2seqEncoder(Encoder):
    """用于序列到序列学习循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2seqEncoder, self).__init__(**kwargs)
        # 嵌入层，二维输入数据进行维度扩展，增加一个特征向量维。怎么计算的暂时先不管
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X) # 作为真实的输入数据，至于什么时候用这个，为什么要用，其实就把他当做one-hot()之后，
        X = X.permute(1, 0, 2)
        output, state =  self.rnn(X)
        return output, state

# 用普通的深度RNN实例化解码器
class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层，数据维度扩展，这里输入的还是X，和编码器类似嘛
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这里是输入数据和编码器中获取的上下文变量（本文中为隐状态）进行拼接，故而维度增加，由于初始隐状态等于编码器的最终隐状态，因此，输出维度是一样的
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 循环神经网络层输出隐状态，再用全连接进行结果的输出
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # 初始化隐状态：编码器的最终隐状态作为解码器的初始隐状态
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] # output, state = encoder(X) # enc_outputs[1] = state

    def forward(self, X, state):
        # 使用的输入数据还是编码器中的输入时间，进行维度扩展操作
        # 关键是输入数据是怎么做的
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1) # 原本的state是[2, 4, 16],context是[7,  4, 16]
        print("context:", context.shape)
        X_and_context = torch.cat((X, context), 2)
        print("X_and_context:", X_and_context.shape)
        print("state:", state.shape)
        output, state = self.rnn(X_and_context, state) #初始隐状态可有可无，如果没有的话，在GRU类中的forward中初始化为0了都
        output = self.dense(output).permute(1, 0, 2) #计算隐状态之后的输出
        print("解码器的输出：", output.shape)
        return output, state


if __name__ == "__main__":
    # 接下来半小时要完全的在状态：做到了，奖励50
    #给个数据测试一下, 搞懂两个东西：nn.Embedding(vocab_size, embed_size)， 以及X.permute(X)
    # encoder = Seq2seqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)
    # Y = X.permute(1, 0, 2)
    # print(Y)
    # output, state = encoder(X)
    # 1. 测试nn.Embedding(vocab_size, embed_size)
    # 首先给定X的用法是
    embedding = nn.Embedding(10, 3)
    print(embedding.weight.data.shape)
    print(embedding.parameters())
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]]) # 原本的[2, 4]变成了现在的[2, 4, 3]
    Y = embedding(input)
    print(Y.shape)
    # print(stateY)
    """
    故：nn.Embedding(n, m)的用法就是给X扩充维度，作为模型的实际输入，一般放在模型的第一层
    其运算，可以看做和nn.Linear(n, m)有一样的效果，做了一次线性运算。
    只是nn.Linear(n, m)(X)中的X.shape=[B, n], 这个n在语言模型中=len(vocab)=vocab_size=词表大小。一般是给定当前词元的数字索引，然后经过one-hot编码得到的一个len(vocab)大小的向量
    而nn.Embedding(n, m)(X)中的X则无需做one-hot，X.shape=[T, B],输出维度=Y.shape=[T, B, m]
    也就是输入[时间步, 批量]
    """
    # 2.测试一下X.permute(1, 0, 2)这种转置的还有X.transpose()以后要专门搞一遍
    Z = Y.permute(1, 0, 2)
    print(Z.shape)
    # print(Z) # 结果发现是第一维度以及第二维发生了变化
    x = torch.linspace(1, 24, steps=24).view(2, 3, 4)  # 设置一个三维数组
    # torch.linspace()#返回一个一维步长张量，该张量的长度为steps, 元素为从start到end等间隔
    print(x)
    """
    permute(0, 1, 2)中数字表达的意思
    0=X.shape[0]
    1=X.shape[1]
    2=X.shape[2]
    Y = X.permute(1, 0, 2)意思就是，计算后的矩阵Y.shape如下
    Y.shape[0] = X.shape[1]
    Y.shape[1] = X.shape[0]
    Y.shape[2] = X.shape[2]
    维度知道了，现在是，我该知道怎么做的这个交换的
    做个猜测，以X.permute(2, 0, 1)为例Embedding
    这个东西确实不太好想象，靠代码就行了
    """
    # 第三个问题：关于循环神经网络的官方API的使用，到底是包含输出还是只有隐状态
    print("------------------nn.rnn()的使用-------------------")
    vocab_size = 1027
    num_hidden = 256
    rnn_layer = nn.RNN(input_size=1027, hidden_size=256, )
    num_steps = 35
    batch_size = 2
    state = None #初始隐状态
    X = torch.rand(num_steps, batch_size, 1027)
    Y, state_new = rnn_layer(X, state) #目前只知道这两个应该都是隐状态,Y存储了历史以来（35个）的隐状态，state_new则存储的是最后一个隐状态
    print(Y[-1])
    print(state_new==Y[-1])
    """
    所以截止目前为止
    1. nn.RNN()的输入已经明确：
        1）第一个参数输入len(vocab)也就是词表大小
        2）第二个参数输入该次隐状态计算时的隐藏层信息，也就是隐藏层单元数
        3）第三个参数输入循环层数，也就是隐藏层数
        4）接下来还包含是否包含bias
    2. nn.RNN()的使用也已经明确
        1）先创立RNN层：rnn_layer = nn.RNN(input_size, num_hidden, num_layers)
        2）使用之rnn_layer(X)或者将该层加入到模型序列中：比如自己创建的一个类RNNModel()
    3. 其中输出也已经明确
        H， state_last = rnn_layer(X)
        其中H.shape=[num_steps, batch_size, num_hidden]为第一个时间步到第T个时间步的隐状态信息
        state_last为时间步T时的隐状态
        如果是双向的或者多层的时候：state_last.shape=[num_layers*num_directions, batch_size, num_hidden]
    关于nn.GRU()以及nn.LSTM()的输出信息，后面再看
    """
    print("---------------------nn.GRU以及nn.LSTM的使用----------------------------")
    gru_layer = nn.GRU(input_size=1027, hidden_size=num_hidden)
    Y, state_new = gru_layer(X)
    print(Y.shape)
    print(state_new.shape)
    print(state_new == Y[-1])
    """
    因此nn.GRU()在使用，以及结果上与nn.RNN()别无二致，只是在计算的时候，GRU较为复杂一些：需要计算重置门，更新门，根据重置门计算候选隐状态，然后计算隐状态
    注意两者在数据的预处理上也是一样的，但是格外注意：
    rnn_layer(X)中的X.shape=[num_steps, batch_size, vocab_size]
    而实际上拿到的数据经过词元化，数字化，one-hot之后，数据形状是batch_size, num_steps, vocab_size
    此时需要转置顶X=X.permute(1, 0, 2)
    而如果不做one-hot,则需要做X=nn.Embedding(vocab_size, 自己设置的维度)，进行一次和nn.Linear()类似的处理
    使之维度发生变化
    """
    #实验一下nn.LSTM()
    lstm = nn.LSTM(input_size=vocab_size, hidden_size=num_hidden, num_layers=1)
    Y, (state_new, c_n) = lstm(X)
    print(Y.shape)
    print(state_new.shape)
    print(c_n.shape)
    # print(state_new==Y[-1])
    """
    因此nn.LSTM()在使用的时候，与其他两者也无二致，计算的时候同样更为复杂
    输出则变成三项，增加了一项记忆元的输出，因为记忆元参与下一个时间步记忆元的生成，其维度与隐状态一样
    数据的处理同上
    """
    print("-------------------------编码器解码器的实战------------------------------")
    # 1. GRU网络实现编码器功能
    # 编码器的功能就是将输入序列转换为隐状态后，再转换为上下文变量。但是有个问题是，我在代码中没有看到如何转换为上下文变量呀！测试一下
    encoder = Seq2seqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print(output.shape)
    print(state.shape)
    print(output[-1]==state)
    # 所以左看右看，并没有进行上下文变量计算的部分，只做了隐状态的计算。从输入序列到隐状态。是否可以认为，此处的隐状态=上下文变量？。暂时就这样考虑吧

    # 2. 接下来看看编码器
    # 搞清楚两个问题：1）其中输入是什么？怎么和编码器相联系的。其中输入包括编码器的上下文变量C也就是上面例子中的output。编码器输出的最后一个时间步的隐状态用于解码器的初始隐状态。因此解码器与编码器的隐藏层和隐藏单元数需要相等
    # 2）我暂时连问题都问不出来呀。逐行解析下代码
    # 实例化测试，实验一下
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # decoder.eval() # 进入测试模型，表示不做梯度的记录和传递
    state = decoder.init_state(encoder(X)) # 设置解码器的初始化隐状态，等于编码器的最后一个隐状态
    output, state = decoder(X, state)
    print(output.shape)
    print(state.shape)
    # print(output[-1] == state)








