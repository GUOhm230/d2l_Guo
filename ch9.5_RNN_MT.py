"""
第九章，第五节：机器翻译与数据集
本章目的：综合循环神经网络内容
本节快速过。无需自己动手写完
2023.9.10 Created by G.tj
"""
import os
import torch
from d2l import torch as d2l

def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """预处理英语-法语数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 用空格替换不间断空格。不间断空格: 为了换行时，可以不在两行显示的空格符号表示：U+00A0。普通的空格：\u202f
    # 用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)

# 文本数据集词元化
def tokenize_nmt(text, num_examples=None):
    # 词元化英语-法语数据集
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

# 绘制词元的直方图
def show_list_len_pair_hist(lagend):
    pass

# @save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    print("line:", line)
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab(['<pad>'])) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# 训练模型
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab



if __name__ == "__main__":
    # 1.j取出文本数据集，数据集的词元化以及构建词表
    d2l.DATA_HUB['frg-eng'] = (d2l.DATA_URL + 'frg-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
    # 数据读取
    raw_text = read_data_nmt()
    print(raw_text[:80])
    # 数据预处理：小写转大写，标点加空格
    text = preprocess_nmt(raw_text)
    print(text[:80])
    # 将英语和法语分开
    source, target = tokenize_nmt(text)
    # 创建词表,也就是构建语料库: 文本中出现单词根据词频排序
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens = ['<pad>', '<bos>', '<eos>'])
    print(src_vocab.token_to_idx)
    print(len(src_vocab))
    Y = truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
    print(Y)
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        break