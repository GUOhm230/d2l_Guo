"""
本章主要是计算机视觉的实例
本处内容看完，真的是东西太多了
1. 搞明白怎么做图片的预处理（图像增广），哪些代码，要自己学会
2. 怎么做模型微调（其实是迁移的一种方法），这个微调不是用预训练模型换个数据做训练即可，而是要修改模型的输出层，并复制其中参数，然后对特定的层进行模型初始化，然后再做训练
3. 有关目标检测的算法：锚框生成--->IOU--->标注与偏移量的生成--->nms(非极大值抑制)预测边界框--->训练中通过卷积生成特征图，对特征图进行分类预测，并生成锚框，然后预测偏移量和标注（有点乱）
4. 多尺度目标检测：何为多尺度锚框，这个多尺度指的是什么
5. 目标检测数据集的制作
6. 单发多框检测：就是应用多尺度锚框的一个实例：模型有5大组件：类别预测以及边界框预测（用卷积实现，通道实现类别预测值），连接多尺度的预测，高宽减半快，网络基础快。本处关于模型的构建还是能吸取一些经验的
7. RCNN，该部分只讲了一些概念，没具体讲，暂时跳过，但是需要学习
8. 语义分割部分还不是很清楚，我都不知道啥叫语义分割，说是像素级的分类，又有图像分割和实例分割，好像有点乱
9. 转置卷积其实是普通卷积的一种逆运算，本处关于维度需要深入一下还
10. 全卷积网络，其实就是转置卷积的一个应用。使得输出高宽等于输入高宽。普通卷积能使高宽减半，而转置卷积则可以使高宽加倍
11. 风格迁移：就是一个内容图像，一个风格图像一个合成图像。本处除了方法之外，有关模型的训练，保持模型参数不动，而灵活的将合成图像作为唯一参数进行更新的方法让我有些惊艳。再次加深了对模型的理解
原来模型不仅仅是需要卷积啊，循环啊，线性这种东西，而是可以随便指定一些参数，让这些参数加入优化算法中，在优化算法中告知其中参数
12. kaggle两个例子用例外的篇幅，不在此处使用
2023.10.16 Created by G.tj
第十三章，居然整整一个半月的时间在整理
关于目标检测（SSD）主要是搞明白了以下几个点：
1. 锚框的定义是在图片确立之时就能完成的。如果使用多尺度锚框，则在生成特征图之后，照样可以做完。
2. 训练时：锚框生成之后，根据标签锚框，可以得到该锚框对应的类别和偏移量。而这就是后面做损失时的cls_labels和bbox_labels
3. 模型输入的是一张图片，经过卷积，得到两个输出：[batch_size, a*分类数， h, w]以及[batch_size, a*4, h, w]。其中第一个是置信度系数矩阵。第二个是偏移量矩阵。后两个维度表示对应的单元，也就是像素位置。
4. 置信度系数矩阵后，可以根据置信度系数进行NMS也就是非极大值抑制。通过NMS去筛选掉一批重合度比较高的边界框。注意其中的输入是预测边界框
5. 训练时的损失计算包括两个：1）偏移量损失（偏移量预测值，生成锚框后与真实边界框可计算其损失）。2）类别预测损失（预测的类别，根据置信度最大值得到，真实类别为锚框与真实边界框的取值）
6. 由此可以得到训练时的流程：给出输入图像--->对图像进行锚框生成--->对生成的锚框根据训练标签中的真实边界框计算互相的IOU值做标注，得到对应锚框的类别和偏移量---->建立模型，输入模型得到两个输出：类别置信度以及偏移量信息--->计算损失更新参数
7. 而在测试阶段：给出输入图像--->对图像进行锚框生成--->预测偏移量和类别--->类别预测使用softmax得到概率值--->根据预测的偏移量和类别与之前生成的锚框计算预测边界框--->nms得到最后输出
"""
# %matplotlib inline
import torch
import torchvision
from torch.utils import data
from torch import nn
from d2l import torch as d2l
import numpy as np
import math
import os
from torch.nn import functional as F
# --------------------------1. 图像增广------------------------------ #
# 在输入img上多次运行增广方法
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    # print(Y[0].size)
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 数据处理成可迭代对象的时候，需要对图片进行相应的前处理。数据迭代方式
def load_cifar10(is_train, augs, batch_size):
    datasets = torchvision.datasets.CIFAR10(root="../data", train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """多卡训练"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) # 使用多卡训练。
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i+1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i+1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {metric[0] / metric[2]:.3f}, train acc' 
           f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on'
          f'{str(devices)}')

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


# 2. 微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size, shuffle=False)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    # 主要是这个地方有疑问。这样一看没啥疑问。主要是如何对不同的层使用不同的学习率的问题。weight_decay则用于进行L1正则化
    if param_group:
        param_1x = [param for name,  param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        # 这里是对不同的参数选用了不同的学习率
        trainer = torch.optim.SGD([{'params': param_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001) # weight_decay也就是权重衰减指数，是用于正则化的
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 3. 锚框
# 边界框是标注物体对象的矩形框。有两种表示方法：1. 左上右下坐标 2. 中心点坐标以及宽高
def box_corner_to_center(boxes):
    # 从（左上，右下）坐标切换到（中间，宽度，高度）
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] # 这里的维度变化和下面的是一样的
    cx = (x1+x2) / 2
    cy = (y1+y2) / 2
    w = x2-x1
    h = y2-y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes

def box_center_to_corner(boxes):
    # 从（中间，宽度，高度）坐标切换到（左上，右下）， 输入维度是[边界框数量， 4]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] # 生成的每个维度为：[边界框数量]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1) #[边界框数量，4]
    return boxes

def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],  fill=False, edgecolor=color, linewidth=2)

# 给定一张图片，对每个像素生成锚框。这里阻挡了我好几天
# 怎么对每个像素设置不同的锚框
# 这里有很多代码需要学习的
def multibox_prior(data, sizes, ratios):
    """
    以每个像素点为中心生成具有不同形状的锚框
    sizes表示未按比例还原的锚框面积占原始图片经过归一化（宽除以w,高除以h）的比例
    ratio是锚框的高宽比（未按比例还原）
    该方法的做法是：
    1. 对图片进行归一化，使之成为一个正方形框
    2. 求该图片中每个像素的中心点
    3. 计算未按比例还原的锚框的h以及w值。
    4. 根据像素中心点坐标，对锚框进行还原
    弄懂原理之后，代码的难点不在公式计算, 而在于输入输出维度，以及计算过程中对维度和张量的复制
    输入：原始图片img信息，锚框的缩放比例以及高宽比（均是列表，因为一个中心点可以有很多个锚框，每个锚框可以拥有不同的缩放比例以及高宽比）
    输出：[锚框总数=h*w(n+m-1), 4= 坐标情况]
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1) # 每个像素点生成的
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # 为了将锚框的中心移动到像素中心点，而设置了偏移量。像素点高宽为1,故而设置偏移量=0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # param_group在y轴上的缩放步长，为什么要做这个
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h # 像素点的起始高度加上偏移量，就是该像素点的中心点，乘1/h的意思是将整个高度缩放到1。这样做的目的是什么呢
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij') # shift_x.shape=shift_y.shape=[561, 728]
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1) # 相当于展开成一维
    # print(shift_y.shape)
    # 生成boxes_per_pixel个高和宽, 用于创建锚框的四角坐标（xmin, xmax, ymin, ymax).就是这里的写法不太明白
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    print("w:", w) # 长度为: n+m-1=5
    print("h:", h)
    # # 除以2来获得半高和半宽
    anchor_manipulation = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2 # 维度(2042040, 4),dagais
    # print("anchor_manipulation:", anchor_manipulation)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulation
    return output.unsqueeze(0)

# 交并比计算 J(A, B) = |A交B|/|A并B|用于度量某个锚框与真实边界框或者锚框与锚框之间的相似性
# 我想看看交集是怎么计算的.在数学上不难，但是想看看代码如何做到的
def box_iou(boxes1, boxes2):
    """给定两个框列表：真实边界框或者锚框，输出框的交并比。结果和输入顺序无关。但是元素位置对应和输入顺序是有关的，要注意一下，不然索引出错"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) # 简单的计算函数可以用lambda表达式
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts的维度为[boxes1的数量， boxes2的数量， 2]
    # 将boxes1的维度增加，目的是为让boxes中的每个框都和boxes中的每个框进行比较，这里当然是可以用重复计算，而不必使用广播的。
    # 取出其中x1，y1最大值部分（注意xy轴坐标的方向）得到交集的左上部分，获取x2, y2的最小值部分得到交集的右下坐标
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # 这里就是
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # 写错了吧，我说呢。就这个，搞了我半天
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) # clamp的意思是对tensor进行限幅。取值在[min, max]之间，如果大于max则替换为max,如果小于min则取值为min
    inter_areas = inters[:, :, 0] * inters[:, :, 1] # 每个交集的面积计算
    union_areas = areas1[:, None] + areas2 - inter_areas # 并集的面积，所以要减去一个交集
    return inter_areas / union_areas

# 生成锚框之后，需要标注锚框的类别和偏移量。这一步的运算，也和模型是没有任何关系的。是一些传统算法的应用。
# 类别就是该锚框框住的物体类别，而偏移量是对锚框进行重新定位而得到对应的预测边界框，该框还要进行筛选，才能获得最后的边界框作为最终的目标检测框
# 所以有两个问题：如何对框进行准确的标注？给定锚框A1, A2, A3, A4....An，给定真实边界框B1, B2, B3, B4...Bm。计算得到其对应的IOU矩阵X(n*m)
# 其过程：找到X中最大者，然后消去同行同列者，接着继续，直到所有的m个锚框完成工作。剩下的锚框继续遍历。找到没有遍历的Ai锚框对应的IOU最大的Bj框，大于阈值才分配。
# 上面的算法: 锚框分配算法。这个算法的难点，照样是张量的处理。所以这里卡住的不是学习内容，而是代码的提升截断
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    # 输入真实边界框和锚框,维度：[锚框或者真实边界框的数量]。输出：每个锚框对应真实边界框的索引（0, 1, 2...），若没有对应的真实边界框，则索引为-1.是一个长度为锚框数量的一维张量
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth) # J值，也就是交并比矩阵；j矩阵维度：[锚框数量，真实边界框数量]
    # print("交并比矩阵：", jaccard)
    anchors_bbox_map = torch.full((num_anchors, ), -1, dtype=torch.long, device=device) # 对每个锚框分配的真实边界框的张量。这个full应该怎么用呢
    max_ious, indices = torch.max(jaccard, dim=1) # 按照列求值，则列消失。实际是得到每个锚框对应边界框的最大值及其索引。max_ious是一个一维向量长度=锚框数量
    # print("max_ious:", max_ious)
    # print("indices:", indices) # indices返回的是对应的是当前最大值真实边界框的哪一个
    # print("max_ious >= iou_threshold:", max_ious >= iou_threshold)
    # 定位max_ious中大于阈值的索引,返回的是ij数值.reshape(-1)就是把这个变成了一维的向量。我有了这个，为什么还要索引呢
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 这个索引指的是阈值超过0.5的锚框编号是多少
    # print("anc_i:", anc_i) # 这个索引为什么要重复计算呢。我终于知道了
    box_j = indices[max_ious >= iou_threshold] # 这个索引指的是阈值超过0.5的锚
    # print("box_j:", box_j)
    anchors_bbox_map[anc_i] = box_j # 这里其实做了剩余的遍历，接下来只要做在矩阵X中找最大值然后删除的部分了
    # print("anchors_bbox_map:", anchors_bbox_map)
    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)
    for _ in range(num_gt_boxes): # 循环次数为真实边界框的数量
        max_idx = torch.argmax(jaccard) # 最大值
        # print("max_idx:", max_idx)
        # 以下两个是获得其行列所在的位置
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard # 对同行者消除，也就是赋-1操作
        jaccard[anc_idx, :] = row_discard # 对同列者也是
    return anchors_bbox_map

# 标注类别和偏移量:其实分配了边界框，也就相当于分配了类别
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    # 偏移量的计算是针对中心点坐标的。输入是：锚框坐标，以及被分配到的真实边界框的坐标。返回的就是偏移量的四个数字，长度等于锚框数量。其实实际中对所有的锚框都做这样的计算，只是如果没有分配到真实边界框，则全部取0
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + assigned_bb[:, :2]) / c_anc[:, 2:]
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset

def multibox_target(anchors, labels):
    """
    使用真实边界框标注锚框：偏移量以及类别。输入锚框:[batch_size, 锚框数量，锚框左上右下坐标], labels:[batch_size, 锚框数量，数字信息(类别，x1,y1,x2,y2)]
    输出：
        bbox_offset: 锚框对应的偏移量：维度[批量大小，锚框数量，4]。而这里计算做了一个reshape操作，最后维度：[批量大小， 锚框数量*4]
        bbox_mask: 掩码变量，用于过滤负类偏移量：维度：同上
        class_labels：锚框对应的类别：维度[批量大小， 锚框数量]
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0) #为什么要有一个squeeze()去消除维度为1的维.因为前面增加了一个1的维度，所以这步其实么有必要，假装有个batch_size维度
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device) # 返回锚框对应真实边界框的索引，负类锚框则为-1
        print("anchors_bbox_map:", anchors_bbox_map)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4) # 表示正类锚框者。正类锚框为1, 负类锚框为0。。为什么按4重复呢？，这个4是对应坐标的。这样就能把负类边框的偏移量清零了
        print("bbox_mask:", bbox_mask)
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device) # 锚框对应的类别, 初始化类别均为背景，也就是0
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device) # 这个4又是意思？这个应该是.这个4,是后面计算偏移量的时候要使用的坐标，我他么的，忘了有这步骤了
        print("assigned_bb:", assigned_bb)
        indices_true = torch.nonzero(anchors_bbox_map >= 0) #正类锚框的索引
        print("indices_true", indices_true)
        bb_idx = anchors_bbox_map[indices_true] #正类锚框对应的真实边界框的索引。
        print("bb_idx:", bb_idx)
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # 锚框对应的类别。0：背景。故而之前的类别延后。操作的时候只需要对正类的毛框进行赋值就行
        print("class_labels:", class_labels)
        assigned_bb[indices_true] = label[bb_idx, 1:]
        print("assigned_bb:", assigned_bb)
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

# 非极大值抑制NMS
# 通过偏移量以及原锚框生成预测边界框
def offset_inverse(anchors, offset_preds):
    # 输入：锚框坐标:[x1, y1, x2, y2]. 预测偏移量：也是4个值。这里就有点不懂了，为什么偏移量可以预测啊。便宜量预测之后便可以得到预测边界框。其实就是偏移量计算的逆运用
    # 输出：锚框对应的预测边界框。维度：[锚框数量，4]
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框进行置信度排序: 置信度又是怎么得来的：对于每个预测边界框，目标检测模型会计算每个类别的预测概率，其中对应的最大概率就是该预测边界框对应的类别，而这个概率就是置信度。
    一句话说：置信度就是预测边界框属于某类别的概率
    输入： 预测边界框，每个预测边界框对应的置信度，设置的阈值
    输出：非极大值抑制后的结果。得到的是剩下的预测边界框的置信度。这些预测边界没有一对比较相似"""
    B = torch.argsort(scores, dim=-1, descending=True) # B是置信度排序
    keep = []
    while B.numel() > 0:
        i = B[0] # 找到当前B的最大值，以此为基准
        keep.append(i) # 存储基准预测框
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].rehspae(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1) # 求基准值边界框与其他边界框之间的iou
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1) # iou中小于阈值的边界框的索引
        B = B[inds + 1] # 加1是要排除本身。按照原列表计算，计算后的依然是排好序的
        return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nums_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框
        输入: cls_probs：边界预测框对所有类别的预测概率[batch_size, 类别数，边界预测框数量]
        offset_preds: 锚框的预测偏移量。根据预测偏移量计算锚框的预测框
        anchors: 锚框
        这个代码没有细看
        """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_probs, offset_pred = cls_probs[i], offset_preds[i].reshape[-1, 4] # 这里的便宜量全部取0所以
        conf, class_id = torch.max(cls_probs[i:], dim=0) # 取出最大置信度，并得到其中最大的索引。按行取值，则行消失，得到的是列的最大值，也就是每个锚框中对每类预测的最大值，并得到索引，索引指示分类
        predicted_bb = offset_inverse(anchors, offset_pred) # 锚框转化成预测框
        keep = nms(predicted_bb, conf, nums_threshold) # 非极大值抑制，得到置信度最高且和其他锚框均不相似的锚框索引

        # 找到所有的non_keep的索引。并将其类别设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True) # 得到其中不重复的元素
        non_keep = uniques[counts==1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        below_min_idx = (conf < pos_threshold)
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

import pandas as pd

# 5. 目标检测数据集: 本处代码暂时忽略。不是重要的点
def read_data_bananas(is_train=True):
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv()

# 6. 单发多框监测（SSD）
# 1）类别预测层：用卷积的通道层表示类别。得到每个锚框对于该类别的预测概率.这里的疑问就是，为什么卷积能做这样的预测
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 2）边界框预测：其实是对每个锚框预测4个便宜量
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度预测
def flatten_pred(pred):
    torch.nn.Flatten()
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # 表示从维度1开始展平，也就是后三个维度展平

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 用于高宽减半块。模块构建。这里需要学习一下模型构建的方法，其实之前的网络建模和这个还是很像的
# 定义一个高宽减半的卷积块。卷积层+BN+ReLU这算是进行了一次完整的卷积运算。该blk中包含两个这样的卷积层。卷积层的输出通道数是output_channels.最后做一个最大汇聚，并不改变通道数。而是将高宽减半
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# 基础网络块：该基础模块实现了了3个高宽减半，通道数加倍块。并构成了一个nn.Sequential()模型
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# 定义模型顺序：先进行基础网络模块，以得到不同高宽的feature map。然后定义高宽减半块，同样当做特征图，作为锚框生成图
def get_blk(i):
    # 首先定义的是基础网络块，
    if i== 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    print(Y)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 这段代码是整合SSD模型：搞清楚一件事，所有的运算其实都是在特征图上完成，得到的锚框也是归一化后的[0, 1]区间，最后画图的时候才还原
# 那这个模型做啥了呢？模型中做了很多不是模型，而是传统的算法
# 构建模型--->forward() 模型输入X，也就是图片。模型输出：anchors: 多尺度锚框，cls_preds：不同尺度下的类别预测结果，bbox_preds:
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f"blk_{i}", get_blk(i))
            setattr(self, f"cls_{i}", cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        # 因为经过的是5个模型块，所以这里直接用5
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                                                                     getattr(self, f'cls_{i}'),
                                                                     getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        # print("anchors:", anchors.shape)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

# 9. 转置卷积
# 转置卷积的实现：假设步幅=1，填充=0
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w -1))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j+ w] += X[i, j] * K
    return Y

# 初始化转置卷积层, 由双线性插值怎么引出参数初始化呢？有点不明白. 也不是很重要的点，暂时放过吧
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (1 - torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def predict(img):
    # 对测试图片的每个通道都进行标准化
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap


# 风格迁移
# 步骤：做数据，构建模型且模型不做参数，对内容图像和风格图像分别提取内容特征，风格特征，然后构建参数也就是初始化合成图像，尝试一下
# 模型参数设置
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs) # 这个地方要百度一下，总是忘，为什么一定要这步
        self.weight = nn.Parameter(torch.rand(*img_shape))
    def forward(self):
        return self.weight

gen_img = SynthesizedImage((3, 320, 480))

# 数据前处理
def preprocess(img, image_shape):
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    sgb_std = torch.tensor([0.220, 0.224, 0.225])
    # Compose中有call方法
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean= rgb_mean, std= rgb_std)
    ])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1) # 这里的限制搞得有点莫名其妙的。懂了，这里本来还是[0, 1].后面才做了[0, 255]
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 模型构建， 直接用vgg就行了
# 风格特征提取，内容特征提取
def extract_features(X, content_layers, style_layers):
    contents, styles = [], []
    for i in range(len(net)):
        X = net[i](X) # 注意这里是需要逐层运算的。因为要收集中间层的结果
        if i in content_layers:
            contents.append(X)
        if i in style_layers:
            styles.append(X)
    return contents, styles

# 获取内容特征数据
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    styles_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(styles_X, content_layers, style_layers)
    return styles_X, styles_Y

# 定义损失函数
# 1. 内容损失
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

# 2. 风格特征：计算gram矩阵
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
def tv_loss(Y_hat): #这里的是
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 总损失
content_weight, style_weight, tv_weight = 1, 1e3, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight #
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# 模型训练前要做的事
def get_init(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# 模型训练
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_init(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs], legend=['content', 'style', 'TV'], ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step() # 学习率调度器

        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])
            d2l.plt.show()
    return X

if __name__ == "__main__":
    print("------------------------生成多个锚框-----------------------")
    # 本处生成多个锚框，我居然真就不知道作何解，只有对着代码来看了
    # 两点疑问：1. 缩放比是什么意思 2. s,r为什么取s1, r1即可呢
    # torch.set_printoptions(precision=4, sci_mode=False) # 取消使用科学记数法并指定位数
    # h, w = 561, 728
    # X = torch.rand(size=(1, 3, h, w))
    # Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    # print(Y)
    # # 维度懂了，但是现在还是hs
    # boxes = Y.reshape(h, w, 5, 4)
    # print(boxes)
    # 测试一下网格图的生成
    # X = torch.arange(4)
    # Y = torch.arange(7)
    # x, y = torch.meshgrid(X, Y, indexing='ij') # 得到新的想x, y的维度等于输入张量的长度其中x内元素每行相同列则相反。如果indexing='xy‘则x.shape=y.shape=[len(Y), len(X)].其中x内元素每列相同行则相反。
    # #
    # print(x)
    # print(y)
    # print(561*728*5) # 生成的总数
    # cat和stack的用法
    # x = torch.arange(3)
    # y = torch.arange(3, 7)
    # print(x, y)
    # z = torch.cat((x, y), dim=1)
    # print(z)
    # for i in range(2):
    #     print(i)
    print("------------------------------1. 图像增广---------------------------")
    # 1. 翻转和裁剪
    img = d2l.Image.open("/home/gtj/Pictures/personal photo/2017.jpg")
    # d2l.plt.imshow(img)
    # aug1 = torchvision.transforms.RandomHorizontalFlip() # 随机左右翻转
    # apply(img, aug1)
    # aug2 = torchvision.transforms.RandomVerticalFlip() # 随机上下翻转
    # apply(img, aug2)
    #
    # shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # # scale指的是裁剪后的面积为原始面积的比例, ratio为裁剪后图像的宽高比为ratio中值得均匀抽样值, size为最后的输出结果resize为该数值
    # apply(img, shape_aug)
    #
    # # 2. 改变颜色
    # color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
    # # 四个参数分别为亮度，对比度，饱和度以及色调
    # apply(img, color_aug)
    #This is helpful when you want to visualize data over some
    # # 结合多种图像增广方法
    # augs = torchvision.transforms.Compose([aug1, color_aug, shape_aug])
    # apply(img, augs)
    # d2l.plt.show()
    #
    # # 使用图像增广进行训练
    # # 一般只对训练样本进行图像增广，而对预测样本则不进行该步骤
    # all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
    # d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
    # d2l.plt.show()
    # train_augs = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor()
    # ])
    # test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
    # train_with_data_aug(train_augs, test_augs, net)
    # d2l.plt.show()
    """
    本处需要学到的东西是：
    1. 图像预处理，关于transforms的float使用param_group。图像增广：水平翻转，左右翻转，按照一定的sr比例进行图像裁剪，图像颜色的改变：设置亮度，对比度，饱和度和色调。
    以及几种增广方法的综合使用：torchvision.transforms.Compose([])
    2. 有关模型训练的完整过程：数据制作，模型参数初始化，多卡运行模型，其实是把模型参数复制到多卡中，并把数据也放在多卡中
    3. 其他关于模型的训练，测试则没有新鲜内容了
    """
    print("--------------------------------2. 微调-------------------------------")
    # 本部分主要需要解决的是模型参数的复制. 任务依然是图片分类. 模型微调就算迁移学习的一部分
    # 有关这里的模型参数我一定要做专题整理一下
    # 获取数据
    d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
    data_dir = d2l.download_extract('hotdog')
    train_img = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_img = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
    # 这个方法是什么？怎么取到的数据，里面的数据组织形式是什么？
    hotdogs = [train_img[i][0] for i in range(8)]
    not_hotdogs = [train_img[-i-1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    d2l.plt.show()

    # 对图片进行增广处理以及归一化, 这个用法normalize也要记住，常用的
    # 这三个数字是根据
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 这些参数的设置是个问题，需要学习一下
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    # 定义和初始化模型
    pretrained_net = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1) # 警告，也就是pretrained这个形参已经被弃用了。后面用weights可以指定模型的版本
    print(pretrained_net.fc.weight.data.shape) #得到一层fc.其中权重为原来的转置
    # print(pretrained_net.fc.in_features)
    # print(pretrained_net.fc.out_features) # 吸取的教训就是多看看源代码呀。这个在nn.Linear上有相同的属性
    pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)
    # print(pretrained_net) # 这里不就已经改了代码层了嘛
    nn.init.xavier_uniform_(pretrained_net.fc.weight) # 对该层进行初始化

    # 微调模型并开始进行模型的训练
    # train_fine_tuning(pretrained_net, 5e-5)
    # # 定义一个没有进行模型微调的训练一下试试
    # net = torchvision.models.resnet18()
    # net.fc = nn.Linear(net.fc.in_features, 2)
    # train_fine_tuning(net, 5e-4, param_group=False)
    # d2l.plt.show()offset_preds
    # 其实关于本处还有几个地方需要重新学习一下
    # 1. torchvision.transform.Normalize(mean, std)其中的mean和std两个值
    # 2. 定义网络模型，默写一下
    # 定义网络模型
    # myNet = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # # 更改输出层
    # myNet.fc = nn.Linear(myNet.fc.in_features, 2) # 注意一个十分简单的内容：参数在创建的开始就要搞清楚他是否为叶子节点，即require_grad=True
    # # 该层的参数初始化
    # nn.init.xavier_uniform_(myNet.fc.weight) # 里面要给定参数就表明对哪个参数进行初始化
    # 模型的训练，数据的制作，后面迟早要再来一次的
    # 还有难点就是如何对不同的参数使用不同的学习率
    print("-----------------------------3. 锚框-------------------------")
    # 本章有几个概念：目标检测问题以及边界框，锚框，IOU交并比，非极大值抑制。本章内容实在太多，今天要解决部分
    # 1. 边界框的表示方法 每天学习的时间太晚了，总结原因还是晚上睡太晚，而晚上睡太晚的原因还是心绪不定导致。不看知乎，今天必须学完13.6节。明早完成最后一个难点，然后晚上之前把第十三章搞定
    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    boxes = torch.tensor((dog_bbox, cat_bbox))
    # print(boxes)
    # print(boxes[:, 0])
    print(box_corner_to_center(boxes))
    print(box_center_to_corner(box_corner_to_center(boxes)))
    # 有关stack的使用方法在本处进行一次实验
    # stack中两个拼接的张量维度必须一致。第二个维度为拼接后维度取值范围内
    # X = torch.arange(4, dtype=torch.float32) #.reshape(2, 4)
    # Y = torch.rand(4)
    # print(X, Y)
    # Z = torch.stack((X, Y), dim=0) # dim=0表示在行上进行维度扩展。之后的维度为[2, 4]
    # print(Z)
    # X = torch.arange(6).reshape(2, 3)
    # Y = torch.rand(6).reshape(2, 3)
    # print(X, "\n", Y)
    # Z = torch.stack((X, Y), dim=0)
    # print(Z)# 在第一维上进行的扩展，扩展之后的维度为[2, 2, 3]
    # Z = torch.stack((X, Y), dim=1)
    # print(Z)# 在第二维上进行的扩展，扩展之后的维度为[2, 2, 3]
    # Z = torch.stack((X, Y), dim=-1)
    # print(Z)  # 在最后一个维上进行的扩展，扩展之后的维度为[2, 3, 2]
    # 至于怎么做的扩展，其实暂时还不太重要
    # fig = d2l.plt.imshow(img)
    # fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    # fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    # d2l.plt.show()
    # 绘制锚框
    Z = torch.arange(3)
    ZZ = Z[None, :, None]
    print(ZZ.shape)
    device = d2l.try_gpu()
    bbox_scale = torch.tensor([728, 561, 728, 561])
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]]).to(device)  # [2, 4]
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98],
                           [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]]).to(device) # [5, 4]
    # print("交并比计算")
    # jaccard = box_iou(anchors, ground_truth)
    # print(jaccard)
    # print(boxes1[:, None, :2])
    # print(boxes2[:, :2])
    # inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # # print("inter_upper")
    # print("inter_upperlefts:", inter_upperlefts) # 其维度信息：[boxes1的数量，boxes2的数量，2]
    # # torch.max的使用居然可以针对不同维度的实现
    # inter_lowerrights = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # print("inter_lowerrights:", inter_lowerrights)
    # inters = (inter_lowerrights - inter_upperlefts)
    # print("inters:", inters)
    # print("有关torch.max的使用")
    # # 1. 输入只有一个张量
    # # 输入的是一个一维向量
    # a = torch.arange(4)
    # print(a)
    # a_max = torch.max(a)
    # print(a_max) # 该张量中的最大值
    # # 输入的是一个二维张量
    a_2D = torch.arange(6).reshape(2, 3)
    # a2D_max = torch.max(a_2D) # 没有指定维度之前，得到的结果，依然是一个值，是取所有元素里最大的那个
    print("a_2D:", a_2D)
    # # 尝试指定维度进行最大值获取
    a2D_max2, indices = torch.max(a_2D, dim=1) # 当指定维度为列时，得到的结果就消失列的维度。和torch.sum, torch.mean是一样的
    # s = torch.sum(a_2D, dim=1) #指定哪个维度，则该维度的数字最终消失
    # print(s)
    print("指定维度的时候:", a2D_max2)
    print(indices) # 返回的是最大值所在的位置
    device=d2l.try_gpu()
    # anchors_bbox_map = assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5)
    # print("锚框分配算法：", anchors_bbox_map)
    # # 尝试一下3个维度进行运算
    # a_3D = torch.arange(24).reshape(2, 3, 4)
    # print(a_3D)
    # a3D_max = torch.max(a_3D, dim=0) # 按照行计算，则行的维度小时。
    # print(a3D_max)
    # # 2. 输入为两个张量
    # print("输入为两个张量")
    # a = torch.arange(3)
    # b = torch.rand(3)
    # print(a,"\n",b)
    # ab_max = torch.max(a, b) # 一维向量，每个元素进行对比
    # print(ab_max)
    # a_2D = torch.arange(12).reshape(3, -1)
    # b_2D = torch.rand((3, 4))
    # print(a_2D,"\n",b_2D)
    # ab2D_max = torch.max(a_2D, b_2D)
    # print(ab2D_max)
    # a_3D = torch.arange(24).reshape(2, 3, -1)
    # b_3D = torch.rand((2, 3, 4))
    # print(a_3D, "\n", b_3D)
    # ab3D_max = torch.max(a_3D, b_3D) # 两个维度相同的张量进行运算。此时仍然是对应的数据进行比较，输出维度与输入维度是一样的嗯
    # print(ab3D_max)
    # ab23D_max = torch.max(b_3D, a_2D)
    # print("二三维原数据：","\n",a_2D,"\n",b_3D)
    # print("二三维：", ab23D_max) #两个维度不同的张量进行比较，对维度较低者进行广播。两个维度不同，但是有两个维度是相同的
    # # 一个二维，一个三维。但是只有一个维度是相同的
    # a_3D = torch.arange(6).reshape(2, 3)
    # # a_3D = a_3D[:, None, :]
    # b_2D = torch.rand((4, 3))
    # c = torch.max(a_3D, b_2D)# 本处我也懂了，a_3D的维度为[2,1,3] 而b的维度为[4,3]此时要将a广播至[2, 4, 3] 1这个位置，要么是1要么等于4.不能是其他维度，否则是不可以广播的
    # print(a_3D, "\n", b_2D)
    # print(c) # [2, 4, 3]
    # 有关torch.max的使用完成
    # torch.full()的使用
    # a = torch.full((3, 3), -1) # 前面size指的是生成数据的维度，-1表示要填充的数据
    # print(a)
    # b = torch.nonzero(a)
    # print(b.reshape(-1))

    # print()device
    # torch.argmax()
    a = torch.rand(3, 4)
    b = torch.arange(12)
    print("a:", a)
    print(torch.argmax(a)) # 可知，最终结果为展平后的结果。
    print((b >= 2).float()) # 这样的数据类型转换并没有错
    print("获取类别和偏移量")
    labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0)) # squeeze是压缩的意思。unsqueeze就是扩展的意思。其实是解缩。在相应的维度增加一个1
    bbox_offset, bbox_mask, class_labels = labels
    print(bbox_offset.shape)
    print(bbox_mask.shape)
    print(class_labels.shape)
    print("torch.nonzero用法")
    a = torch.arange(12).reshape(3, 4)
    print(a)
    b = torch.nonzero(a%2!=0) # 返回非零元素所在的位置
    # print(a%2!=0)
    print(b.reshape(-1))
    print("---------------------------")
    offset_preds = torch.tensor([0] * 4).unsqueeze(dim=0)
    print(offset_preds[0].reshape(-1, 4)) # 真的可以一维变2维
    print("-----------------------4. 多尺度目标检测----------------")
    # 这个尺度是特征图的宽高尺度。当特征图宽高不一样时，求得的锚框数量也不一样
    # 针对特征图得到其中锚框数量，而w, h仍然是原图的
    print("-----------------------5. 目标检测数据集-----------------")
    # 下载数据集
    d2l.DATA_HUB["banana-detection"] = (d2l.DATA_URL + 'banana-detection.zip','5de26c8fce5ccdea9f91267273463dc968d20d72')

    # 参数设置
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 619], [0.71, 0.79], [0.88, 0.96]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    print("----------------------6. 单发多框检测-------------------")
    # 构建网络模型
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print(anchors.shape)
    print(cls_preds.shape)
    print(bbox_preds.shape)
    # 模型开始训练，数据处理，设置超参数
    # batch_size = 32
    # train_iter, _ = d2l.load_data_bananas(batch_size)
    # device, net = d2l.try_gpu(), TinySSD(num_classes=1)
    # trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    # # 定义损失函数和评价函数
    # cls_loss = nn.CrossEntropyLoss(reduction="none")
    # bbox_loss = nn.L1Loss(reduction='none')
    # def calc_loss(cls_preds,float cls_labels, bbox_preds, bbox_labels, bbox_masks):
    #     batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    #     cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    #     bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    #     return cls + bbox
    #
    # def cls_eval(cls_preds, cls_labels):
    #     """由于类别预测结果放在最后一维，因此argmax求最大值就需要指定最后一维"""
    #     return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum()) # 这个用法之前用过，这里
    #
    # def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    #     return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
    #
    # # 模型开始训练. 该部分代码还是要细看一下。
    # num_epochs, timer = 20, d2l.Timer()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    # net.to(device)
    # for epoch in range(num_epochs):
    #     metric = d2l.Accumulator(4)
    #     net.train()
    #     for features, target in train_iter:
    #         timer.start()
    #         trainer.zero_grad()
    #         X, Y = features.to(device), target.to(device)
    #         anchors, cls_preds, bbox_preds = net(X)
    #         bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
    #         l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
    #         l.mean().backward()
    #         trainer.step()
    #         metric.add(cls_eval(cls_preds,  cls_labels), cls_labels.numel(), bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    #     cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    #     animator.add(epoch + 1,  (cls_err, bbox_mae))
    # print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    # print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on' f'{str(device)}')
    print("----------------------------7. 区域卷积神经网络RCNN----------------------")
    #这部分我需要另外找书学习一下
    print("----------------------------8. 语意分割数据集-------------------------------------")
    print("----------------------------9. 转置卷积-------------------------------------")
    # 所谓转置卷积算是卷积运算的一个逆运算。这里也有填充，步幅之类的东西
    # 测试一下转置卷积，该代码的写法和卷积是一个思路：先根据输出大小，初始化值
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = trans_conv(X, K)
    print(Y)
    # 使用官方API
    X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=3, bias=False) # 注意输入X要4个维度：[batch_size, ]
    # print("转置卷积的参数", tconv.weight.data)
    tconv.weight.data = K
    Y = tconv(X)
    print(Y)
    # 设定有填充和步幅以及多通道: 其实发现，方法名不一样之外，其他的和Conv2d是一模一样的
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
    # 关于转置卷积的维度公式需要学习
    print("-------------------------10. 全卷积网络----------------------------------")
    # 使用卷积网络和池化可以导致图片的高宽的减半，降低，为了将图片恢复，使用转置卷积
    # 构建模型
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    # print(list(pretrained_net.children())) # children()得到的是模型的子块。块中可能会有很多网络层
    # print("模型设置：", list(pretrained_net.children())[4])
    net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 这个children要学习一下
    X = torch.rand(size=(1, 3, 320, 480))
    print(net(X).shape) # [1, 512, 10, 15] # 高宽缩小32倍
    # 修改一下模型
    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1)) # 给模型添加子网络
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    kernel_size = 64
    og = (1 - torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    # print(og[0].shape)
    # print(og[1].shape)
    # 全卷积网络训练
    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

    # 定义损失
    def loss(inputs, targets):
        return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
    print("设备:", len(devices))
    # trainer = torch.optim.SGD(net.paramsquare() takes 1 positional argument but 2 were giveneters(), lr=lr, weight_decay=wd)
    # d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    # voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    # test_images, test_labels = d2l.read_voc_images(voc_dir, False)
    # n, imgs = 4, []
    # for i in range(n):
    #     crop_rect = (0, 0, 320, 480)
    #     X = torchvision.transforms.functional.crop(test_images[i], * crop_rect)
    #     pred = label2image(predict(X))
    #     imgs += [X.permute(1, 2, 0), pred.cpu(), torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
    # d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
    # d2l.plt.show()
    print("--------------------------------11. 风格迁移-----------------------------")
    # 风格迁移其实比较简单：有三幅图：风格图，特征图
    style_img = d2l.Image.open("/home/gtj/Pictures/personal photo/b0.jpg")
    content_img = d2l.Image.open("/home/gtj/Pictures/personal photo/b2.jpg")

    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    pretrained_net = torchvision.models.vgg19(pretrained=True)
    print("vgg:", pretrained_net.features)
    print("children: ", list(pretrained_net.children()))
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)]) # 模型计算的时候不要输出，因此把后面两个可以删掉

    device, image_shape = d2l.try_gpu(), (300, 450)
    net = net.to(device)
    conten_X, contents_Y = get_contents(image_shape, device)
    _, styles_Y = get_styles(image_shape, device)
    output = train(conten_X, contents_Y, styles_Y, device, 0.3, 500, 50)
    """
    是时候总结一下了，内容实在太多，搞了半个月了：
    回顾下这部分的知识点：
    1. 关于图像增广：上下，水平翻转及其使用代码
    2. 模型迁移的方法：微调：模型输出层改变，复制预训练模型的其他层的参数和网络结构。然后对不同的模型参数使用不同的学习率进行重新计算
    3. 锚框生成算法：锚框归一化处理--->寻找像素中心点并计算每个像素中心点的偏移量---->对计算锚框的宽高（归一化下）---->与中心点做网格化计算相加
    ---->最后锚框实际的大小还要乘以原图的w, h
    4. 锚框生成以后，要对这些锚框进行筛选：就有了交并比算法，也是度量两个框的相似度的方法
    5. 之后需要对锚框进行分配真实边界框：算法是计算锚框与真实边界框的iou值，然后找到最大值，并删除该行与该列，接着继续找到最大值，如此直到所有的真实边界框都有对应的锚框。剩下的锚框则在行里找到最大值，然后当阈值到一定程度则取消之
    6. 上面得到了预测边界框之后要对这些预测框进行筛选，筛去重合度较大的边框：使用非极大值抑制。非极大值抑制，抑制的是置信度非极大的部分步骤为
    step1: 计算预测边界框对于类别的预测置信度，并选择概率最大者，排序得到L
    step2: 选择L中最大值对应的预测边界框，该框和其他框做IOU，删去IOU值超过一定阈值者，即删去和这个基准框相近，且置信度非极大值者。一般来说，他们会属于同一个类别
    step3: 依次为基准，直到最后所有框都为基准结束
    7. 这部分内容较多，有几次算法，有点捋不清了。算法其实主要是：分配预测边界框，NMS，好像还有一个
    8. 然后就是语义分割部分。这里其实就没大讲清楚。
    9. 单发多框监测（SSD）。多框也就是多尺度目标检测。主要是针对同一张输入图片，使用不同卷积之后，获取了不同的宽高。然后在不同的宽高下进行锚框生成
    宽高较大的时，可以生成较小锚框，检测更小的目标，而较小的宽高，会生成较大的锚框，可以监测较小的目标。
    10. 关于单发多框监测的模型步骤。构建模型的几种
    11. 转置卷积：可以看做是卷积的逆运算，卷积实现了高宽的减半，而转置卷积可以实现高宽的加倍。这样的话，就能让模型输出也和输入相同了
    12. 全卷积：实现像素级别的预测。得到的输出和输入是一样的，这是通过转置卷积实现的。而核，步幅和填充也是根据图片缩小的比例进行确定。
    全卷积的输出为[分类数，原图H，原图W]。这样每个像素则表示属于某个类别的概率
    13. 风格迁移：风格图像，内容图像和合成图像。其中内容图像提取内容特征，风格图像提取风格特征。由于合成图像需要保留内容图像的大致内容而不保留其中的细节
    因此，内容特征取卷积网络的上面一些的卷积层。而风格特征则取卷积网络的部分层
    预训练模型无需做训练，因为是用来提取特征的。而实际参数只有合成图像，合成图像作为模型的唯一参数
    损失的计算有三个：内容损失：合成图像的内容特征与内容图像的内容特征之间做差的平方。风格损失：合成图像的风格特征和风格图像的风格特征，其中要对特征
    做gram矩阵，然后对gram矩阵做差平方。全变分损失：合成图像做损失
    
    该部分学习到的函数方法
    torch.clamp(): 限幅函数
    torch.nonzero(): tensor元素中非零元素所在的位置
    torch.max(): 既有最大值，也有最大最大值所在位置
    torch.argmax(): 最大值所在的索引
    net.children()
    net.features()
    *list(net.features())
    torch.stack()
    torch.cat()
    torch.mul()
    """
