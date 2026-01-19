#### Seq2Seq模型
将一个输入序列转化为另一个输出序列。输入和输出的长度不一定相同

* 编码：首先将输入序列（如中文句子）转化为一个隐藏的向量或矩阵，代表输入的语义。
* 解码：然后使用这个向量或矩阵生成对应的输出序列（如英文句子）。

Transformer 是一种广泛应用于 Seq2Seq 任务的模型，最初就是为了机器翻译而设计的。通过这样的编码-解码过程，几乎所有 NLP 任务（如文本分类、词性标注等）都可以被看作 Seq2Seq 问题
![encoder_decoder](https://github.com/liu66-qing/NOTEBOOK/blob/main/encoder_decoder.png)

Transformer 由 Encoder 和 Decoder 组成，每个 Encoder（Decoder）又由 6 个 Encoder（Decoder）Layer 组成。输入源序列会进入 Encoder 进行编码，到 Encoder Layer 的最顶层后，编码结果会被输出给 Decoder Layer 的每一层。经过 Decoder 的解码处理后，最终得到输出目标序列。

---
#### 前馈神经网络
前馈神经网络（Feed Forward Neural Network，FNN），每一层的神经元都和上下两层的每一个神经元完全连接的网络结构。每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。

前馈神经网络的实现
```
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```

--- 
#### 层归一化
层归一化（Layer Norm）是深度学习中的一种经典归一化操作，它是神经网络常用的归一化方法之一。常见的归一化技术包括批归一化（Batch Norm）和层归一化（Layer Norm）。

* 归一化的目的：

归一化的核心目的是确保不同层的输入具有相似的取值范围或分布。在深度神经网络中，每一层的输入来自于上一层的输出。随着网络层数的增加，随着参数的变化，较高层的输入分布会发生较大的变化。这种变化会随着网络深度的增加而加剧，从而导致模型训练过程中预测误差的增加。

为了减少这种误差，通常需要在深度神经网络中进行归一化操作，将每一层的输入标准化为标准正态分布。这种标准化操作能有效缓解深度网络中的 内部协方差偏移（Internal Covariate Shift），即不同层的输入分布不一致的问题。

* 批归一化与层归一化：

批归一化（Batch Norm）：在一个 mini-batch 上对每个特征进行归一化。即对于整个 mini-batch，首先计算每个特征维度上的均值和方差，然后使用这些统计量对每个样本进行归一化。然而，批归一化存在一些缺点，特别是在以下场景中：

当 mini-batch 较小，计算的均值和方差可能无法准确反映全局分布，影响归一化效果。

在处理 RNN 等时间序列任务时，由于每个时间步的数据分布可能不同，批归一化失去了有效性。

在测试阶段，由于变长输入的特性，可能没有足够的统计信息支持推理过程。

* 层归一化（Layer Norm）：层归一化的解决方案是，在每个样本的每一层中，计算该层所有特征的均值和方差，然后进行归一化。这意味着对于每个样本，层归一化都会在每个层内部进行标准化，而不依赖于整个批次的数据。层归一化和批归一化的原理类似，只是归一化的维度不同，层归一化是对每个样本内的所有特征进行归一化。

层归一化的优势：

层归一化避免了批归一化在小批量数据和时间序列数据中可能遇到的局限性。

它特别适用于 RNN、变长序列等任务，因为它不依赖于批量数据的整体统计特性，而是基于单个样本进行处理。

在测试阶段，由于层归一化在每个样本内进行归一化，它避免了批归一化在推理时遇到的统计信息不足的问题。

实现如下：
```
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
	super().__init__()
    # 线性矩阵做映射
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
	# 在统计每个样本所有维度的值，求均值和方差
	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
---
#### 残差连接

由于 Transformer 模型结构较为复杂且层数较深，为了避免模型退化，Transformer 引入了 残差连接（Residual Connection） 的设计。残差连接的核心思想是，在每个子层的输出中，不仅包含上一层的输出，还包括上一层的输入。通过这种方式，模型能够直接将较低层的信息传递到较高层，帮助高层更专注于学习变化的部分，即 残差。

在 Transformer 的 Encoder 中，残差连接的具体实现方式如下：

1. 在第一个子层，输入会先通过 多头自注意力机制（Multi-Head Attention）进行处理，同时该输入会直接传递到该层的输出。然后，经过自注意力层的输出会与原输入相加，再进行 层归一化（Layer Norm） 操作。

2. 在第二个子层，处理过程类似，输入经过前馈神经网络（Feed-Forward Network）后，也会与该层的输入相加，再进行标准化。

在实际代码实现中，残差连接通过在 forward 计算中将输入加到输出上来实现：

注意力计算：输入先通过注意力层进行处理，然后将输入加到该输出上。

前馈神经网络：经过注意力层处理后，再通过前馈神经网络进行进一步处理，同样地，输入会加到前馈网络的输出上。

通过这种设计，模型能够确保信息在多层之间不会消失，同时也能有效防止梯度消失问题，促进网络的训练过程。
```
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```
