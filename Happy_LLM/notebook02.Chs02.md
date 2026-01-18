#### 注意力机制
自然语言处理领域，把重点注意力集中在一个或几个 token就可以而取得更高效高质的计算效果

注意力机制核心变量：Query（查询值）、Key（键值）、Value（真值）

Q和K进行运算得到权重，表示从 Query 出发，对文本每一个 token 应该分布的注意力相对大小


得到的 x 即反映了 Query 和每一个 Key 的相似程度，再通过一个 Softmax 层将其转化为和为 1 的权重：

最后，我们再将得到的注意力分数和值向量做对应乘积即可，如果 Q 和 K 对应的维度比较大，softmax 放缩时就非常容易受影响，因此要将 Q 和 K 乘积的结果做一个放缩

![Attention](https://github.com/liu66-qing/NOTEBOOK/blob/main/attention.png?raw=true)

```python
'''注意力计算函数'''
def attention(query,key,value,dropout=None):
    '''
    query:查询值矩阵
    key:键值矩阵
    value:真值矩阵
    '''
    d_k=query.size(-1)  #键向量的维度与值向量相同
    #计算Q与K内积并除以根号d_k
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)#transpose交换倒数第二维和最后一维
    #soft
    p_attn=scores.softmax(dim=-1)   #scores的形状是scores.shape = (batch, seq_len_q, seq_len_k)，dim=-1最后一维：沿着key做softmax
	if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
```
#### 自注意力
使用自注意力机制能够在同一输入序列中建模每个元素之间的关系，而无需依赖外部的 Key 和 Value。通过这种方式，能够有效捕捉序列中各个 token 之间的相关性，从而充分挖掘序列内部的依赖信息，尤其是在长文本中，能够更好地处理元素之间的长期依赖关系

```
attention(x,x,x)
```

#### 掩码自注意力
掩码自注意力 = 并行计算的自注意力 + 强制屏蔽未来信息 = 又快又符合自回归规律的注意力机制
生成Mask矩阵

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵，
# query 的长度、key 的长度、全部填充 float("-inf"),float("-inf") 在 PyTorch 里就是 -∞
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
```

在注意力计算时，我们会将计算得到的注意力分数与这个掩码做和，再进行 Softmax 操作

```python
# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```
通过做求和，上三角区域（也就是应该被遮蔽的 token 对应的位置）的注意力分数结果都变成了 -inf，而下三角区域的分数不变。再做 Softmax 操作，-inf 的值在经过 Softmax 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽

#### 多头注意力

