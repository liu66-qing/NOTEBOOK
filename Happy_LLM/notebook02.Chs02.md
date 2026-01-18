#### 注意力机制
自然语言处理领域，把重点注意力集中在一个或几个 token就可以而取得更高效高质的计算效果

注意力机制核心变量：Query（查询值）、Key（键值）、Value（真值）

Q和K进行运算得到权重，表示从 Query 出发，对文本每一个 token 应该分布的注意力相对大小


现在得到的 x 即反映了 Query 和每一个 Key 的相似程度，再通过一个 Softmax 层将其转化为和为 1 的权重：


通过把权重和 Value 进行运算)：从 Query 出发计算整个文本注意力得到的结果
```
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
QKV都是由同一个输入通过对*Wq*、*Wk*、*Wv*做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系
```
attention(x,x,x)
```
