#### 注意力机制
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
