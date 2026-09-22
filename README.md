# 动手实现 Transformer

![1790080570637](image/README/1790080570637.png)

## Embedding + Positional Encoding

**涉及代码：**

`model.py`: 
- `class InputEmbeddings()`
- `class PositionalEncoding()`

![1790077175180](image/README/1790077175180.png)

$$
e^{-\ln(10000)\cdot \frac{2i}{d} } = \frac{1}{10000^{\frac{2i}{d}}}
$$

所以实际写代码算的是$LHS$那种形式，可以高效计算

```python
torch.exp(
  torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
)
```

## Layer Normalization

**涉及代码：**

`model.py`: 
- `class LayerNormalization()`

![1790080878751](image/README/1790080878751.png)

$$
y = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

```python
  def forward(self, x):
      mean = x.mean(dim=-1, keepdim=True) # -1表示最后一维，d_model维
      std = x.std(dim=-1, keepdim=True)
      return self.alpha * (x - mean) / (std + self.eps) + self.bias
```


## Feed-Forward Layer**涉及代码：**

**涉及代码：**

`model.py`: 
- `class FeedForwardBlock()`

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

```python
def forward(self, x):
    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

## Multi-Head Attention

**涉及代码：**

`model.py`: 
- `class MultiHeadAttentionBlock()`

![1790083079937](image/README/1790083079937.png)

其中把$Q', K', V'$切成多头的操作：

```python
# (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
```

最后交换`dim 1`和`dim 2`是因为希望数据排列成：

```
- Batch 1
  - head 1
    - token 1
    - token 2
    - ...
  - head 2
    - ...
- ...
```

把不同头分成不同维度，然后矩阵乘法

### 计算注意力分数

$$
Attention(Q, K, V) = softmax(\frac{QK^\top}{\sqrt{d_{model}}}) V
$$


## Residual Connection

**涉及代码：**

`model.py`: 
- `class ResidualConnection()`


$$
output = x + Dropout(Sublayer(LayerNorm(x)))
$$

```python
return x + self.dropout(sublayer(self.norm(x)))
```

`sublayer`指的就是框起来的`Multi-Head Attetion`/`Feed Forward`

![1790089315342](image/README/1790089315342.png)

原始论文应该是先`sublayer`再`norm`的，实际反过来做也可以吧

## Encoder Block

**涉及代码：**

`model.py`: 
- `class EncoderBlock()`

![1790089516647](image/README/1790089516647.png)

- 1个多头注意力块
- 2个加和归一化
- 1个前馈神经网络

```python
def forward(self, x, src_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x
```

> lambda 参数列表: 返回表达式

## 组装成 Encoder

