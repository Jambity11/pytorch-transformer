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
>
> lambda 的作用：**包装成一个只接收单个参数的函数，把固定的 src_mask 捕获进去**

## 组装成 Encoder

```python
def forward(self, x, mask):
    for layer in self.layers:
        x = layer(x, mask)
    return self.norm(x)
```


## Dncoder Block

```python
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```

## Projection Layer(最后的线性层)

```python
def forward(self, x):
    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_Len)
    return torch.log_softmax(self.proj(x), dim = -1)
```

`d_model` 维映射成 `vocab_size` 个概率


## Transformer

传入的参数：

- `encoder`: 编码器，多层 EncoderLayer + 最终 Norm
- `decoder`: 解码器，多层 DncoderLayer + 最终 Norm
- `src_embed`: 把 token 映射为 d_model 维向量
- `tgt_embed`: 解码器端的词嵌入
- `src_pos`: 给嵌入加位置信息
- `tgt_pos`: 解码器端的位置编码
- `projection_layer`: 把隐藏特征映射到词表概率

把所有 `Transformer` 所需要的组件都包装在一个类里

## Build Transformer 设置 `Transformer` 中用到的超参数

```python
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, 
                      d_model: int=512, N: int=6, h: int=8, 
                      dropout: float=0.1, d_ff: int=2048) -> Transformer:
```

- `N`: 重复的层数

## 构建训练代码


> 准备数据、分词器


### Tokenizer

涉及代码：

- `train.py`


## 把数据（双语文本）转换成 Transformer 可以训练的格式

转换数据的工具会在 `train.py` 中使用

一般继承 `Dataset` 类数据就能被 `DataLoader` 批量读取

数据格式：

```
[
 {
  "translation":{
      "en":"I love you",
      "zh":"我爱你"
  }
 }
]
```

`Dataset` 的作用就是把他转换成：

```
encoder_input:
[101, 23, 56, 89, 102, 0,0,0]

decoder_input:
[101, 77, 88, 99,0,0,0]

label:
[77,88,99,102,0,0,0]
```

的形式

### class BilingualDataset

```python
def __init__(
    self,
    ds,               # 原始数据
    tokenizer_src,    # 源语言 tokenizer
    tokenizer_tgt,    # 目标语言 tokenizer
    src_lang,         # 源语言 en
    tgt_lang,         # 目标语言 zh
    seq_len           # 句子最大长度
):
```

## Config 配置训练任务

涉及代码：

- `config.py`

```python
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
```
