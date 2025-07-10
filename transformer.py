import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=80):
        '''
        d_model: 输入的特征维度
        max_len: 最大序列长度
        '''
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / d_model)))
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        '''
        d_model: 输入和输出的特征维度
        nhead: 注意力头的数量
        dropout: dropout的概率
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead # 每个注意力头的维度
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        '''
        query: 查询向量，形状为 (batch_size, seq_len, d_model)
        key: 键向量，形状为 (batch_size, seq_len, d_model)
        value: 值向量，形状为 (batch_size, seq_len, d_model)
        mask: 掩码，形状为 (batch_size, 1, seq_len, seq_len)
        dropout: dropout层
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) # qk/sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9) 
        
        # attn_weights的形状为 (batch_size, nhead, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        
        # context的形状为 (batch_size, nhead, seq_len, head_dim)
        context = torch.matmul(attn_weights, value)
        
        return context
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分割成多个头
        # query, key, value的形状为 (batch_size, seq_len, d_model)
        # 转换为 (batch_size, seq_len, nhead, head_dim)
        # 然后转置为 (batch_size, nhead, seq_len, head_dim)
        query = self.q_linear(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = self.attention(query, key, value, mask, self.dropout)

        # 将多个头的输出拼接起来
        # scores的形状为 (batch_size, nhead, seq_len, head_dim)
        # 转置为 (batch_size, seq_len, nhead, head_dim)
        # 然后变形为 (batch_size, seq_len, d_model)
        context = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        
        return output
    
class FeedForward(nn.Modules):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        '''
        d_model: 输入和输出的特征维度
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout的概率
        '''
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        '''
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        '''
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        '''
        d_model: 输入的特征维度
        eps: 防止除零的极小值
        '''
        super(NormLayer, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        '''
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        '''
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        '''
        d_model: 输入和输出的特征维度
        nhead: 注意力头的数量
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout的概率
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm1 = NormLayer(d_model)
        self.norm2 = NormLayer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        '''
        src: 输入张量，形状为 (batch_size, seq_len, d_model)
        mask: 掩码，形状为 (batch_size, 1, seq_len, seq_len)
        '''
        src = src + self.dropout1(self.self_attn(src, src, src, mask))
        src = self.norm1(src)
        src = src + self.dropout2(self.ffn(src))
        src = self.norm2(src)
        return src
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dropout):
        '''
        vocab_size: 词汇表大小
        d_model: 输入和输出的特征维度
        num_layers: 编码器层的数量
        nhead: 注意力头的数量
        dropout: dropout的概率
        '''
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.norm = NormLayer(d_model)

    def forward(self, src, mask=None):
        '''
        src: 输入张量，形状为 (batch_size, seq_len)
        mask: 掩码，形状为 (batch_size, 1, seq_len, seq_len)
        '''
        src = self.embedding(src)
        src = self.pe(src)  # 添加位置编码
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        '''
        d_model: 输入和输出的特征维度
        nhead: 注意力头的数量
        dropout: dropout的概率
        '''
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm1 = NormLayer(d_model)
        self.norm2 = NormLayer(d_model)
        self.norm3 = NormLayer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        '''
        tgt: 目标张量，形状为 (batch_size, seq_len, d_model)
        memory: 编码器的输出，形状为 (batch_size, src_seq_len, d_model)
        tgt_mask: 目标掩码，形状为 (batch_size, 1, seq_len, seq_len)
        memory_mask: 编码器输出的掩码，形状为 (batch_size, 1, src_seq_len, src_seq_len)
        '''
        tgt = tgt + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask))
        tgt = self.norm1(tgt)
        
        tgt = tgt + self.dropout2(self.cross_attn(tgt, memory, memory, memory_mask))
        tgt = self.norm2(tgt)
        
        tgt = tgt + self.dropout3(self.ffn(tgt))
        tgt = self.norm3(tgt)
        
        return tgt
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dropout):
        '''
        vocab_size: 词汇表大小
        d_model: 输入和输出的特征维度
        num_layers: 解码器层的数量
        nhead: 注意力头的数量
        dropout: dropout的概率
        '''
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.norm = NormLayer(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        '''
        tgt: 目标张量，形状为 (batch_size, seq_len)
        memory: 编码器的输出，形状为 (batch_size, src_seq_len, d_model)
        tgt_mask: 目标掩码，形状为 (batch_size, 1, seq_len, seq_len)
        memory_mask: 编码器输出的掩码，形状为 (batch_size, 1, src_seq_len, src_seq_len)
        '''
        tgt = self.embedding(tgt)
        tgt = self.pe(tgt)  # 添加位置编码
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dropout=0.1):
        '''
        vocab_size: 词汇表大小
        d_model: 输入和输出的特征维度
        num_layers: 编码器和解码器层的数量
        nhead: 注意力头的数量
        dropout: dropout的概率
        '''
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, nhead, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_layers, nhead, dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        '''
        src: 输入张量，形状为 (batch_size, src_seq_len)
        tgt: 目标张量，形状为 (batch_size, tgt_seq_len)
        src_mask: 输入掩码，形状为 (batch_size, 1, src_seq_len, src_seq_len)
        tgt_mask: 目标掩码，形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        '''
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        output = self.output_layer(output)  # 输出层将解码器的输出转换
        output = F.log_softmax(output, dim=-1)  # 应用softmax
        return output
