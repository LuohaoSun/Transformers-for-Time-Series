"""
各种位置嵌入.**注意**:
返回包含位置信息的嵌入表达,而不是位置编码(为了兼容加性\乘性等各种编码方式).
"""

import math

import torch
import torch.nn as nn


class PosEmbedding(object):
    """位置编码词典,通过字符串初始化一个已定义的位置编码实例"""

    def __new__(cls, posEmbeddingType="sin", d_model=512, max_len=512):
        pos_embedding_dic = {
            "sin": SinPosEmbedding,
            "addLearnable": AddLearnablePosEmbedding,
            "add": AddLearnablePosEmbedding,
            "learnable": AddLearnablePosEmbedding,
            "Learnable": AddLearnablePosEmbedding,
            "multLearnable": MultLearnablePosEmbedding,
            "mult": MultLearnablePosEmbedding,
            "complexLearnable": ComplexLearnablePosEmbedding,
            "complex": ComplexLearnablePosEmbedding,
            "ro": RotaryPosEmbedding,
            "rotary": RotaryPosEmbedding,
            "RoPE": RotaryPosEmbedding,
            "rope": RotaryPosEmbedding,
        }
        return pos_embedding_dic[posEmbeddingType](d_model, max_len)


class SinPosEmbedding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super().__init__()
        assert d_model % 2 == 0
        # Create a positional embedding tensor with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(
            10000.0, -torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (B x T x C) with B=1 and register as a non-trainable buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: Tensor with shape (Batch, Time, Channel)
        x_pe = self.pe[:, : x.size(1)]
        # The addition operation automatically broadcasts the tensors
        return x + x_pe


class AddLearnablePosEmbedding(nn.Module):
    """加性可学习位置编码"""

    def __init__(self, d_model, max_len=200):
        super().__init__()

        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):

        position_ids = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        pe = self.pe(position_ids)

        return x + pe


class MultLearnablePosEmbedding(nn.Module):
    """乘性可学习位置编码"""

    def __init__(self, d_model, max_len=200):
        super().__init__()

        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):

        position_ids = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        pe = self.pe(position_ids)

        return x * pe


class ComplexLearnablePosEmbedding(nn.Module):
    """复合可学习位置编码,同时包含加性和乘性位置编码"""

    def __init__(self, d_model, max_len=200) -> None:
        super().__init__()
        self.add_pe = nn.Embedding(max_len, d_model)
        self.mult_pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        position_ids = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        add_pe = self.add_pe(position_ids)
        mult_pe = self.mult_pe(position_ids)
        return x * mult_pe + add_pe


class RotaryPosEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, d_model, max_len=200):
        super(RotaryPosEmbedding, self).__init__()
        assert d_model % 2 == 0
        self.dim = d_model
        self.max_len = max_len
        self.rotary_emb = self.create_rotary_embedding()

    def create_rotary_embedding(self):
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2) * -(math.log(10000.0) / self.dim)
        )
        phase = position * div_term
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def apply_rotary_embedding(self, x, rotary_emb):
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotary_emb1, rotary_emb2 = rotary_emb[..., ::2], rotary_emb[..., 1::2]
        return torch.cat(
            (
                (x1 * rotary_emb1 - x2 * rotary_emb2),
                (x1 * rotary_emb2 + x2 * rotary_emb1),
            ),
            dim=-1,
        )

    def forward(self, x):
        seq_len = x.shape[1]
        rotary_emb = self.rotary_emb[:seq_len, :].to(x.device)
        return self.apply_rotary_embedding(x, rotary_emb)
