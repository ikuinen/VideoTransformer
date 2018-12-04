import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionEncoder(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]) \
            .unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = torch.sin(torch.add(torch.mul(pos, freqs), phases))
        self.pos_encoding = self.pos_encoding.transpose(1, 0)
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)

    def forward(self, x):
        return x + self.pos_encoding


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None, element_width=None, head_width=None):
        super().__init__()
        self.dropout = dropout
        self.element_width = element_width
        self.head_width = head_width

    # Q: [nb, nh, len1, hid1], K: [nb, nh, len2, hid2], V: [nb, nh, len2, hid2], mask: [nb, len2]
    def forward(self, Q, K, V, mask=None):
        if self.head_width is None:
            if self.element_width is not None:
                K = F.pad(K, (0, 0, self.element_width, self.element_width))
                V = F.pad(V, (0, 0, self.element_width, self.element_width))
                mask = F.pad(mask, (self.element_width, self.element_width))
            out = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))  # [nb, nh, len1, len2]
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
                out = out.masked_fill(mask == 0, -1e30)  # [nb, nh, len1, len2]
            if self.element_width is not None:
                mask = torch.zeros(out.size(-2), out.size(-1))
                for i in range(self.element_width, out.size(-2) - self.element_width):
                    mask[i, i-self.element_width:i+self.element_width+1] = 1
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(0)
                mask = mask.cuda()
                out = out.masked_fill(mask == 0, -1e30)
            attn = F.softmax(out, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)
            out = torch.matmul(attn, V)  # [nb, nh, len1, hid2]
        else:
            if self.element_width is not None:
                K = F.pad(K, (0, 0, self.element_width, self.element_width))
                V = F.pad(V, (0, 0, self.element_width, self.element_width))
                mask = F.pad(mask, (self.element_width, self.element_width))
            if self.head_width is not None:
                K = F.pad(K, (0, 0, 0, 0, self.head_width, self.head_width))
                V = F.pad(V, (0, 0, 0, 0, self.head_width, self.head_width))
                Q = F.pad(Q, (0, 0, 0, 0, self.head_width, self.head_width))
            element_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
            if self.element_width is not None:
                element_mask = torch.zeros(Q.size(-2), K.size(-2))
                for i in range(self.element_width, Q.size(-2) - self.element_width):
                    element_mask[i, i-self.element_width:i+self.element_width+1] = 1
                element_mask = element_mask.unsqueeze(0)  # [1, len1, len2]
                element_mask = element_mask.unsqueeze(0)  # [1, 1, len1, len2]
                element_mask = element_mask.cuda()

            num_heads = Q.size(1)
            attn_matrices = []
            K_T = K.transpose(-2, -1)
            for h in range(self.head_width, num_heads - self.head_width):
                attn_matrix = torch.matmul(Q[:, h:h+1], K_T[:, h-self.head_width:h+self.head_width]) / math.sqrt(Q.size(-1))
                # h->h-n...h+n: [nb, nrh, len1, len2]
                if mask is not None:
                    attn_matrix = attn_matrix.masked_fill(mask == 0, -1e30)
                if element_mask is not None:
                    attn_matrix = attn_matrix.masked_fill(element_mask == 0, -1e30)
                attn_matrices.append(attn_matrix)

            # softmax
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                attn_matrix = attn_matrices[i]
                nb, nrh, len1, len2 = attn_matrix.shape
                attn_score = F.softmax(attn_matrix.transpose(1, 2).contiguous().view(nb, len1, nrh * len2), -1)
                attn_score = attn_score.transpose(1, 2).contiguous().view(nb, nrh, len1, len2)
                attn_matrices[i] = attn_score

            outs = []
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                out = torch.matmul(attn_matrices[i],
                                   V[:, h-self.head_width:h+self.head_width])  # [nb, nrh, len1, hid2]
                out = torch.sum(out, 1)  # [nb, len1, hid2]
                outs.append(out)
                # print(out.shape)
            out = torch.stack(outs, 0).transpose(0, 1)
            # outs: [nb, nh, len1, hid2]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, element_width=None, head_width=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.element_width = element_width
        self.head_width = head_width
        self.attn = ScaledDotProductAttention(self.dropout, element_width=element_width, head_width=head_width)

    def forward(self, Q, K, V, mask=None):
        # Q: [nb, len1, d_model], K: [nb, len2, d_model], V: [nb, len2, d_model]
        num_batches = Q.size(0)
        Q, K, V = [
            l(x).view(num_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (Q, K, V))
        ]
        # Q: [nb, nh, len1, d_k], K: [nb, nh, len2, d_k], V: [nb, nh, len2, d_k]
        x = self.attn(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.d_k)
        # [nb, len1, d_model]
        return self.linears[-1](x)



class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        res = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.feed_forward(self.norm2(x))
        x = self.dropout(x)
        x = res + x
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class MyEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(layers[0].size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.norms = clones(LayerNorm(size), 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, target_mask):
        res = x
        x = self.norms[0](x)
        x = self.self_attn(x, x, x, target_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norms[1](x)
        x = self.src_attn(x, memory, memory, src_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.feed_forward(self.norms[2](x))
        x = self.dropout(x)
        x = res + x
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class MyDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(layers[0].size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_size, tgt_size, num_heads, d_model, d_ff, n_encoder_layers, n_decoder_layers):
        super().__init__()
        self.encoder = Encoder(EncoderLayer(src_size,
                                            MultiHeadAttention(num_heads, d_model),
                                            PositionWiseFeedForward(d_model, d_ff)), n_encoder_layers)
        self.decoder = Decoder(DecoderLayer(tgt_size,
                                            MultiHeadAttention(num_heads, d_model),
                                            MultiHeadAttention(num_heads, d_model),
                                            PositionWiseFeedForward(d_model, d_ff)), n_decoder_layers)

    def forward(self, src, src_mask, tgt, tgt_mask):
        x = self.encoder(src, src_mask)  # [nb, len1, hid]
        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        return x


if __name__ == '__main__':
    nb = 1
    nh = 1
    len1 = 4
    len2 = 3

    a = torch.randn(nb, 1, 1, len2)
    print(a)
    b = torch.randn(nb, nh, len1, len2)
    print(b.masked_fill(a >= 0, -1e30))