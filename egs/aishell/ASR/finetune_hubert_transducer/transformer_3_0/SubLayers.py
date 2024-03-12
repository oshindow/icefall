'''
Description: self attention 
Version: 2.0
Autor: Chunhui Wang
Email: wangchunhui@xiaobing.ai
Date: 2021-12-13 14:17:02
LastEditors: Chunhui Wang
LastEditTime: 2021-12-13 14:17:03
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer_3_0.Modules import ScaledDotProductAttention


class Condional_LayerNorm(nn.Module):

    def __init__(self,
                normal_shape,
                epsilon=1e-5
                ):
        super(Condional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = 384
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale
        y += bias
        # y *= scale.unsqueeze(1)
        # y += bias.unsqueeze(1)

        return y
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, cln=False, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.qs = nn.Linear(d_model, n_head * d_k)
        self.ks = nn.Linear(d_model, n_head * d_k)
        self.vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, non_pad_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        repeat_non_pad_mask = non_pad_mask.repeat(n_head, 1, 1)
        # print('mask:', mask.shape, mask)
        # print('repeat_non_pad_mask:', repeat_non_pad_mask.shape, repeat_non_pad_mask)
        q = q * repeat_non_pad_mask
        k = k * repeat_non_pad_mask
        v = v * repeat_non_pad_mask
        # print('q:', q)
        output, attn = self.attention(q, k, v, mask=mask)
        # print(output)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output * non_pad_mask

        output = self.layer_norm(output + residual)
        
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, cln=False, dropout=0.1):
        super().__init__()

        # Use Conv1D

        # position-wise
        self.w_1 = nn.Conv1d(
            d_in , d_hid, kernel_size=3, padding=1)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=3, padding=1)

        if cln:
            self.spk_embedding = True
            self.layer_norm = Condional_LayerNorm(d_in)
        else:
            self.spk_embedding = False
            self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x, cond_embedding=None, mask=None):
        residual = x
        output = x.transpose(1, 2)
        output = F.relu(self.w_1(output))
        if mask is not None:
            output = output.transpose(1, 2)
            output = output * mask
            output = output.transpose(1, 2)
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)

        if self.spk_embedding and cond_embedding is not None:
            output = self.layer_norm(output + residual, cond_embedding)
        else:
            output = self.layer_norm(output + residual)

        return output
