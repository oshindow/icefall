'''
Description: base module for gl ac
Version: 2.0
Autor: Chunhui Wang
Email: wangchunhui@xiaobing.ai
Date: 2021-12-13 14:15:53
LastEditors: Chunhui Wang
LastEditTime: 2021-12-13 14:21:09
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
# from transformer_3_0.BAM import BAM
from transformer_3_0.SubLayers import MultiHeadAttention, PositionwiseFeedForward, Condional_LayerNorm


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class PreNet(nn.Module):
    """
    Pre Net before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x



class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.LayerNorm(n_feats))
            if i == 0:
                m.append(act)


        #self.body = nn.Sequential(*m)
        self.body = nn.ModuleList(m)
        self.res_scale = res_scale
        self.layernorm = nn.LayerNorm(n_feats)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #res = self.body(x).mul(self.res_scale)
        res = x
        for i,layer in enumerate(self.body):
            if i in [1, 4]:
                res = res.transpose(1,2)
                res = layer(res)
                res = res.transpose(1,2)
            else:
                res = layer(res)
        res *= self.res_scale
        #res = self.dropout(res)
        res += x

        res = res.transpose(1,2)
        res = self.layernorm(res)
        res = res.transpose(1,2)
        
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 l_n,
                 cln_ffd=False,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.l_n = l_n
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, cln=cln_ffd, dropout=dropout)
        #conv, n_feats, kernel_size,bias=True, bn=True, act=nn.ReLU(True), res_scale=1
        self.local = nn.ModuleList([ResBlock(
                    default_conv, 384, 3, True, True, nn.ReLU(True), res_scale=1/l_n
                    ) for _ in range(l_n)])

        self.layer_norm = nn.LayerNorm(d_model) 

        #self.fc = nn.Linear(d_model * 2, d_model)
        #nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, local_output=None, cond_embedding=None, non_pad_mask=None, slf_attn_mask=None):
        global_input = enc_input
        # print(global_input)
        # print('slf_attn_mask:', slf_attn_mask)
        global_output, enc_slf_attn = self.self_attn(global_input, global_input, global_input,
                                        mask=slf_attn_mask, non_pad_mask=non_pad_mask)
        if local_output:
            local_output = local_output.transpose(1,2)
            for i, layer in enumerate(self.local):
                local_output = layer(local_output)

            local_output = local_output.transpose(1,2)
            enc_output = self.layer_norm(global_output + local_output)
            local_output = local_output * non_pad_mask
        else:
            # print('non local_output')
            enc_output = self.layer_norm(global_output)
        enc_output = enc_output * non_pad_mask 
        #base_feat *= non_pad_mask

        enc_output = self.pos_ffn(enc_output, cond_embedding=cond_embedding, mask=non_pad_mask)
        enc_output = enc_output * non_pad_mask
        
        if local_output:
            return enc_output, enc_slf_attn, local_output
        else:
            return enc_output

class ConvFFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 l_n,
                 cln_ffd=False,
                 dropout=0.1):
        super(ConvFFTBlock, self).__init__()
        self.l_n = l_n
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, cln=cln_ffd, dropout=dropout)
        #conv, n_feats, kernel_size,bias=True, bn=True, act=nn.ReLU(True), res_scale=1
        self.local = nn.ModuleList([ResBlock(
                    default_conv, 1024, 3, True, True, nn.ReLU(True), res_scale=1/l_n
                    ) for _ in range(l_n)])

        self.layer_norm = nn.LayerNorm(d_model) 

        #self.fc = nn.Linear(d_model * 2, d_model)
        #nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, local_output=None, cond_embedding=None, non_pad_mask=None, slf_attn_mask=None):
        global_input = enc_input
        
        # print(global_input)
        # print('slf_attn_mask:', slf_attn_mask)
        global_output, enc_slf_attn = self.self_attn(global_input, global_input, global_input,
                                        mask=slf_attn_mask, non_pad_mask=non_pad_mask)
        
        local_output = local_output.transpose(1,2)
        for i, layer in enumerate(self.local):
            local_output = layer(local_output)

        local_output = local_output.transpose(1,2)
        enc_output = self.layer_norm(global_output + local_output)

        local_output = local_output * non_pad_mask
        enc_output = enc_output * non_pad_mask 
        #base_feat *= non_pad_mask

        enc_output = self.pos_ffn(enc_output, cond_embedding=cond_embedding, mask=non_pad_mask)
        enc_output = enc_output * non_pad_mask
        
        return enc_output, local_output
        
class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal

class DeConv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 output_padding=0,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param output_padding: the additional size added to one side of the output shape
        :param w_init: str. weight inits with xavier initialization.
        """
        super(DeConv, self).__init__()

        self.deconv = nn.ConvTranspose1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.deconv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.deconv(x)
        x = x.contiguous().transpose(1, 2)
        return x
