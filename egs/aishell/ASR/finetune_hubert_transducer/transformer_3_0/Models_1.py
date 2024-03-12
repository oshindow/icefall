'''
Description: gl ac model 
Version: 2.0
Autor: Chunhui Wang
Email: wangchunhui@xiaobing.ai
Date: 2021-12-13 14:15:08
LastEditors: Chunhui Wang
LastEditTime: 2021-12-13 14:16:36
'''
import torch
import torch.nn as nn
import numpy as np

import transformer_3_0.Constants as Constants
from transformer_3_0.Layers import FFTBlock
from collections import OrderedDict

# Models_1(Encoder-Decoder) -> Layers(attention layer) -> SubLayers(attention module)

class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 dropout,
                 cln=False,
                 cln_ffd=False,
                 cln_all=False,
                 l_n=2):

        super(Encoder, self).__init__()
        
        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, l_n, cln_ffd=cln_ffd, dropout=dropout)] +
            [FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, l_n, cln_ffd=False, dropout=dropout) for _ in range(1, n_layers)])
        # self.layer_stack = nn.ModuleList([FFTBlock(
        #     d_model, d_inner, n_head, d_k, d_v, l_n, cln=cln, cln_ffd=False, cln_all=cln_all, dropout=dropout) for _ in range(0, n_layers -1)] +
        #     [FFTBlock(
        #     d_model, d_inner, n_head, d_k, d_v, l_n, cln=cln, cln_ffd=cln_ffd, cln_all=cln_all, dropout=dropout)])

    def forward(self, enc_output, slf_attn_mask, non_pad_mask, cond_embedding=None, return_attns=False):

        enc_slf_attn_list = []
        dec_local_output = enc_output

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn, dec_local_output = enc_layer(
                enc_output,
                local_output=dec_local_output,
                cond_embedding=cond_embedding, 
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, enc_slf_attn_list


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 dropout,
                 cln=False,
                 l_n=5):

        super(Decoder, self).__init__()

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, l_n, dropout=dropout) for _ in range(n_layers)])

    def forward(self, dec_output, slf_attn_mask, non_pad_mask, cond_embedding=None, return_attns=False, attention_mask=None, speakers=None):

        dec_slf_attn_list = []
        dec_local_output = dec_output
        total = len(self.layer_stack)

        for i, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_slf_attn, dec_local_output = dec_layer(
                    dec_output,
                    local_output=dec_local_output,
                    cond_embedding=cond_embedding, 
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)
            
            if i != total -1 and speakers is not None:
                dec_output += speakers
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, dec_slf_attn_list
