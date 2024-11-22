# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask
from scaling_zip import ScaledLinear
from base import BaseModule
import torch.nn.functional as F
from joiner_zip import Joiner
import transformer_3_0.Constants as Constants
from transformer_3_0.Models_1 import Encoder, Decoder
from transformer_3_0.Layers import Linear
from transformer_3_0.Layers import FFTBlock, ConvFFTBlock

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class AsrModel(nn.Module):
    def __init__(
        self,
        # encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        # self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.f0_conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(512, 1024, 3, 1, 1), 
            torch.nn.GroupNorm(8, 1024), 
            Mish())
        
        self.f0_fft1 = nn.ModuleList([
            ConvFFTBlock(d_model=1024, d_inner=1024, n_head=4, d_k=1024, d_v=1024, l_n=2, cln_ffd=False, dropout=0.1),
            ConvFFTBlock(d_model=1024, d_inner=1024, n_head=4, d_k=1024, d_v=1024, l_n=2, cln_ffd=False, dropout=0.1)])
            

        # self.f0_conv2 = torch.nn.Sequential(
        #     torch.nn.Conv1d(512, 1024, 3, 2, 1), 
        #     torch.nn.GroupNorm(8, 1024), 
        #     Mish())
        
        # self.f0_fft2 = FFTBlock(d_model=1024, d_inner=1024, n_head=4, d_k=1024, d_v=1024, l_n=2, cln_ffd=False, dropout=0.1)
        
        # f0 embedding
        self.min_pitch = 0
        self.max_pitch = 1600
        self.pitch_step = 1
        self.pitch_bin_size = int((self.max_pitch - self.min_pitch) / self.pitch_step)
        
        self.pitch_emb = nn.Embedding(
            self.pitch_bin_size, 512, padding_idx=0
        )
        
        self.res_conv = torch.nn.Conv1d(512, 1024, 3, 1, 1)
        self.layer_norm = torch.nn.LayerNorm(1024)
        # f0 joiner
        self.output_linear = nn.Linear(512, 1024)
        
        # self.project_hubert = ScaledLinear(1024, 512, initial_scale=0.25)
        # self.project_pitch = ScaledLinear(1024, 512, initial_scale=0.25)

        # self.pitch_joint = ConvFFTBlock(d_model=1024, d_inner=1024, n_head=4, d_k=1024, d_v=1024, l_n=6, cln_ffd=False, dropout=0.1)
        self.pitch_joint = nn.ModuleList([ConvFFTBlock(d_model=1024, d_inner=1024, n_head=4, d_k=1024, d_v=1024, l_n=2, cln_ffd=False, dropout=0.1)
                                           for _ in range(6)])
        self.layer_norm2 = torch.nn.LayerNorm(1024)

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_pit_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor,is_training: True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        # x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        # src_key_padding_mask = make_pad_mask(x_lens)
        # x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, is_training=is_training)

        # encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)
        
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        f0: torch.Tensor,
        f0_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        # assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens, is_training=is_training)
        
        f0_mask = make_pad_mask(f0_lens)
        f0_mask = ~f0_mask
        f0_mask = f0_mask.to(torch.int)

        # convert int f0_lens into bool
        f0_mask = f0_mask.unsqueeze(-1)

        # f0 embedding
        f0 = self.pitch_emb(f0.squeeze(-1))
        f0 = f0.transpose(1, 2)

        # f0 encoder conv subsampling 1 
        h = self.f0_conv1(f0) 
        
        # f0 encoder transformer 1
        h = h.transpose(1, 2)
        # h = self.f0_fft1(h, non_pad_mask=f0_mask, slf_attn_mask=None)
        h_local_output = h
        for f0_layer in self.f0_fft1:
            h, h_local_output = f0_layer(
                h,
                local_output=h_local_output,
                non_pad_mask=f0_mask,
                slf_attn_mask=None)
            
        h = h.transpose(1, 2)
    
        f0_emb = h + self.res_conv(f0)
        f0_emb = f0_emb.transpose(1, 2)
        f0_emb = self.layer_norm(f0_emb)
        # hubert features has the same size as f0_emb
        if encoder_out.shape[1] != f0_emb.shape[1]:
            if encoder_out.shape[1] > f0_emb.shape[1]:
                pad_length = encoder_out.shape[1] - f0_emb.shape[1]
                f0_emb = F.pad(f0_emb, pad=(0,0,0,pad_length))
            else:
                pad_length = f0_emb.shape[1] - encoder_out.shape[1]
                encoder_out = F.pad(encoder_out, pad=(0,0,0,pad_length))
                
        # A1. add on
        encoder_out = encoder_out + f0_emb
        encoder_out = self.layer_norm2(encoder_out)

        # A2. project
        # encoder_out = self.project_hubert(encoder_out) + self.project_pitch(f0_emb)
        # B. attention
        # encoder_out = self.output_linear(torch.tanh(encoder_out))

        # C. FFT
        # encoder_out = self.pitch_joint(encoder_out, local_output=encoder_out, non_pad_mask=f0_mask, slf_attn_mask=None)
        # enc_slf_attn_list = []
        
        # D. ConvFFT
        dec_local_output = encoder_out
        for enc_layer in self.pitch_joint:
            encoder_out, dec_local_output = enc_layer(
                encoder_out,
                local_output=dec_local_output,
                non_pad_mask=f0_mask,
                slf_attn_mask=None)
            
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss