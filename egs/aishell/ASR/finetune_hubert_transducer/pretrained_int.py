#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                2022  Xiaomi Corp                   Xiaoyu Yang)
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
"""
Usage:

(1) greedy search
./finetune_hubert_transducer/pretrained.py \
    --checkpoint ./finetune_hubert_transducer/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method greedy_search \
    /path/to/foo.wav \
    /path/to/bar.wav

(2) modified beam search
./finetune_hubert_transducer/pretrained.py \
    --checkpoint ./finetune_hubert_transducer/exp/pretrained.pt \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --method modified_beam_search \
    --beam-size 4 \
    /path/to/foo.wav \
    /path/to/bar.wav

You can also use `./finetune_hubert_transducer/exp/epoch-xx.pt`.

Note: ./finetune_hubert_transducer/exp/pretrained.pt is generated by
./finetune_hubert_transducer/export.py
"""


import argparse
import logging
import math
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from beam_search_score import greedy_search, greedy_search_batch, modified_beam_search
from hubert_encoder import HubertEncoder
from torch.nn.utils.rnn import pad_sequence
# from train import get_params, get_transducer_model
import kaldialign
# from hubert_encoder import HubertEncoder
from icefall.lexicon import Lexicon
from finetune_hubert_transducer.train_phone_model import get_params, get_transducer_model

from icefall.phone_graph_compiler import PhoneCtcTrainingGraphCompiler

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. " f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    import time
    begin_time = time.time()
    parser = get_parser()
    HubertEncoder.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(params.bpe_model)
    lexicon = Lexicon(params.bpe_model)
    graph_compiler = PhoneCtcTrainingGraphCompiler(
        params.bpe_model,
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )
    params.blank_id = lexicon.token_table["<eps>"]
    params.vocab_size = max(lexicon.tokens) + 1

    # <blk> is defined in local/train_bpe_model.py
    # params.blank_id = sp.piece_to_id("<blk>")
    # params.unk_id = sp.piece_to_id("<unk>")
    # params.vocab_size = sp.get_piece_size()

    logging.info(f"{params}")


    logging.info(f"device: {device}")

    logging.info("Creating model")
    start_time = time.time()
    params.hubert_model_dir = '/data2/xintong/pretrained_models/chinese-hubert-base-fairseq-ckpt.pt'
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    print(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    model.device = device
    end_time = time.time()
    print('load model cost:', end_time - start_time)
    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    start_time = time.time()
    waves_lengths = [w.size(0) for w in waves]

    # pad waveform input
    waves = pad_sequence(waves, batch_first=True, padding_value=math.log(1e-10))

    waves_lengths = torch.tensor(waves_lengths, device=device)

    # supervisions = batch["supervisions"]
    # feature_lens = supervisions["num_samples"].to(device)

    layer_results, encoder_out_lens = model.encoder(
        x=waves, x_lens=waves_lengths, is_training=False
    )
    if layer_results.ndim == 4:
        encoder_out = layer_results[-1]
    else:
        encoder_out = layer_results

    num_waves = encoder_out.size(0)
    hyps = []
    msg = f"Using {params.method}"
    if params.method == "beam_search":
        msg += f" with beam size {params.beam_size}"
    logging.info(msg)

    if params.method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )

        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens, logits_list = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            return_timestamps=False,
        )
        
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        # hyp_tokens: id
        # hyps: str
        # compute score 
        args.text = args.text.replace('*', 'a1')
        text_id = [lexicon.token_table[token] for token in args.text.split()]
        # text_id = text_id
        ali = kaldialign.align(text_id, hyp_tokens[0], 0, sclite_mode=False)
        # print(ali)
        p = []
        idx = 0
        # ali_idx = 0
        
        for ali_idx in range(len(ali)):
            # logits  
            # print(ali_idx,idx)
            token_id = ali[ali_idx][0]
            if ali[ali_idx][0] == 0: 
                # ali_idx += 1
                idx += 1
            elif ali[ali_idx][1] == 0:
                p.append((lexicon.token_table[0], lexicon.token_table[ali[ali_idx][0]], 0))
            else:
                logits = torch.nn.functional.softmax(logits_list[idx], dim=1)
                p.append((lexicon.token_table[ali[ali_idx][0]], lexicon.token_table[ali[ali_idx][1]], logits[0, token_id].item() * 100))
                idx += 1
        print(p)
        # # print(logits.shape)
        
        # logits_list.append(logits)
        # emitted = False
        # for i, v in enumerate(y):
        #     if v not in (blank_id, unk_id):
        #         hyps[i].append(v)
        #         timestamps[i].append(t)
        #         scores[i].append(logits[i, v].item())
        #         emitted = True
    else:
        for i in range(num_waves):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            else:
                raise ValueError(f"Unsupported method: {params.method}")

            for i in range(encoder_out.size(0)):
                hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)
    end_time = time.time()
    logging.info("Decoding Done")
    print('decode cost:', end_time - start_time)
    print('all cost:', end_time - begin_time)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
