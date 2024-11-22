#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang)
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
./finetune_hubert_transducer/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./finetune_hubert_transducer/exp_960h \
    --max-duration 600 \
    --decoding-method greedy_search \
    --hubert-model-dir ./path/to/pretrained_hubert_model \
    --encoder-dim 1024 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --use-averaged-model 1 \
    --input-strategy AudioSamples

(2) modified beam search
./finetune_hubert_transducer/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./finetune_hubert_transducer/exp_960h \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --hubert-model-dir ./path/to/pretrained_hubert_model \
    --encoder-dim 1024 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --use-averaged-model 1 \
    --input-strategy AudioSamples

(3) fast beam search
./finetune_hubert_transducer/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./finetune_hubert_transducer/exp_960h \
    --max-duration 600 \
    --decoding-method search \
    --beam-size 4 \
    --hubert-model-dir ./path/to/pretrained_hubert_model \
    --encoder-dim 1024 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --use-averaged-model 1 \
    --input-strategy AudioSamples
"""


import sys
sys.path.insert(0, '/home/xintong/lhotse')
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
# from asr_datamodule import LibriSpeechAsrDataModule
from asr_datamodule_new_40ms import AishellAsrDataModule
from beam_search import (
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from lhotse.cut import Cut
from hubert_encoder import HubertEncoder
from finetune_hubert_transducer.train_phone_model_f0_emb_conv_att_fft_40ms import get_params, get_transducer_model

from icefall.phone_graph_compiler import PhoneCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

import torch.nn.functional as F
from icefall.utils import add_sos, make_pad_mask
from model_zip_f0_emb_conv_att_fft_5ms import Mish


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="finetune_hubert_transducer/exp_960h",
        help="The experiment dir",
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_char",
        help="The lang dir containing word table and LG graph",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search, fast_beam_search_LG,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="Used only when --decoding-method is fast_beam_search*",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="Used only when --decoding-method fast_beam_search",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--encoder-dim", type=int, default=1024, help="Encoder output dim"
    )

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    # sp: spm.SentencePieceProcessor,
    lexicon: Lexicon,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    batch: dict,
    # word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 2

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_samples"].to(device)
    # pitch = batch['customs']['pitch'].long().to(device)
    f0 = batch['customs']['f0'][0].long().to(device)
    f0_lens = batch['customs']['f0_length'][0].to(device)
    f0 = torch.where(f0 < 0, 0, f0)
    
    layer_results, encoder_out_lens = model.encoder(
        x=feature, x_lens=feature_lens, is_training=False
    )
    if layer_results.ndim == 4:
        encoder_out = layer_results[-1]
    else:
        encoder_out = layer_results

    f0_mask = make_pad_mask(f0_lens)
    f0_mask = ~f0_mask
    f0_mask = f0_mask.to(torch.int)

    # if f0.size(1) % 4 != 0:
    #     pad_length = 4 - f0.size(1) % 4
    #     f0 = F.pad(f0, pad=(0,0,0,pad_length))
    #     f0_mask = F.pad(f0_mask, pad=(0,pad_length), value=0)

    # convert int f0_lens into bool
    f0_mask = f0_mask.unsqueeze(-1)

    # f0 embedding
    f0 = model.pitch_emb(f0.squeeze(-1))
    f0 = f0.transpose(1, 2)
    length = f0.shape[2]

    # f0 encoder conv subsampling 1 
    h = model.f0_conv1(f0) 
    # subsampled_length = length // 2 + 1
    # f0_mask = f0_mask[:,::2,:][:, :subsampled_length,:]
    
    # f0 encoder transformer 1
    h = h.transpose(1, 2)
    h = model.f0_fft1(h, non_pad_mask=f0_mask, slf_attn_mask=None)
    h = h.transpose(1, 2)

    f0_emb = h + model.res_conv(f0)
    f0_emb = f0_emb.transpose(1, 2)
    f0_emb = model.layer_norm(f0_emb)

    if encoder_out.shape[1] != f0_emb.shape[1]:
        if encoder_out.shape[1] > f0_emb.shape[1]:
            pad_length = encoder_out.shape[1] - f0_emb.shape[1]
            f0_emb = F.pad(f0_emb, pad=(0,0,0,pad_length))
        else:
            pad_length = f0_emb.shape[1] - encoder_out.shape[1]
            encoder_out = F.pad(encoder_out, pad=(0,0,0,pad_length))

    encoder_out = encoder_out + f0_emb
    encoder_out = model.output_linear(torch.tanh(encoder_out))
    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        if params.decoding_method == "fast_beam_search":
            # for hyp in sp.decode(hyp_tokens):
            #     hyps.append(hyp.split())
            for i in range(encoder_out.size(0)):
                hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        # for hyp in sp.decode(hyp_tokens):
        #     hyps.append(hyp.split())
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        # for hyp in sp.decode(hyp_tokens):
        #     hyps.append(hyp.split())
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
    
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i + 1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            # hyps.append(sp.decode(hyp).split())
            hyps.append([lexicon.token_table[idx] for idx in hyp])

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
        if "LG" in params.decoding_method:
            key += f"_ngram_lm_scale_{params.ngram_lm_scale}"
        return {key: hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    # sp: spm.SentencePieceProcessor,
    # word_table: Optional[k2.SymbolTable] = None,
    lexicon: Lexicon,
    lexicon2: Lexicon,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    graph_compiler2: PhoneCtcTrainingGraphCompiler,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 10

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        phone_ids = graph_compiler2.texts_to_ids(texts)
        # print('texts:', texts, 'phones:', phone_ids)
        texts = []
        for btc in phone_ids:
            texts.append([lexicon2.token_table[phone_id] for phone_id in btc])
        
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            # sp=sp,
            # decoding_graph=decoding_graph,
            lexicon=lexicon,
            graph_compiler=graph_compiler,
            # word_table=word_table,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                # ref_words = ref_text.split()
                this_batch.append((cut_id, ref_text, hyp_words))

            results[name].extend(this_batch)
            # print(results)
        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    # LibriSpeechAsrDataModule.add_arguments(parser)
    AishellAsrDataModule.add_arguments(parser)
    HubertEncoder.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "fast_beam_search",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
        if "LG" in params.decoding_method:
            params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    # sp = spm.SentencePieceProcessor()
    # sp.load(params.bpe_model)

    # # <blk> and <unk> are defined in local/train_bpe_model.py
    # params.blank_id = sp.piece_to_id("<blk>")
    # params.unk_id = sp.piece_to_id("<unk>")
    # params.vocab_size = sp.get_piece_size()
    lexicon = Lexicon(params.lang_dir)
    graph_compiler = PhoneCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )
    lexicon2 = Lexicon('data/lang_phone_latic')
    graph_compiler2 = PhoneCtcTrainingGraphCompiler(
        'data/lang_phone_latic',
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )
    params.blank_id = lexicon.token_table["<eps>"]
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    logging.info("Set hubert encoder training to false")
    model.encoder.training = False

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            print(f"{params.exp_dir}/epoch-{params.epoch}.pt")
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    if params.decoding_method == "fast_beam_search":
        word_table = None
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        word_table = None
        decoding_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    # librispeech = LibriSpeechAsrDataModule(args)
    aishell = AishellAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0
    # test_clean_cuts = librispeech.test_clean_cuts()
    # test_other_cuts = librispeech.test_other_cuts()

    # test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    # test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    # test_sets = ["test-clean", "test-other"]
    # test_dl = [test_clean_dl, test_other_dl]
    dev_cuts = aishell.valid_cuts()
    dev_cuts = dev_cuts.filter(remove_short_utt)
    dev_dl = aishell.valid_dataloaders(dev_cuts)

    test_cuts = aishell.test_cuts()
    test_cuts = test_cuts.filter(remove_short_utt)
    test_dl = aishell.test_dataloaders(test_cuts)
    
    # test_sets = ["dev", "test"]
    # test_dls = [dev_dl, test_dl]

    test_sets = ['test']
    test_dls = [test_dl]
    # with torch.no_grad():
    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            # sp=sp,
            # word_table=word_table,
            lexicon=lexicon,
            lexicon2=lexicon2,
            graph_compiler=graph_compiler,
            graph_compiler2=graph_compiler2,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()