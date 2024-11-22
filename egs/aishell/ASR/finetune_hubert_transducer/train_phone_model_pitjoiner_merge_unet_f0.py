#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,
#                                                  Zengwei Yao,
#                                                  Xiaoyu Yang)
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

(1) 100h Large

export CUDA_VISIBLE_DEVICES="0,1,2,3"
./pruned_transducer_stateless6/train.py \
    --world-size 4 \
    --num-epochs 50 \
    --save-every-n 4000 \
    --exp-dir finetune_hubert_transducer/exp_100h_HubertL \
    --full-libri 0 \
    --max-duration 80 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --input-strategy AudioSamples \
    --hubert-model-dir /path/to/the/pretrained/model \
    --hubert-freeze-finetune-updates 10000 \
    --hubert-mask-channel-length 64 \
    --hubert-mask-prob 0.5 \
    --hubert-mask-channel-prob 0.5 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --encoder-dim 1024 \
    --enable-spec-aug 0 \
    --TSA-init-lr 5e-7 \
    --TSA-warmup-lr 3e-5 \
    --TSA-end-lr 1.5e-6 \
    --TSA-total-steps 80000 \
    --enable-musan 1


(2) 960h Large
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./pruned_transducer_stateless6/train.py \
    --world-size 8 \
    --num-epochs 25 \
    --exp-dir finetune_hubert_transducer/exp_960h_HubertL \
    --full-libri 1 \
    --max-duration 80 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --input-strategy AudioSamples \
    --hubert-model-dir /path/to/the/pretrained/model \
    --hubert-freeze-finetune-updates 10000 \
    --hubert-mask-channel-length 64 \
    --hubert-mask-prob 0.25 \
    --hubert-mask-channel-prob 0.5 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --encoder-dim 1024 \
    --enable-spec-aug 0 \
    --TSA-init-lr 5e-7 \
    --TSA-warmup-lr 3e-5 \
    --TSA-end-lr 1.5e-6 \
    --TSA-total-steps 320000 \
    --enable-musan 1
"""
import sys
sys.path.insert(0, '/home/xintong/lhotse')
import argparse
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
# from asr_datamodule import LibriSpeechAsrDataModule
from asr_datamodule_new import AishellAsrDataModule
from icefall.lexicon import Lexicon
from icefall.phone_graph_compiler import PhoneCtcTrainingGraphCompiler
from decoder_zip import Decoder
from hubert_encoder import HubertEncoder
from joiner_zip import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
# from model import Transducer
from model_zip_pitjoiner_unet_f0 import AsrModel
from optim import Eden, Eve, TriStateScheduler
# from optim_scaledadam import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12358,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless6/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_phone",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)" "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=8000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--encoder-type", type=str, default="hubert", help="Encoder type"
    )

    parser.add_argument(
        "--encoder-dim", type=int, default=1024, help="Encoder output dim"
    )

    parser.add_argument(
        "--use-tri-state-optim",
        type=str2bool,
        default=True,
        help="Whether to use tri state adam as optimizer",
    )

    parser.add_argument(
        "--TSA-init-lr", type=float, default=5e-7, help="TSA initial lr"
    )

    parser.add_argument(
        "--TSA-warmup-lr", type=float, default=3e-5, help="TSA warmup lr"
    )

    parser.add_argument("--TSA-end-lr", type=float, default=1.5e-6, help="TSA end lr")

    parser.add_argument(
        "--TSA-total-steps", type=int, default=80000, help="TSA total steps"
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - encoder_dim: Hidden dim for multi-head attention model.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for hubert transducer
            "encoder_dim": 1024,
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # only support hubert model right now
    encoder = HubertEncoder(
        model_dir=params.hubert_model_dir,
        output_size=params.encoder_dim,
        freeze_finetune_updates=params.hubert_freeze_finetune_updates,
        mask_prob=params.hubert_mask_prob,
        mask_channel_prob=params.hubert_mask_channel_prob,
        mask_channel_length=params.hubert_mask_channel_length,
        subsample_output=params.hubert_subsample_output,
        subsample_mode=params.hubert_subsample_mode,
        training=True,
    )

    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder

def get_pitch_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=256,
        decoder_dim=params.encoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder

def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    # pitch_decoder = get_pitch_decoder_model(params)
    joiner = get_joiner_model(params)

    # model = Transducer(
    #     encoder=encoder,
    #     decoder=decoder,
    #     joiner=joiner,
    #     encoder_dim=params.encoder_dim,
    #     decoder_dim=params.decoder_dim,
    #     joiner_dim=params.joiner_dim,
    #     vocab_size=params.vocab_size,
    # )
    model = AsrModel(
        # encoder_embed=encoder_embed,
        encoder=encoder,
        # pitch_decoder=pitch_decoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    # sp: spm.SentencePieceProcessor,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    batch: dict,
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute RNN-T loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Hubert
        transducer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]

    # at entry, feature is (N, T)
    assert feature.ndim == 2
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_samples"].to(device)

    texts = batch["supervisions"]["text"]
    f0s = batch['customs']['f0']
    f0s_length = batch['customs']['f0_length']
    
    # print(pitch.shape,type(pitch))
    # y = sp.encode(texts, out_type=int)
    y = graph_compiler.texts_to_ids(texts)
    y = k2.RaggedTensor(y).to(device)

    info = MetricsTracker()

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss,_ = model(
            x=feature,
            f0s=f0s,
            f0s_length=f0s_length,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            # reduction="none",
            # is_training=is_training,
        )
        simple_loss_is_finite = torch.isfinite(simple_loss)
        pruned_loss_is_finite = torch.isfinite(pruned_loss)
        is_finite = simple_loss_is_finite & pruned_loss_is_finite
        if not torch.all(is_finite):
            logging.info(
                "Not all losses are finite!\n"
                f"simple_loss: {simple_loss}\n"
                f"pruned_loss: {pruned_loss}"
            )
            display_and_save_batch(batch, params=params, sp=sp)
            simple_loss = simple_loss[simple_loss_is_finite]
            pruned_loss = pruned_loss[pruned_loss_is_finite]
            # If the batch contains more than 10 utterances AND
            # if either all simple_loss or pruned_loss is inf or nan,
            # we stop the training process by raising an exception
            if feature.size(0) >= 10:
                if torch.all(~simple_loss_is_finite) or torch.all(
                    ~pruned_loss_is_finite
                ):
                    raise ValueError(
                        "There are too many utterances in this batch "
                        "leading to inf or nan losses."
                    )

        simple_loss = simple_loss.sum()
        pruned_loss = pruned_loss.sum()
        # after the main warmup step, we keep pruned_loss_scale small
        # for the same amount of time (model_warm_step), to avoid
        # overwhelming the simple_loss and causing it to diverge,
        # in case it had not fully learned the alignment yet.
        pruned_loss_scale = (
            0.0 if warmup < 1.0 else (0.1 if warmup > 1.0 and warmup < 2.0 else 1.0)
        )
        loss = params.simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
        # if is_training and params.enable_distillation:
        #     assert codebook_loss is not None
        #     loss += params.codebook_loss_scale * codebook_loss

    assert loss.requires_grad == is_training

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # info["frames"] is an approximate number for two reasons:
        # (1) The acutal subsampling factor is ((lens - 1) // 2 - 1) // 2
        # (2) If some utterances in the batch lead to inf/nan loss, they
        #     are filtered out.
        info["frames"] = (feature_lens // 320).sum().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    # sp: spm.SentencePieceProcessor,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            # sp=sp,
            graph_compiler=graph_compiler,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    # sp: spm.SentencePieceProcessor,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    logging.warning("Test valid dataloader")
    valid_info = compute_validation_loss(
        params=params,
        model=model,
        # sp=sp,
        graph_compiler=graph_compiler,
        valid_dl=valid_dl,
        world_size=world_size,
    )
    logging.warning("Finished")

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                # sp=sp,
                graph_compiler=graph_compiler,
                batch=batch,
                is_training=True,
                warmup=(params.batch_idx_train / params.model_warm_step),
            )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.
        scaler.scale(loss).backward()
        scheduler.step_batch(params.batch_idx_train)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if params.print_diagnostics and batch_idx == 30:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )
            pass

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                # sp=sp,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()

    # Note: it's better to set --spec-aug-time-warp-factor=-1
    # when doing distillation with vq.
    assert args.spec_aug_time_warp_factor < 1

    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    graph_compiler = PhoneCtcTrainingGraphCompiler(
        params.lang_dir,
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
    print(model)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        # model_avg = copy.deepcopy(model)
        model_avg = get_transducer_model(
            params
        )  # dirty hack to avoid non-leaf node deepcopy
        model_avg.load_state_dict(model.state_dict())

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if params.use_tri_state_optim:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-08
        )
        scheduler = TriStateScheduler(
            optimizer,
            init_lr=params.TSA_init_lr,
            warmup_lr=params.TSA_warmup_lr,
            end_lr=params.TSA_end_lr,
            phase=[0.1, 0.4, 0.5],
            total_steps=params.TSA_total_steps,
        )
    else:
        optimizer = Eve(model.parameters(), lr=params.initial_lr)
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    # optimizer = ScaledAdam(
    #     get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
    #     lr=params.base_lr,  # should have no effect
    #     clipping_scale=2.0,
    # )

    # scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])
        logging.info(scheduler.state_dict())

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    # librispeech = LibriSpeechAsrDataModule(args)
    aishell = AishellAsrDataModule(args)

    train_cuts = aishell.train_cuts()
    valid_cuts = aishell.valid_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = aishell.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_dl = aishell.valid_dataloaders(valid_cuts)

    if False and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            # sp=sp,
            graph_compiler=graph_compiler,
            params=params,
            warmup=0.0 if params.start_epoch == 1 else 1.0,
        )

    scaler = GradScaler(enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            graph_compiler=graph_compiler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # sp: spm.SentencePieceProcessor,
    graph_compiler: PhoneCtcTrainingGraphCompiler,
    params: AttributeDict,
    warmup: float,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    # sp=sp,
                    graph_compiler=graph_compiler,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    HubertEncoder.add_arguments(parser)
    args = parser.parse_args()
    args.lang_dir = Path(args.lang_dir)
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()