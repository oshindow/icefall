# from finetune_hubert_transducer.train_phone_model_f0_emb_conv_att_mel_coarse import get_parser, get_params
import sys
sys.path.insert(0, '/home/xintong/lhotse')
from finetune_hubert_transducer.asr_datamodule_new import AishellAsrDataModule
from icefall.utils import (
    AttributeDict,
    str2bool,)
import numpy as np
import argparse
import matplotlib.pyplot as plt

def run(rank, world_size, args):
    params = get_params()

    # Note: it's better to set --spec-aug-time-warp-factor=-1
    # when doing distillation with vq.
    assert args.spec_aug_time_warp_factor < 1

    params.update(vars(args))
    aishell = AishellAsrDataModule(args)

    train_cuts = aishell.train_cuts()
    # if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
    #     # We only load the sampler's state dict when it loads a checkpoint
    #     # saved in the middle of an epoch
    #     sampler_state_dict = checkpoints["sampler"]
    # else:
    sampler_state_dict = None
        
    train_dl = aishell.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    f0s = []
    cnt = 0
    f0s_cnt = [0 for i in range(16)]
    for batch_idx, batch in enumerate(train_dl):
        f0 = batch['customs']['f0'][0].long() 
        f0_lens = batch['customs']['f0_length'][0]
        cnt += f0.shape[0]
        # if cnt and cnt % 100 == 0:
        #     print(cnt) 
        for it_idx in range(f0.shape[0]):
            # print(f0[it_idx].shape)
            cnt += 1
            if cnt and cnt % 1000 == 0:
                print(cnt) 

            raw_f0 = f0[it_idx][:f0_lens[it_idx]].squeeze(-1).numpy()
            raw_f0_filtered = raw_f0[raw_f0 != 0]
            raw_f0_filtered_int = raw_f0_filtered // 100
            f0s_cnt[raw_f0_filtered_int] += 1

            # raw_f0_filtered_int_log = np.log(raw_f0_filtered_int)
            print(np.max(raw_f0_filtered_int_log), np)
            f0s.append(raw_f0_filtered)

    print(f0s_cnt)
    
    # plt.show np.log(f0)
    # print(len(f0s))


def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    # HubertEncoder.add_arguments(parser)
    args = parser.parse_args()
    # args.lang_dir = Path(args.lang_dir)
    # args.exp_dir = Path(args.exp_dir)

    # world_size = args.world_size
    # assert world_size >= 1
    # if world_size > 1:
    #     mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    # else:
    run(rank=0, world_size=1, args=args)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
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
        default=12354,
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
            # "env_info": get_env_info(),
        }
    )

    return params

if __name__ == "__main__":
    main()