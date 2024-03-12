hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-base-fairseq-ckpt.pt
# hubert_model_dir=/home/xintong/macroyang1998/icefall/egs/aishell/ASR/finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt
# hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0,1"
# /zipformer/train_phone.py \
#   --world-size 2 \
#   --num-epochs 60 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --context-size 1 \
#   --enable-musan 0 \
#   --exp-dir zipformer/exp_phone \
#   --max-duration 500 \
#   --enable-musan 0 \
#   --base-lr 0.045 \
#   --lr-batches 7500 \
#   --lr-epochs 18 \
#   --spec-aug-time-warp-factor 20

python finetune_hubert_transducer/train_phone_model_pitenc_merge.py \
    --world-size 2 \
    --num-epochs 20 \
    --exp-dir /data2/xintong/icefall/aihsell/finetune_hubert_transducer/exp_aishell_phone_model_pitenc_merge \
    --max-duration 50 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --lang-dir data/lang_phone_merge \
    --input-strategy AudioSamples \
    --hubert-model-dir $hubert_model_dir \
    --hubert-freeze-finetune-updates 10000 \
    --hubert-mask-channel-length 64 \
    --hubert-mask-prob 0.25 \
    --hubert-mask-channel-prob 0.5 \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    --encoder-dim 1024 \
    --enable-spec-aug 0 \
    --spec-aug-time-warp-factor -1 \
    --base-lr 5e-7 \
    --TSA-init-lr 5e-7 \
    --TSA-warmup-lr 3e-5 \
    --TSA-end-lr 1.5e-6 \
    --TSA-total-steps 320000 \
    --enable-musan 0