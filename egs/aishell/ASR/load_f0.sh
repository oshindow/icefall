hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-base-fairseq-ckpt.pt
# hubert_model_dir=/home/xintong/macroyang1998/icefall/egs/aishell/ASR/finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt
# hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0,1,2"
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

python load_f0.py \
    --world-size 3 \
    --num-epochs 20 \
    --start-epoch 1 \
    --exp-dir /data2/xintong/icefall/aihsell/finetune_hubert_transducer/exp_aishell_phone_model_f0_conv_att_10ms \
    --max-duration 100 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --lang-dir data/lang_phone_merge \
    --input-strategy AudioSamples \
    --enable-spec-aug 0 \
    --spec-aug-time-warp-factor -1 \
    --enable-musan 0 \
    --context-size 2