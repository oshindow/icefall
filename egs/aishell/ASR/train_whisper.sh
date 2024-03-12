hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-base-fairseq-ckpt.pt
# hubert_model_dir=/home/xintong/macroyang1998/icefall/egs/aishell/ASR/finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt
# hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
# export CPATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cuda_runtime/include:$CPATH
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0,1,2,3"
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

torchrun --nproc_per_node 4 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir /data2/xintong/whisper/exp_large_v2 \
  --model-name large-v2 \
  --manifest-dir data/fbank_whisper \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json \
  --enable-musan 0