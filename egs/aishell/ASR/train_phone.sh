export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0,1,2"

# E1
# ./pruned_transducer_stateless7/train.py \
#   --world-size 2 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --exp-dir pruned_transducer_stateless7/exp \
#   --max-duration 300

# E2
python ./zipformer/train_phone.py \
  --world-size 3 \
  --num-epochs 60 \
  --start-epoch 1 \
  --use-fp16 1 \
  --context-size 2 \
  --enable-musan 0 \
  --exp-dir /data2/xintong/icefall/aihsell/zipformer/exp_phone_merge_context2 \
  --lang-dir data/lang_phone_merge \
  --max-duration 500 \
  --enable-musan 0 \
  --base-lr 0.045 \
  --lr-batches 7500 \
  --lr-epochs 18 \
  --spec-aug-time-warp-factor 20

zipformer phone asr
  ./zipformer/train_phone.py \
  --world-size 3 \
  --num-epochs 60 \
  --start-epoch 1 \
  --use-fp16 1 \
  --context-size 2 \
  --enable-musan 0 \
  --exp-dir zipformer/exp_phone \
  --max-duration 500 \
  --enable-musan 0 \
  --base-lr 0.045 \
  --lr-batches 7500 \
  --lr-epochs 18 \
  --spec-aug-time-warp-factor 20