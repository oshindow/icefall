export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include

export CUDA_VISIBLE_DEVICES="0"

# ./pruned_transducer_stateless7/decode.py \
#     --epoch 28 \
#     --avg 15 \
#     --exp-dir ./pruned_transducer_stateless7/exp \
#     --max-duration 600 \
#     --decoding-method fast_beam_search \
#     --beam 4 \
#     --max-contexts 4 \
#     --max-states 8

# for m in greedy_search modified_beam_search fast_beam_search ; do
#   ./zipformer/decode.py \
#     --epoch 55 \
#     --avg 17 \
#     --exp-dir ./zipformer/exp \
#     --lang-dir data/lang_char \
#     --context-size 1 \
#     --decoding-method $m
# done

# for m in greedy_search modified_beam_search fast_beam_search ; do
#   ./zipformer/decode_phone.py \
#     --epoch 55 \
#     --avg 17 \
#     --exp-dir ./zipformer/exp_phone \
#     --lang-dir data/lang_phone \
#     --context-size 1 \
#     --decoding-method $m
# done

for m in greedy_search; do
  ./zipformer/decode_phone_rnn.py \
    --epoch 55 \
    --avg 17 \
    --exp-dir /data2/xintong/icefall/aihsell/zipformer/exp_phone_merge_rnn \
    --lang-dir data/lang_phone_merge \
    --decoding-method $m 
    # --context-size 1 \
done
