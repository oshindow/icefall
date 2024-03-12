export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0"
./finetune_hubert_transducer/pretrained_int.py \
    --checkpoint ./finetune_hubert_transducer/exp_aishell_phone_model/epoch-20.pt \
    --bpe-model ./data/lang_phone \
    --method greedy_search \
    --text "d a2 * er3 * un2 x iao4 * uan2 l e5 t a1 d e5 * u4 l ei4 * iu2 l ai2 d i4 l iu4 b an3 d e5 g ao3 z iy5" \
    /home/xintong/icefall/egs/aishell/ASR/000100002.WAV