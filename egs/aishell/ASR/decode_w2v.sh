export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
# export CUDNN_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/lib
# export CUDNN_INCLUDE_PATH=/home/xintong/miniconda3/envs/k2/lib/python3.8/site-packages/nvidia/cudnn/include
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:120
export CUDA_VISIBLE_DEVICES="0"
# hubert_model_dir=/home/xintong/icefall/egs/aishell/ASR/chinese-hubert-base-fairseq-ckpt.pt
# hubert_model_dir=/home/xintong/macroyang1998/icefall/egs/aishell/ASR/finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt
# hubert_model_dir=/data2/xintong/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt
hubert_model_dir=/data2/xintong/pretrained_models/chinese-wav2vec2-base-fairseq-ckpt.pt

for epoch in 7 15 19; do
    for avg in 2 4 6 8; do

        python3 finetune_hubert_transducer/decode_phone_w2v_ctc.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir /data2/xintong/icefall/aihsell/finetune_hubert_transducer/train_phone_model_w2v_ctc_sos_eos \
            --lang-dir data/lang_phone_merge \
            --max-duration 50 \
            --decoding-method greedy_search \
            --beam-size 4 \
            --hubert-model-dir $hubert_model_dir \
            --encoder-dim 1024 \
            --hubert-subsample-output 1 \
            --hubert-subsample-mode concat_tanh \
            --use-averaged-model 1 \
            --input-strategy AudioSamples 
    done
done