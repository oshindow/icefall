export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

nj=15
stage=0
stop_stage=5

# log() {
#   # This function is from espnet
#   local fname=${BASH_SOURCE[1]##*/}
#   echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
# }

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
#   log "Stage 2: Prepare latic manifest data/fbank"
  python3 ./local/compute_f0_aishell.py
fi