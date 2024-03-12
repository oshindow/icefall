#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/xintong/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xintong/miniconda3/envs/k2/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

set -eou pipefail

nj=15
stage=30
stop_stage=30
perturb_speed=true

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell
#      You can find data_aishell, resource_aishell inside it.
#      You can download them from https://www.openslr.org/33
#
#  - $dl_dir/lm
#      This directory contains the language model downloaded from
#        https://huggingface.co/pkufool/aishell_lm
#
#        - 3-gram.unpruned.arpa
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=/data2/xintong

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bbpe_xxx,
# data/lang_bbpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  # 2000
  # 1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell $dl_dir/aishell
  #
  # The directory structure is
  # aishell/
  # |-- data_aishell
  # |   |-- transcript
  # |   `-- wav
  # `-- resource_aishell
  #     |-- lexicon.txt
  #     `-- speaker.info

  if [ ! -d $dl_dir/aishell/data_aishell/wav/train ]; then
    lhotse download aishell $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aishell manifest"
  # We assume that you have downloaded the aishell corpus
  # to $dl_dir/aishell
  if [ ! -f data/manifests/.aishell_manifests.done ]; then
    mkdir -p data/manifests
    lhotse prepare aishell $dl_dir/aishell data/manifests
    touch data/manifests/.aishell_manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for aishell"
  if [ ! -f data/fbank/.aishell.done ]; then
    mkdir -p data/fbank
    python3 ./local/compute_fbank_aishell.py --perturb-speed ${perturb_speed}
    touch data/fbank/.aishell.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

lang_phone_dir=data/lang_phone_merge
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  mkdir -p $lang_phone_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/aishell/resource_aishell/lexicon_merge.txt |
    sort | uniq > $lang_phone_dir/lexicon.txt

  ./local/generate_unique_lexicon.py --lang-dir $lang_phone_dir

  if [ ! -f $lang_phone_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_phone_dir
  fi


  # Train a bigram P for MMI training
  if [ ! -f $lang_phone_dir/transcript_words.txt ]; then
    log "Generate data to train phone based bigram P"
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_train_uid=$dl_dir/aishell/data_aishell/transcript/aishell_train_uid
    find $dl_dir/aishell/data_aishell/wav/train -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_train_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_train_uid $aishell_text |
	    cut -d " " -f 2- > $lang_phone_dir/transcript_words.txt
  fi

  if [ ! -f $lang_phone_dir/transcript_tokens.txt ]; then
    ./local/convert_transcript_words_to_tokens.py \
      --lexicon $lang_phone_dir/uniq_lexicon.txt \
      --transcript $lang_phone_dir/transcript_words.txt \
      --oov "<UNK>" \
      > $lang_phone_dir/transcript_tokens.txt
  fi

  if [ ! -f $lang_phone_dir/P.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 2 \
      -text $lang_phone_dir/transcript_tokens.txt \
      -lm $lang_phone_dir/P.arpa
  fi

  if [ ! -f $lang_phone_dir/P.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$lang_phone_dir/tokens.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      $lang_phone_dir/P.arpa > $lang_phone_dir/P.fst.txt
  fi
fi

lang_char_dir=data/lang_char
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare char based lang"
  mkdir -p $lang_char_dir
  # We reuse words.txt from phone based lexicon
  # so that the two can share G.pt later.

  # The transcripts in training set, generated in stage 5
  cp $lang_phone_dir/transcript_words.txt $lang_char_dir/transcript_words.txt

  cat $dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt |
  cut -d " " -f 2- > $lang_char_dir/text

  (echo '<eps> 0'; echo '!SIL 1'; echo '<SPOKEN_NOISE> 2'; echo '<UNK> 3';) \
    > $lang_char_dir/words.txt

  cat $lang_char_dir/text | sed 's/ /\n/g' | sort -u | sed '/^$/d' \
     | awk '{print $1" "NR+3}' >> $lang_char_dir/words.txt

  num_lines=$(< $lang_char_dir/words.txt wc -l)
  (echo "#0 $num_lines"; echo "<s> $(($num_lines + 1))"; echo "</s> $(($num_lines + 2))";) \
    >> $lang_char_dir/words.txt

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py --lang-dir $lang_char_dir
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare Byte BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bbpe_${vocab_size}
    mkdir -p $lang_dir

    cp $lang_char_dir/words.txt $lang_dir
    cp $lang_char_dir/text $lang_dir

    if [ ! -f $lang_dir/bbpe.model ]; then
      ./local/train_bbpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/text
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bbpe.py --lang-dir $lang_dir
    fi
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"

  mkdir -p data/lm

  # Train LM on transcripts
  if [ ! -f data/lm/3-gram.unpruned.arpa ]; then
    python3 ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lang_char_dir/transcript_words.txt \
      -lm data/lm/3-gram.unpruned.arpa
  fi

  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm
  if [ ! -f data/lm/G_3_gram_char.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="$lang_phone_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/lm/3-gram.unpruned.arpa > data/lm/G_3_gram_phone.fst.txt

    python3 -m kaldilm \
      --read-symbol-table="$lang_char_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/lm/3-gram.unpruned.arpa > data/lm/G_3_gram_char.fst.txt
  fi

  if [ ! -f $lang_char_dir/HLG.fst ]; then
    lang_phone_dir=data/lang_phone
    ./local/prepare_lang_fst.py  \
      --lang-dir $lang_phone_dir \
      --ngram-G ./data/lm/G_3_gram.fst.txt
  fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile LG & HLG"
  ./local/compile_hlg.py --lang-dir $lang_phone_dir --lm G_3_gram_phone
  ./local/compile_hlg.py --lang-dir $lang_char_dir --lm G_3_gram_char
  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bbpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir --lm G_3_gram_char
  done

  ./local/compile_lg.py --lang-dir $lang_phone_dir --lm G_3_gram_phone
  ./local/compile_lg.py --lang-dir $lang_char_dir --lm G_3_gram_char
  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bbpe_${vocab_size}
    ./local/compile_lg.py --lang-dir $lang_dir --lm G_3_gram_char
  done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Generate LM training data"

  log "Processing char based data"
  out_dir=data/lm_training_char
  mkdir -p $out_dir $dl_dir/lm

  if [ ! -f $dl_dir/lm/aishell-train-word.txt ]; then
    cp $lang_phone_dir/transcript_words.txt $dl_dir/lm/aishell-train-word.txt
  fi

  # training words
  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-train-word.txt \
    --lm-archive $out_dir/lm_data.pt

  # valid words
  if [ ! -f $dl_dir/lm/aishell-valid-word.txt ]; then
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_valid_uid=$dl_dir/aishell/data_aishell/transcript/aishell_valid_uid
    find $dl_dir/aishell/data_aishell/wav/dev -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_valid_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_valid_uid $aishell_text |
	    cut -d " " -f 2- > $dl_dir/lm/aishell-valid-word.txt
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-valid-word.txt \
    --lm-archive $out_dir/lm_data_valid.pt

  # test words
  if [ ! -f $dl_dir/lm/aishell-test-word.txt ]; then
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_test_uid=$dl_dir/aishell/data_aishell/transcript/aishell_test_uid
    find $dl_dir/aishell/data_aishell/wav/test -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_test_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_test_uid $aishell_text |
	    cut -d " " -f 2- > $dl_dir/lm/aishell-test-word.txt
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-test-word.txt \
    --lm-archive $out_dir/lm_data_test.pt
fi


if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Stage 11: Sort LM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of tokens
  # in a sentence.

  out_dir=data/lm_training_char
  mkdir -p $out_dir
  ln -snf ../../../librispeech/ASR/local/sort_lm_training_data.py local/

  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data.pt \
    --out-lm-data $out_dir/sorted_lm_data.pt \
    --out-statistics $out_dir/statistics.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data_valid.pt \
    --out-lm-data $out_dir/sorted_lm_data-valid.pt \
    --out-statistics $out_dir/statistics-valid.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data_test.pt \
    --out-lm-data $out_dir/sorted_lm_data-test.pt \
    --out-statistics $out_dir/statistics-test.txt
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  log "Stage 11: Train RNN LM model"
  python ../../../icefall/rnn_lm/train.py \
    --start-epoch 0 \
    --world-size 1 \
    --num-epochs 20 \
    --use-fp16 0 \
    --embedding-dim 512 \
    --hidden-dim 512 \
    --num-layers 2 \
    --batch-size 400 \
    --exp-dir rnnlm_char/exp \
    --lm-data $out_dir/sorted_lm_data.pt \
    --lm-data-valid $out_dir/sorted_lm_data-valid.pt \
    --vocab-size 4336 \
    --master-port 12345
fi

# whisper large-v3 using 128 mel bins, others using 80 mel bins
whisper_mel_bins=80
output_dir=data/fbank_whisper
if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
  log "Stage 30: Compute ${whisper_mel_bins} dim fbank for whisper model fine-tuning"
  # if [ ! -f $output_dir/.aishell.whisper.done ]; then
    mkdir -p $output_dir
    # ./local/compute_fbank_aishell.py --perturb-speed ${perturb_speed} --num-mel-bins ${whisper_mel_bins} --whisper-fbank true --output-dir $output_dir
    ./local/compute_fbank_latic.py --num-mel-bins ${whisper_mel_bins} --whisper-fbank true --output-dir $output_dir
    ./local/compute_fbank_musan.py --num-mel-bins ${whisper_mel_bins} --whisper-fbank true --output-dir $output_dir
    # touch $output_dir/.aishell.whisper.done
  # fi
fi