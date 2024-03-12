import os
import sys
sys.path.insert(0, '/home/xintong/lhotse')
# os.environ['PYTHONPATH'] = '/home/xintong/lhotse'
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy, Features
import numpy as np

part = 'train' 
# cuts = load_manifest_lazy(
#             "/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_" + part + ".jsonl.gz"
#         )
f0_cuts = load_manifest_lazy("data/fbank/latic_cuts_f0_40ms_test.jsonl.gz")
# f0_cuts = load_manifest_lazy("data/fbank/aishell_cuts_f0_40ms_train.jsonl.gz")
new_f0_cuts = CutSet.from_cuts(f0_cuts)
part = 'train' 
# cuts = load_manifest_lazy(
#             "/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_" + part + ".jsonl.gz"
#         )
fbank_cuts = load_manifest_lazy("data/fbank/latic_cuts_test.jsonl.gz")
# fbank_cuts = load_manifest_lazy("data/fbank/aishell_cuts_train.jsonl.gz")
new_fbank_cuts = CutSet.from_cuts(fbank_cuts)

cnt = 0
for key, value in new_fbank_cuts.cuts.items():
    # new_fbank_cuts[key].custom = {}
    new_fbank_cuts[key].custom = new_f0_cuts[key].features
    if cnt and cnt % 1000 == 0:
        print(cnt)
    cnt += 1
new_fbank_cuts.to_file('/home/xintong/icefall/egs/aishell/ASR/data/fbank/latic_cuts_test_f0.jsonl.gz')
# new_fbank_cuts.to_file('/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_train_f0.jsonl.gz')