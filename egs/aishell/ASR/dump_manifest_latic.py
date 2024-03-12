import os
import sys
sys.path.insert(0, '/home/xintong/lhotse')
# os.environ['PYTHONPATH'] = '/home/xintong/lhotse'
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy, Features
import numpy as np
import torch

part = 'train' 
# cuts = load_manifest_lazy(
#             "/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_" + part + ".jsonl.gz"
#         )
cuts = load_manifest_lazy("data/fbank/latic_cuts_test_f0.jsonl.gz")
new_cuts = CutSet.from_cuts(cuts)
for key, value in new_cuts.cuts.items():
    features = Features(
        type=value.custom['type'],
        num_frames=value.custom['num_frames'],
        num_features=value.custom['num_features'],
        frame_shift=value.custom['frame_shift'],
        sampling_rate=value.custom['sampling_rate'],
        start=value.custom['start'],
        duration=value.custom['duration'],
        storage_type=value.custom['storage_type'],
        storage_path=value.custom['storage_path'],
        storage_key=value.custom['storage_key']
        )
    feats = features.load(start=features.start, duration=features.duration)
    if feats.shape[0] - features.num_frames == 1:
        feats = feats[: features.num_frames, :]
    elif feats.shape[0] - features.num_frames == -1:
        feats = np.concatenate((feats, feats[-1:, :]), axis=0)
    
first_manifest = getattr(new_cuts[0], 'f0')
def _read_features(cut):
    return torch.from_numpy(cut.load_features())
features  = _read_features(new_cuts[0])
new_cuts[0].load()
root = '/data2/xintong/aishell/data_aishell/wav/' + part
# root = '/data2/xintong/LATIC/WAVE'
cnt = 0
for key, value in new_cuts.cuts.items():
    spk = key.split('-')[0]
    spk_num = 'SPEAKER00' + str(int(key[:5]))
    if cnt and cnt % 1000 == 0:
        print(cnt)
    try:
        pitch = np.load(os.path.join(root, spk_num, 'SESSION0', spk + '.pw.pit.npy'))
        f0 = np.load(os.path.join(root, spk_num, 'SESSION0', spk + '.pw.f0.npy'))
        uv = np.load(os.path.join(root, spk_num, 'SESSION0', spk + '.pw.uv.npy'))
    except Exception as e:
        print(e)
        # print()
        continue
    new_cuts.cuts[key].custom = {'pitch': pitch.tolist(),'uv': uv.tolist(), 'f0': f0.tolist()}
    cnt += 1
print('.')
new_cuts.to_file('/home/xintong/icefall/egs/aishell/ASR/data/fbank/latic_cuts_test_new.jsonl.gz')