import os
import sys
sys.path.insert(0, '/home/xintong/lhotse')
# os.environ['PYTHONPATH'] = '/home/xintong/lhotse'
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
import numpy as np
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from collections import Counter

# part = 'dev' 
# param = ['5ms','256bin']

# root = '/data2/xintong/aishell/data_aishell/wav/' + part
# cuts = load_manifest_lazy(
#             "/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_" + part + ".jsonl.gz"
#         )
# new_cuts = CutSet.from_cuts(cuts)
cuts = load_manifest_lazy('data/fbank/aishell_cuts_f0_40ms_train.jsonl.gz')
# cuts = load_manifest_lazy('data/fbank/aishell_cuts_train.jsonl.gz')
import torch
# cut_list = []   
f0_range = [[0,0], [0,0], [0,0]]
# new_cuts = CutSet.from_cuts(cuts)
features = {}
sp1_1 = {}
sp0_9 = {}
sp0 = {}
for idx, cut in enumerate(cuts):
    if idx and idx % 1000 == 0:
        print(idx)
    
    key = cut.id
    # print(key)s
    if 'BAC009S0198W0441' in key:
        # print(key)
        features[key] = torch.from_numpy(cut.load_features())
        print(features[key].shape)
        # cut_list.append(cut)
    features[key] = torch.from_numpy(cut.load_features()).squeeze(-1)
    features[key] = torch.where(features[key] < 0, 0, features[key])
    features[key] = (features[key] // 100).long()
    # features[key] = features[key].numpy()
    # features[key] = 1127 * np.log(1 + features[key] / 700)
    # features[key] = features[key] // 100
    # counts = np.bincount(features[key])
    counts = Counter(features[key])
    counts = dict(counts)
    if '_sp1.1' in key:
        # sp1_1.update(counts)
        sp1_1 = {key: sp1_1.get(key, 0) + counts.get(key, 0) for key in set(sp1_1) | set(counts)}
        # print('.')
        # f0_range[2][0] = max(torch.max(features[key]).item(), f0_range[2][0])
        # f0_range[2][1] = min(torch.min(features[key]).item(), f0_range[2][1])
    elif '_sp0.9' in key:
        sp0_9 = {key: sp0_9.get(key, 0) + counts.get(key, 0) for key in set(sp0_9) | set(counts)}
        # print('.')
        # f0_range[0][0] = max(torch.max(features[key]).item(), f0_range[0][0])
        # f0_range[0][1] = min(torch.min(features[key]).item(), f0_range[0][1])
    else:
        sp0 = {key: sp0.get(key, 0) + counts.get(key, 0) for key in set(sp0) | set(counts)}
        # print('.')
        # f0_range[1][0] = max(torch.max(features[key]).item(), f0_range[1][0])
        # f0_range[1][1] = min(torch.min(features[key]).item(), f0_range[1][1])

    
    # min_pitch = 0
    # max_pitch = 1600
    # pitch_step = 1
    # pitch_bin_size = int((max_pitch - min_pitch) / pitch_step)
    # pitch_bins = torch.linspace(min_pitch, max_pitch, pitch_bin_size)
    # index = torch.bucketize(pitch, pitch_bins)
    
sp0_list = [count for count in sp0.values()]
sp0_9_list = [count for count in sp0_9.values()]
sp1_1_list = [count for count in sp1_1.values()]


# BAC009S0198W0441-0
# BAC009S0198W0441-0_sp1.1
# BAC009S0198W0441-0_sp0.9
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=13.5)
bar_width = 0.25
a = list(range(1, 16)) # f0
# a = list(range(1, 14)) 
x1 = [x + bar_width for x in a]
x2 = [x + bar_width for x in x1]
x3 = [x + bar_width for x in x2]

plt.bar(x1, sp0_list[1:], width=bar_width, label='sp1.0')
plt.bar(x2, sp1_1_list[1:], width=bar_width, label='sp1.1')
plt.bar(x3, sp0_9_list[1:], width=bar_width, label='sp0.9')

# for key, value in features.items():
#     value = value.squeeze(-1).numpy()
#     x = range(len(value))
#     plt.plot(x, value, label=key)
plt.xticks(list(range(1, 16)), a)
plt.legend()
plt.xlabel('frequency in mel-frequency')
plt.ylabel('count')
plt.savefig('mel-scale.png')

# cnt = 0
# for key, value in new_cuts.cuts.items():
#     prefix = key.split('-')[0]
#     spk = prefix[6:11]
#     if 'BAC009S0198W0441' in key:
#         print(key)
#     if cnt and cnt % 1000 == 0:
#         print(cnt)
#     try:
#         pitch = np.load(os.path.join(root, spk, prefix + '.pw.pit.' + str(param[0]) + '.' + str(param[1]) + '.npy'))
#         f0 = np.load(os.path.join(root, spk, prefix + '.pw.f0.' + str(param[0]) + '.npy'))
#         uv = np.load(os.path.join(root, spk, prefix + '.pw.uv.' + str(param[0]) + '.npy'))
#     except Exception as e:
#         print(e)
#         continue

#     new_cuts.cuts[key].custom = {'pitch': pitch.tolist(),'uv': uv.tolist(), 'f0': f0.tolist()}
#     cnt += 1

# new_cuts.to_file('/home/xintong/icefall/egs/aishell/ASR/data/fbank/aishell_cuts_' + part + '.' + str(param[0]) + '.' + str(param[1]) + '.jsonl.gz')