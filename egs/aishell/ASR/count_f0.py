import os
import numpy as np
dataset = 'aishell-1'
rootdir = '/data2/xintong/aishell/data_aishell/wav/dev'

f0_floor = 60
f0_ceil = 1400
frame_periods = [5,20,40]
f0_mel_min = 1127 * np.log(1 + f0_floor / 700)
f0_mel_max = 1127 * np.log(1 + f0_ceil / 700)
# f0_bins = [128,256,512]
f0_bins = [256]
file_list = [[],[],[],[],[],[],[],[]]
# print(file_list)
cnt = 0 
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if '.wav' in file:
            basename = file[:-4]
            if basename + '.pw.f0.npy' in files and basename + '.pw.f0.20ms.npy' in files and basename + '.pw.f0.40ms.npy' in files and basename + '.pw.f0.5ms.npy' in files \
            and basename + '.pw.pit.npy' in files and basename + '.pw.pit.20ms.256bin.npy' in files and basename + '.pw.pit.40ms.256bin.npy' in files and basename + '.pw.pit.5ms.256bin.npy' in files \
            and basename + '.pw.uv.npy' in files and basename + '.pw.uv.20ms.npy' in files and basename + '.pw.uv.40ms.npy' in files and basename + '.pw.uv.5ms.npy' in files:
                cnt += 1
            else:
                print(os.path.join(root, file))
print(cnt) 
# local: train 120403, test 7176, dev 14329
# stand: train 120098, test 7176, dev 14326