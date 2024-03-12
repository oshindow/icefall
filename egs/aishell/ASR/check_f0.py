import matplotlib.pyplot as plt
import librosa
import torch
y, sr = librosa.load('BAC009S0219W0272.wav', sr=16000)
y = torch.FloatTensor(y).unsqueeze(0)
import torch
import os

# rootdir = '/data2/xintong/aishell/data_aishell/wav/train'
# all_files = {}
# for root, dirs, files in os.walk(rootdir):
#     for file in files:
#         if '.npz' in file or 'csv' in file:
#             filepath = os.path.join(root, file)
#             os.remove(filepath)
            
def spectra(y, n_fft, hop_size, win_size):
    # pad_mode='constant', +1e-5
    #y:[1, T] or [batch_size, T]
	
    hann_window = torch.hann_window(win_size).to(y.device)
    # y = torch.nn.functional.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # y = y.squeeze(0)
    # stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
    #                   center=False, pad_mode='reflect', normalized=False, onesided=True)
    stft_spec=torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,center=True, pad_mode='constant')

    rea=stft_spec[:,:,:,0] #[batch_size, n_fft//2+1, frames]
    imag=stft_spec[:,:,:,1] #[batch_size, n_fft//2+1, frames]
    
    log_amplitude=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]
    phase=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec

log_amplitude, phase, stft_spec = spectra(y, 1024, 160, 960)

import numpy as np
f0_pesto = np.load('BAC009S0219W0272.pesto.pitch.npy')
f0_pw = np.load('BAC009S0219W0272.pw.pitch.npy')
print(log_amplitude.shape) 
plt.imshow(log_amplitude[0].numpy())
# print(len(f0_pesto))
# f0_scale = min([f for f in f0 if f != 0])

# print(f0.shape)
plt.plot(f0_pesto ,'r')
plt.plot(f0_pw, 'green')
plt.ylim(min(f0_pesto + f0_pw), max(f0_pesto + f0_pw))
plt.tight_layout()
plt.savefig('BAC009S0219W0272.png')
# # You can also predict pitches from audio files directly
# import os

# rootdir = '/data2/xintong/aishell/data_aishell/wav/train'
# all_files = {}
# for root, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file[:-4] not in all_files:
#             all_files[file[:-4]] = {}
#         elif '.wav' in file:
#             all_files[file[:-4]]['wav'] = file

#         elif '.csv' in file:
#             all_files[file[:-4]]['csv'] = file
        
#         elif 'npz' in file:
#             all_files[file[:-4]]['npz'] = file

# cnt = 0
# for key, value in all_files.items():
#     if len(value) != 3:
#         print(key)
#         cnt += 1
# print(cnt)
