import torch

bsz = 3
length = 5
dim = 20
threshold = 1

hidden = torch.ones((bsz, length, dim))
alphas = torch.rand((bsz, length)) # 0~1

print(alphas)
# print(hidden)

def cif(hidden, alphas, threshold: float):
	batch_size, len_time, hidden_size = hidden.size()
	threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device) # 1.0
	
	# loop varss
	# integrate: [0,0]
	# frame: [[0,0,0,...],[0,0,0,...]]
	integrate = torch.zeros([batch_size], dtype=alphas.dtype, device=hidden.device) 
	frame = torch.zeros([batch_size, hidden_size], dtype=hidden.dtype, device=hidden.device) 
	# intermediate vars along time
	list_fires = []
	list_frames = []
	
	# print(alphas.shape)
	for t in range(len_time):
		alpha = alphas[:, t] # alphas shape: (bsz, time)  torch.Size([6, 821])
		distribution_completion = torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device) - integrate
		# integrate is summarion of alpha
		# distribution_completion is delta: 1 - integrate
		integrate += alpha
		list_fires.append(integrate)
		
		# if integrate >= threshold, fire_place = True
		# else fire_place = False
		fire_place = integrate >= threshold

		# if fire_place = True, integrate -= 1 (1.5-1=0.5)
		# else return integrate
		integrate = torch.where(fire_place,
		                        integrate - torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device),
		                        integrate)
		# if fire_place = True, cur = dis
		# else cur = alpha cur is left part when fire
		cur = torch.where(fire_place,
		                  distribution_completion,
		                  alpha)
		
		# if fire_place = True, remainds = alpha - 1
		# else remainds = 0 
		remainds = alpha - cur
		
		frame += cur[:, None] * hidden[:, t, :]
		list_frames.append(frame)
		frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
		                    remainds[:, None] * hidden[:, t, :],
		                    frame)
	
	fires = torch.stack(list_fires, 1) # integrate - 1
	frames = torch.stack(list_frames, 1)
	# frames: (b, t, dim) accumulate frame
	fire_idxs = fires >= threshold
	frame_fires = torch.zeros_like(hidden)
	print(fire_idxs[0])
	print(frames[0,fire_idxs[0]])
	max_label_len = frames[0, fire_idxs[0]].size(0)
	for b in range(batch_size):
		frame_fire = frames[b, fire_idxs[b]]
		frame_len = frame_fire.size(0) # number of True in fire_idx[b]
		frame_fires[b, :frame_len, :] = frame_fire
	
		if frame_len >= max_label_len: # update max_label_len
			max_label_len = frame_len
	frame_fires = frame_fires[:, :max_label_len, :]
	return frame_fires, fires

frame_fires, fires = cif(hidden, alphas, threshold)