import torch
import torch.nn as nn

class CifPredictorV2(torch.nn.Module):
	def __init__(self, idim=1024,
                 l_order=1,
                 r_order=1,
                 threshold=1.0,
                 dropout=0.1,
                 smooth_factor=1.0,
                 noise_threshold=0,
                 tail_threshold=0.0,
    			):
		super().__init__()
		
		# self.pad = nn.ConstantPad1d((l_order, r_order), 0)
		# self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
		self.cif_output = nn.Linear(idim, 1)
		self.threshold = threshold
		self.smooth_factor = smooth_factor
		self.noise_threshold = noise_threshold
		self.tail_threshold = tail_threshold
        
	def forward(self, hidden: torch.Tensor,
	            # mask: torch.Tensor,
	            ):
		# print(hidden.shape)
		alphas, token_num = self.forward_cnn(hidden)
		# mask = mask.transpose(-1, -2).float()
		# mask = mask.squeeze(-1)
		hidden, alphas, token_num = self.tail_process_fn(hidden, alphas)
		# try:
		acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
		# except Exception as e:
		# 	print(e)
		return acoustic_embeds, token_num, alphas, cif_peak
	
	def forward_cnn(self, hidden: torch.Tensor,
	            # mask: torch.Tensor,
	            ):
		h = hidden
		# context = h.transpose(1, 2)
		# queries = self.pad(context)
		# output = torch.relu(self.cif_conv1d(queries))
		# output = output.transpose(1, 2)
		
		output = self.cif_output(hidden)
		alphas = torch.sigmoid(output)
		alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
		# mask = mask.transpose(-1, -2).float()
		# alphas = alphas * mask
		alphas = alphas.squeeze(-1)
		token_num = alphas.sum(-1)

		return alphas, token_num
	
	def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
		b, t, d = hidden.size()
		tail_threshold = self.tail_threshold
		
		zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
		# ones_t = torch.ones_like(zeros_t)
		    
		# mask_1 = torch.cat([mask, zeros_t], dim=1)
		# mask_2 = torch.cat([ones_t, mask], dim=1)
		# mask = mask_2 - mask_1
		# tail_threshold = mask * tail_threshold
		alphas = torch.cat([alphas, zeros_t], dim=1)
		alphas = torch.add(alphas, tail_threshold)

		zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
		hidden = torch.cat([hidden, zeros], dim=1)
		token_num = alphas.sum(dim=-1)
		token_num_floor = torch.floor(token_num)
		
		return hidden, alphas, token_num_floor

def cif(hidden, alphas, threshold: float):
	batch_size, len_time, hidden_size = hidden.size()
	threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device) # 1.0
	
	# loop varss
	integrate = torch.zeros([batch_size], dtype=alphas.dtype, device=hidden.device)
	frame = torch.zeros([batch_size, hidden_size], dtype=hidden.dtype, device=hidden.device)
	# intermediate vars along time
	list_fires = []
	list_frames = []
	
	# print(alphas.shape)
	for t in range(len_time):
		alpha = alphas[:, t] # alphas shape: (bsz, time)  torch.Size([6, 821])
		distribution_completion = torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device) - integrate
		
		integrate += alpha
		list_fires.append(integrate)
		
		fire_place = integrate >= threshold
		integrate = torch.where(fire_place,
		                        integrate - torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device),
		                        integrate)
		cur = torch.where(fire_place,
		                  distribution_completion,
		                  alpha)
		remainds = alpha - cur
		
		frame += cur[:, None] * hidden[:, t, :]
		list_frames.append(frame)
		frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
		                    remainds[:, None] * hidden[:, t, :],
		                    frame)
	
	fires = torch.stack(list_fires, 1)
	frames = torch.stack(list_frames, 1)

	fire_idxs = fires >= threshold
	frame_fires = torch.zeros_like(hidden)
	max_label_len = frames[0, fire_idxs[0]].size(0)
	for b in range(batch_size):
		frame_fire = frames[b, fire_idxs[b]]
		frame_len = frame_fire.size(0)
		frame_fires[b, :frame_len, :] = frame_fire
	
		if frame_len >= max_label_len:
			max_label_len = frame_len
	frame_fires = frame_fires[:, :max_label_len, :]
	return frame_fires, fires

class mae_loss(nn.Module):

    def __init__(self, normalize_length=False):
        super(mae_loss, self).__init__()
        self.normalize_length = normalize_length
        self.criterion = torch.nn.L1Loss(reduction='sum')

    def forward(self, token_length, pre_token_length):
        loss_token_normalizer = token_length.size(0)
        if self.normalize_length:
            loss_token_normalizer = token_length.sum().type(torch.float32)
        loss = self.criterion(token_length, pre_token_length)
        loss = loss / loss_token_normalizer
        return loss