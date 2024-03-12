
# original phoneme sequence
# test phoneme sequence 

# test phoneme sequence
# recog phoneme sequence
def prepare_phone_seqs(canoncial_manual='data/recogs-latic-canoncial-manual.txt', manual_recog='finetune_hubert_transducer/exp_aishell_phone_model_pitjoiner_80_merge/greedy_search/recogs-test-greedy_search-epoch-18-avg-7-context-2-max-sym-per-frame-1-use-averaged-model.txt'):
    
    canoncial_manual_dict = {}
    with open(canoncial_manual, 'r', encoding='utf8') as input:
        for line in input:
            utt = line.strip().split(':\t')[0]
            # print(utt)
            if utt not in canoncial_manual_dict:
                canoncial_manual_dict[utt] = {}
            phone_seq_str = line.strip().split(':\t')[1]
            # print(phone_seq_str)
            if 'ref' in phone_seq_str:
                phone_seq = phone_seq_str.split("=['")[1].strip("']").split("', '")
                # print(phone_seq)
                canoncial_manual_dict[utt]['canoncial'] = phone_seq
            if 'hyp' in phone_seq_str:
                phone_seq = phone_seq_str.split("=['")[1].strip("']").split("', '")
                canoncial_manual_dict[utt]['manual'] = phone_seq

    # manual_recog_dict = {}
    with open(manual_recog, 'r', encoding='utf8') as input:
        for line in input:
            utt = line.strip().split(': ')[0].split('-')[0]
            # if utt not in manual_recog_dict:
            #     manual_recog_dict[utt] = {}
            phone_seq_str = line.strip().split(':\t')[1]
            # if 'ref' in phone_seq_str:
            #     phone_seq = phone_seq_str.split("=['").strip("']")[1].split("', '")
            #     manual_recog_dict[utt]['manual'] = phone_seq
            if 'hyp' in phone_seq_str and utt in canoncial_manual_dict:
                try:
                    phone_seq = phone_seq_str.split("=['")[1].strip("']").split("', '")
                except Exception as e:
                    print(phone_seq_str)
                canoncial_manual_dict[utt].update({'recog': phone_seq})

    return canoncial_manual_dict

canoncial_manual = 'data/recogs-latic-canoncial-manual.txt'
mode = [0 for x in range(15)]
manual_recog = [0 for x in range(15)]
mode[0] = 'zipformer'
manual_recog[0] = 'zipformer/exp_phone_merge/exp_phone_merge/greedy_search/recogs-test-greedy_search_blank_penalty_0.0-epoch-55-avg-17-context-1-max-sym-per-frame-1-blank-penalty-0.0-use-averaged-model.txt'
mode[1] = 'pretrained hubert'
manual_recog[1] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_rmf0_hubertenc/greedy_search/recogs-test-greedy_search-epoch-19-avg-8-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[1] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_rmf0_hubertenc/greedy_search/recogs-test-greedy_search-epoch-15-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[2] = 'finetuned hubert'
manual_recog[2] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_rmf0/greedy_search/recogs-test-greedy_search-epoch-6-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[2] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_rmf0/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[3] = 'raw f0 linear' # replace
# manual_recog[3] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_conv/greedy_search/recogs-test-greedy_search-epoch-6-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[3] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_conv_att_10ms/greedy_search/recogs-test-greedy_search-epoch-19-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[12] = 'raw f0 emb w/o fft_enc'
manual_recog[12] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'

mode[4] = 'raw f0 emb w/ fft_enc'
# manual_recog[4] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv/greedy_search/recogs-test-greedy_search-epoch-6-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
# manual_recog[4] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv/greedy_search/recogs-test-greedy_search-epoch-19-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[4] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[13] = 'raw f0 emb fft_enc 20ms'
manual_recog[13] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_20ms/greedy_search/recogs-test-greedy_search-epoch-19-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[14] = 'raw f0 emb fft_enc 40ms'
manual_recog[14] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'

mode[5] = 'mel-scaled f0 emb'
manual_recog[5] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_mel/greedy_search/recogs-test-greedy_search-epoch-15-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[5] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_mel/greedy_search/recogs-test-greedy_search-epoch-15-avg-2-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[6] = 'coarse'
manual_recog[6] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_mel_coarse/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[6] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_mel_coarse/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[7] = 'att'
manual_recog[7] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[8] = 'FFT'
manual_recog[8] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_FFT/greedy_search/recogs-test-greedy_search-epoch-19-avg-14-context-2-max-sym-per-frame-1-use-averaged-model.txt'
# manual_recog[8] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms/greedy_search/recogs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[9] = 'convFFT'
manual_recog[9] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_convFFT/greedy_search/recogs-test-greedy_search-epoch-7-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[10] = 'wav2vec'
manual_recog[10] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_convFFT_w2v/greedy_search/recogs-test-greedy_search-epoch-19-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
manual_recog[10] = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_convFFT_w2v/greedy_search/recogs-test-greedy_search-epoch-7-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
mode[11] = 'baseline'
manual_recog[11] = 'finetune_hubert_transducer/train_phone_model_w2v_ctc_sos_eos/greedy_search/recogs-test-1best-epoch-7-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
import os
import sys
for m_r in range(len(manual_recog)):
    
    canoncial_manual_dict = prepare_phone_seqs(canoncial_manual=canoncial_manual, manual_recog=manual_recog[m_r])

    import kaldialign
    cnt = 0
    metrics = {'false_rejection': 0, 'true_acceptance': 0, 'true_rejection': 0,'false_acceptance': 0,'correct_diagnosis':0,'diagnostic_errors':0}
        
    for cut_id, value in canoncial_manual_dict.items():
        ali_1 = kaldialign.align(value['canoncial'], value['manual'], "*", sclite_mode=False)
        ali_2 = kaldialign.align(value['canoncial'], value['recog'], "*", sclite_mode=False)

        if len(ali_1) == len(ali_2):
            cnt += 1
        else:
            continue
        
        for phone_idx in range(len(ali_1)):
            if ali_1[phone_idx][0] == ali_1[phone_idx][1]: # actual correct
                if ali_2[phone_idx][0] == ali_2[phone_idx][1]: # detect as correct
                    metrics['true_acceptance'] += 1
                else:
                    metrics['false_rejection'] += 1
            else: 
                if ali_2[phone_idx][0] == ali_2[phone_idx][1]: # detect as correct
                    metrics['false_acceptance'] += 1
                elif ali_1[phone_idx][1] == ali_2[phone_idx][1]: # detect as incorrect and same as manual
                    metrics['true_rejection'] += 1
                    metrics['correct_diagnosis'] += 1
                elif ali_1[phone_idx][1] != ali_2[phone_idx][1]: # detect as incorrect but not same as manual
                    metrics['true_rejection'] += 1
                    metrics['diagnostic_errors'] += 1
                    # TR represents the number of phones labeled as mispronunciations and detected as incorrect.
                    # FA is the number of phones that are mispronounced but misclassified as correct.
    print(mode[m_r], cnt, len(canoncial_manual_dict), metrics)
    false_rejection_rate = metrics['false_rejection'] / (metrics['true_acceptance'] + metrics['false_rejection'])
    false_acceptance_rate = metrics['false_acceptance'] / (metrics['false_acceptance'] + metrics['true_rejection'])
    precision = metrics['true_rejection'] / (metrics['true_rejection'] + metrics['false_rejection']) 
    # precision = TN/(FN + TN))
    recall = metrics['true_rejection'] / (metrics['true_rejection'] + metrics['false_acceptance'])
    
    # precision = metrics['true_acceptance'] / (metrics['true_acceptance'] + metrics['false_acceptance'])
    # precision = TA/(FA + TA)
    # recall = metrics['true_acceptance'] / (metrics['true_acceptance'] + metrics['false_rejection'])
    F1 = 2 * (precision * recall) / (precision + recall)
    print('false_rejection_rate:',false_rejection_rate, 'false_acceptance_rate:',false_acceptance_rate,'precision:',precision,'recall:',recall,'F1:',F1)
    per = manual_recog[m_r].replace('recogs', 'errs')
    print(per)
    os.system('head -1 ' + per)
# False Rejection Rate (FRR; FR/(FR + TA)), 
# False Acceptance Rate (FAR; FA/(FA + TR)), 
# Recall (RE; TA/(FR + TA)), Precision (PR; TA/(FA + TA)), 
# and F1-score (2*(RE * PR)/(RE + PR))