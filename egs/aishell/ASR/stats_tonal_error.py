# errorfile = 'zipformer/exp_phone/greedy_search/errs-test-greedy_search_blank_penalty_0.0-epoch-55-avg-17-context-1-max-sym-per-frame-1-blank-penalty-0.0-use-averaged-model.txt'
errorfile = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms/greedy_search/errs-test-greedy_search-epoch-19-avg-4-context-2-max-sym-per-frame-1-use-averaged-model.txt'
# errorfile = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_FFT/greedy_search/errs-test-greedy_search-epoch-19-avg-14-context-2-max-sym-per-frame-1-use-averaged-model.txt'
# errorfile = 'finetune_hubert_transducer/exp_aishell_phone_model_f0_emb_conv_att_fft_40ms_convFFT/greedy_search/errs-test-greedy_search-epoch-7-avg-6-context-2-max-sym-per-frame-1-use-averaged-model.txt'
import re
pattern = re.compile('[a-z]*[0-9]')
with open(errorfile, 'r', encoding='utf8') as input:
    data = input.readlines()
    start_idx = 0 
    end_idx = 0
    for line_idx in range(len(data)):
        if 'SUBSTITUTIONS: count ref -> hyp' in data[line_idx]:
            start_idx = line_idx
        elif 'DELETIONS: count ref' in data[line_idx]:
            end_idx = line_idx

        if start_idx != 0 and end_idx != 0:
            break

    data_error_count = data[start_idx+1:end_idx-1]
    print(data_error_count[0], data_error_count[-1])

    tone_error = 0
    tone_error_list = []
    for error in data_error_count:
        # 1   a3 -> uang3
        count = error.strip().split('   ')[0]
        try:
            ref = error.strip().split('   ')[1].split(' -> ')[0]
            hyp = error.strip().split('   ')[1].split(' -> ')[1]
        except Exception as e:
            print(e, error)
        ref_phone = ref[:-1]
        ref_tone = ref[-1]
        hyp_phone = hyp[:-1]
        hyp_tone = hyp[-1]
        if pattern.match(ref) and hyp_phone == ref_phone and hyp_tone != ref_tone:
            tone_error += 1
            tone_error_list.append(error.strip())
    print(tone_error / len(data_error_count))
    print(tone_error)
    print(len(data_error_count))
    # print(tone_error_list)
    # print(tone_error)
