# 1. writing /data2/xintong/LATIC/SCRIPT/
#    text_latic.txt,text_latic_train_id.txt, and latic_dict.txt
#    python3 latic_preprocess.py 
# 2. prepare data/lang_phone_latic
#    prepare_latic.sh stage 5
rootdir = '/data2/xintong/LATIC/SCRIPT/Orignial_Notations'
import os
text = open('/data2/xintong/LATIC/SCRIPT/text_latic_original.txt', 'w', encoding='utf8')
# train_id = open('/data2/xintong/LATIC/SCRIPT/text_latic_train_id.txt', 'w', encoding='utf8')
syllable_list = {}
for file in os.listdir(rootdir):
    with open(os.path.join(rootdir, file), 'r', encoding='utf8') as input:
        for line in input:
            utt, syllable_str = line.strip().split('\t')
            syllable_seq = syllable_str.split(' ')
            skip_utt = False
            for syllable_idx in range(len(syllable_seq)):
                syllable = syllable_seq[syllable_idx]
                if 'que' in syllable or 'xue' in syllable or \
                'jue' in syllable or 'yue' in syllable or \
                'lue' in syllable:
                    syllable = syllable.replace('ue','ve')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('ue','ve')
                if 'qu' in syllable or 'xu' in syllable or \
                'ju' in syllable:
                    syllable = syllable.replace('u','v')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('u','v')

                    # print(syllable)
                if 'si' in syllable or 'ci' in syllable or 'zi' in syllable:
                    syllable = syllable.replace('i','iy')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','iy')

                if 'shi' in syllable or 'chi' in syllable or 'zhi' in syllable:
                    syllable = syllable.replace('i','ix')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','ix')
                if 'yi' in syllable:
                    syllable = syllable.replace('yi','i')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yi','i')
                if 'you' in syllable:
                    syllable = syllable.replace('you','iu')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('you','iu')
                if 'yu' in syllable:
                    syllable = syllable.replace('yu','vvv')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yu','vvv')
                if 'yv' in syllable:
                    syllable = syllable.replace('yv','vvv')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yv','vvv')
                if 'ye' in syllable:
                    syllable = syllable.replace('ye','ie')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('ye','ie')
                if 'wo' in syllable:
                    syllable = syllable.replace('wo','uo')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('wo','uo')
                if 'wei' in syllable:
                    syllable = syllable.replace('wei','ui')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('wei','ui')
                if 'wen' in syllable:
                    syllable = syllable.replace('wen','un')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('wen','un')
                if 'wu' in syllable:
                    syllable = syllable.replace('wu','u')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('wu','u')
                if 'yao' in syllable:
                    syllable = syllable.replace('yao','iao')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yao','iao')
                if 'yan' in syllable:
                    syllable = syllable.replace('yan','ian')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yan','ian')
                if 'yong' in syllable:
                    syllable = syllable.replace('yong','iong')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yong','iong')
                if 'yang' in syllable:
                    syllable = syllable.replace('yang','iang')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('yang','iang')                
                if 'quan' in syllable:
                    syllable = syllable.replace('quan','qvan')
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('quan','qvan')             
                if syllable in ['ng1', 'nvr3', 'ng3',
                                'ng5','zuir4' ,'bianr1' ,
                                'beir2' ,'ng4' ,'yir4' ,
                                'shenr2' ,'menr2' ,'dianr3' ,
                                'nar4' ,'weir4' ,'jinr4','ung1','ung2','ir4','uir4']:
                    skip_utt = True
                    break
                syllable_list[syllable] = 1
            if not skip_utt:
                text.write(utt + ' ' + " ".join(syllable_seq)+ '\n')
                # train_id.write(utt + '\n')

text.close()
# train_id.close()

# initials = ['','b','c','ch','d','f','g','h','j','k','l','m','n','p','q','r','s','sh','t','w','x','y','z','zh','vv']
# finals = ['a', 'ai', 'an', 'ang', 'ao',													
# 'e', 'ei', 'en', 'eng', 'er',			
# 'i', 'in', 'ing', 'iao', 'ie', 'ian', 'iy', 'ix', 'ia', 'iang', 'iu', 'iong', 'iz',	
# 'o', 'ong',	'ou',									
# 'u', 'ui', 'un', 'uo', 'uan', 'ua', 'uai', 'uang', 'ueng',
# 'v', 'vn', 've', 'van']											

# for syllable, value in syllable_list.items():
#     if syllable[0] in initials and syllable[:2] not in initials:
#         initial = syllable[0]
#         final = syllable[1:]
#         # tone = syllabel[-1]
#     elif syllable[:2] in initials:
#         initial = syllable[:2]
#         final = syllable[2:]
#         # tone = syllabel[-1]
#     elif syllable[:-1] in finals:
#         initial = ''
#         final = syllable
#     else:
#         print(syllable)
#     try:
#         assert final[:-1] in finals and initial in initials
#     # assert initial in initials
#     except Exception as e:
#         print(syllable, initial, final)
#         continue
#     syllable_list[syllable] = [initial,final]

# syllable_list_sort = sorted(syllable_list.items(), key=lambda x:x[0])
# # with open('/data2/xintong/LATIC/SCRIPT/latic_dict.txt', 'w', encoding='utf8') as output:
# #     for key, value in syllable_list_sort:
# #         if value != 1:
# #             output.write(key + ' ' + " ".join(value) + '\n')


