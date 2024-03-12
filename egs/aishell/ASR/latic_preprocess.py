# 1. writing /data2/xintong/LATIC/SCRIPT/
#    text_latic.txt,text_latic_train_id.txt, and latic_dict.txt
#    python3 latic_preprocess.py 
# 2. prepare data/lang_phone_latic
#    prepare_latic.sh stage 5
zero_initials = [
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 
    'yv','yvan','yvn','yve',
    'wo', 'wa', 'wai', 'wan', 'wang', 'weng', 
    'o', 'ou'
]
u2v = {'ju','juan','jue','jun','qu','quan','que','qun','xu','xuan','xue','xun',
        'yue','yu','yuan','yun','lue'}

convert = {'yi':'iii','ya':'iiia','yan':'iiian','yang':'iiiang','yao':'iiiao','ye':'iiie','yin':'iiin','yong':'iiiong','you':'iiiu', 'ying': 'iiing','wen':'uuun','wei':'uuui','wu':'uuu'}
convert_merge = {'yi':'iii','ya':'iiia','yan':'iiian','yang':'iiiang','yao':'iiiao','ye':'iiie','yin':'iiin','yong':'iiiong','you':'iiiu', 'ying': 'iiing'}

initials = ['','b','c','ch','d','f','g','h','j','k','l','m','n','p','q','r','s','sh','t','w','x','y','z','zh']
finals = ['a', 'ai', 'an', 'ang', 'ao',													
    'e', 'ei', 'en', 'eng', 'er',			
    'i', 'in', 'ing', 'iao', 'ie', 'ian', 'iy', 'ix', 'ia', 'iang', 'iu', 'iong', 'iz',	
    'o', 'ong',	'ou',									
    'u', 'ui', 'un', 'uo', 'uan', 'ua', 'uai', 'uang', 'ueng',
    'v', 'vn', 've', 'van']	

def syllable2phoneme(syllable):
    initials = ['','b','c','ch','d','f','g','h','j','k','l','m','n','p','q','r','s','sh','t','w','x','y','z','zh']
    finals = ['a', 'ai', 'an', 'ang', 'ao',													
        'e', 'ei', 'en', 'eng', 'er',			
        'i', 'in', 'ing', 'iao', 'ie', 'ian', 'iy', 'ix', 'ia', 'iang', 'iu', 'iong', 'iz',	
        'o', 'ong',	'ou',									
        'u', 'ui', 'un', 'uo', 'uan', 'ua', 'uai', 'uang', 'ueng',
        'v', 'vn', 've', 'van']	
    initials += ['vv','ii','uu','aa','oo','ee']
    if syllable[0] in initials and syllable[:2] not in initials:
        initial = syllable[0]
        final = syllable[1:]
        
    elif syllable[:2] in initials: 
        initial = syllable[:2]
        final = syllable[2:]
        
    else:
        print(syllable)

    try:
        assert final[:-1] in finals and initial in initials
    # assert initial in initials
    except Exception as e:
        print(syllable, initial, final)

    return [initial,final]

rootdir = '/data2/xintong/LATIC/SCRIPT/Testing_Notations'
import os
text = open('/data2/xintong/LATIC/SCRIPT/text_latic.txt', 'w', encoding='utf8')
train_id = open('/data2/xintong/LATIC/SCRIPT/text_latic_train_id.txt', 'w', encoding='utf8')
syllable_list = {}
syllable_list_merge = {}
for file in os.listdir(rootdir):
    with open(os.path.join(rootdir, file), 'r', encoding='utf8') as input:
        for line in input:
            utt, syllable_str = line.strip().split('\t')
            syllable_seq = syllable_str.split(' ')
            print(syllable_seq)
            skip_utt = False
            for syllable_idx in range(len(syllable_seq)):
                syllable = syllable_seq[syllable_idx] # with tone

                if syllable[:-1] in convert:
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace(syllable[:-1],convert[syllable[:-1]])
                    
                if syllable_seq[syllable_idx][:-1] in u2v:
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('u','v')

                if syllable_seq[syllable_idx][:-1] in zero_initials:
                    if syllable[0] == 'y':
                        syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('y','vv')
                    elif syllable[0] == 'w':             
                        syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('w','uuu')
                    else:
                        syllable_seq[syllable_idx] = syllable[0] * 2 + syllable_seq[syllable_idx]
                
                if 'si' in syllable or 'ci' in syllable or 'zi' in syllable:       
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','iy')

                if 'shi' in syllable or 'chi' in syllable or 'zhi' in syllable:
                    syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','ix')
                     
                if syllable in ['ng1', 'nvr3', 'ng3',
                                'ng5','zuir4' ,'bianr1' ,
                                'beir2' ,'ng4' ,'yir4' ,
                                'shenr2' ,'menr2' ,'dianr3' ,
                                'nar4' ,'weir4' ,'jinr4','ung1','ung2','ir4','uir4']:
                    skip_utt = True
                    break
                
                phonemes = syllable2phoneme(syllable_seq[syllable_idx])
                phonemes_merge = [phoneme for phoneme in phonemes if phoneme not in ['ii','uu','aa','oo','ee']]
                
                syllable_list[syllable] = phonemes
                syllable_list_merge[syllable] = phonemes_merge

            if not skip_utt:
                text.write(utt + ' ' + syllable_str + '\n')
                train_id.write(utt + '\n')

text.close()
train_id.close()


syllable_list_sort = sorted(syllable_list.items(), key=lambda x:x[0])
with open('/data2/xintong/LATIC/SCRIPT/latic_dict_1.txt', 'w', encoding='utf8') as output:
    for key, value in syllable_list_sort:
        if value != 1:
            output.write(key + ' ' + " ".join(value) + '\n')

syllable_list_sort_merge = sorted(syllable_list_merge.items(), key=lambda x:x[0])
with open('/data2/xintong/LATIC/SCRIPT/latic_dict_2_merge.txt', 'w', encoding='utf8') as output:
    for key, value in syllable_list_sort_merge:
        if value != 1:
            output.write(key + ' ' + " ".join(value) + '\n')


