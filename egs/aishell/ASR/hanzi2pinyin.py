# encoding=utf-8
from pypinyin import pinyin, lazy_pinyin, Style
import re

initials = ['','b','c','ch','d','f','g','h','j','k','l','m','n','p','q','r','s','sh','t','w','x','y','z','zh','vv','ii','uu','aa','oo','ee']
finals = ['a', 'ai', 'an', 'ang', 'ao',													
    'e', 'ei', 'en', 'eng', 'er',			
    'i', 'in', 'ing', 'iao', 'ie', 'ian', 'iy', 'ix', 'ia', 'iang', 'iu', 'iong', 'iz',	
    'o', 'ong',	'ou',									
    'u', 'ui', 'un', 'uo', 'uan', 'ua', 'uai', 'uang', 'ueng',
    'v', 'vn', 've', 'van']	

zero_initials = [
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 
    'yv','yvan','yvn','yve',
    'wo', 'wu', 'wa', 'wai', 'wan', 'wang', 'weng', 'wei','wen', 
    'o', 'ou'
]
u2v = {'ju','juan','jue','jun','qu','quan','que','qun','xu','xuan','xue','xun',
        'yue','yu','yuan','yun','lue'}

convert = {'yi':'iii','ya':'iiia','yan':'iiian','yang':'iiiang','yao':'iiiao','ye':'iiie','yin':'iiin','yong':'iiiong','you':'iiiu', 'ying': 'iiing'}
def convert_pypinyin_to_pinyin(syllable_seq):

    for syllable_idx in range(len(syllable_seq)):
        syllable = syllable_seq[syllable_idx][:-1]
        
        if syllable in convert:
            syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace(syllable,convert[syllable])

        if syllable_seq[syllable_idx][:-1] in u2v:
            syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('u','v')

        if syllable_seq[syllable_idx][:-1] in zero_initials:
            if syllable[0] == 'y':
                syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('y','vv')
            elif syllable[0] == 'w':
                syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('w','uu')
            else:
                syllable_seq[syllable_idx] = syllable[0] * 2 + syllable_seq[syllable_idx]
        
        if 'si' in syllable or 'ci' in syllable or 'zi' in syllable:       
            syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','iy')

        if 'shi' in syllable or 'chi' in syllable or 'zhi' in syllable:
            syllable_seq[syllable_idx] = syllable_seq[syllable_idx].replace('i','ix')
        
    return syllable_seq

def convert_pinyin_to_phone(syllabel_seq):
    		
    for syllable, value in syllabel_seq.items():
        if syllable[0] in initials and syllable[:2] not in initials:
            initial = syllable[0]
            final = syllable[1:]
            # tone = syllabel[-1]
        elif syllable[:2] in initials:
            initial = syllable[:2]
            final = syllable[2:]
            # tone = syllabel[-1]
        else:
            print(syllable)
        try:
            assert final[:-1] in finals and initial in initials
        # assert initial in initials
        except Exception as e:
            print(syllable, initial, final)
            continue
        syllabel_seq[syllable] = [initial,final]

    return [initial,final]