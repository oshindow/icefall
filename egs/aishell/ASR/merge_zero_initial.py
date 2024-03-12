lexicon = '/data2/xintong/aishell/resource_aishell/lexicon.txt'
merged_lexicon = '/data2/xintong/aishell/resource_aishell/lexicon_merge.txt'
# SIL sil
# <SPOKEN_NOISE> sil
# 啊 aa a1
# 啊 aa a2
# 啊 aa a4
# 啊 aa a5
# 啊啊啊 aa a2 aa a2 aa a2
# 啊啊啊 aa a5 aa a5 aa a5
output = open(merged_lexicon, 'w', encoding='utf8')
with open(lexicon, 'r', encoding='utf8') as input:
    for line in input:
        word = line.strip().split(' ')[0]
        phone_seq = line.strip().split(' ')[1:]
        new_phone_seq = [phone for phone in phone_seq if phone not in ['aa','ii','ee','oo','uu']]
        output.write(word + ' ' + " ".join(new_phone_seq) + '\n')
output.close()