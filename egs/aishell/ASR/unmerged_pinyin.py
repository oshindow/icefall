pinyin = {}
with open('data/lang_phone/uniq_lexicon.txt','r', encoding='utf8') as input:
    for line in input:
        utt = line.strip().split(' ')[0]
        phone_sequence = line.strip().split(' ')[1:]
        for phone_idx in range(0,len(phone_sequence),2):

            pinyin[" ".join(phone_sequence[phone_idx:phone_idx + 2])[:-1]] = 1

pinyin = sorted(pinyin.items(), key=lambda x:x[0])
finals = {}
for key, value in pinyin:
    if 'aa' in key or 'ee' in key or 'ii' in key or 'oo' in key or 'uu' in key:
        finals[key.split(' ')[1]] = 1
print(finals.keys())
