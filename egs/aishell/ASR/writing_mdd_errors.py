from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir + '/' +  f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        print(key, results)
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        print(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir +  '/' +   f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        print("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir +  '/' + f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    print(s)

from icefall.phone_graph_compiler import PhoneCtcTrainingGraphCompiler
from icefall.lexicon import Lexicon

device = 'cpu'
lexicon2 = Lexicon('data/lang_phone_latic')
graph_compiler2 = PhoneCtcTrainingGraphCompiler(
    'data/lang_phone_latic',
    device=device,
    oov="<UNK>",
    sos_id=1,
    eos_id=1,
)

# manual labeled
texts = {}
with open('text_latic.txt', 'r', encoding='utf8') as input:
    for line in input:
        utt = line.strip().split(' ')[0]
        text_seq = line.strip().split(' ')[1:]
        # texts.append(text_str)
        phone_ids = graph_compiler2.texts_to_ids(text_seq)
        texts[utt] = []
        for btc in phone_ids:
            for phone_id in btc:
                texts[utt].append(lexicon2.token_table[phone_id])

# canoncial labeled
original_texts = {}
with open('text_latic_original.txt', 'r', encoding='utf8') as input:
    for line in input:
        utt = line.strip().split(' ')[0]
        if utt not in texts:
            continue
        text_seq = line.strip().split(' ')[1:]
        # texts.append(text_str)
        phone_ids = graph_compiler2.texts_to_ids(text_seq)
        original_texts[utt] = []
        for btc in phone_ids:
            for phone_id in btc:
                original_texts[utt].append(lexicon2.token_table[phone_id])

try:
    assert len(original_texts) == len(texts)
except:
    print(len(original_texts), len(texts))
    
results = defaultdict(list)
for name, ref_texts in original_texts.items():
    this_batch = []
    # print(name, ref_texts)
    this_batch.append((name, ref_texts, texts[name]))

    # print(this_batch)
    results['canoncial'].extend(this_batch)

class Param:
    def __init__(self, res_dir, suffix) -> None:
        self.res_dir = res_dir
        self.suffix = suffix


params = Param(res_dir='/home/xintong/icefall/egs/aishell/ASR/data', suffix='manual')
test_set_name = 'latic'
results_dict = results

save_results(params, test_set_name, results_dict)