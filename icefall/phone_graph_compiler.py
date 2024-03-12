# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List
from pathlib import Path
import k2
import torch
from typing import Iterable, List, Tuple, Union
# from icefall.lexicon import Lexicon
from icefall.lexicon import UniqLexicon, read_lexicon

class PhoneCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        uniq_filename: str = "uniq_lexicon.txt",
        device: Union[str, torch.device] = "cpu",
        oov: str = "<UNK>",
        sos_id: int = 1,
        eos_id: int = 1,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang_char/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary token. When a word(token) in the transcript
            does not exist in the token list, it is replaced with `oov`.
        """

        self.lang_dir = Path(lang_dir)
        self.lexicon = UniqLexicon(lang_dir, uniq_filename=uniq_filename)
        self.device = torch.device(device)
        # self.lexicon_tmp = dict(read_lexicon(uniq_filename))
        self.L_inv = self.lexicon.L_inv.to(self.device)

        self.oov_id = self.lexicon.word_table[oov]
        self.sos_id = sos_id
        self.eos_id = eos_id

        # self.build_ctc_topo_P()
    
    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of token IDs.

        Args:
          texts:
            It is a list of strings.
            An example containing two strings is given below:

                ['你好中国', '北京欢迎您']
        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        whitespace = re.compile(r"([ \t])")
        # print(self.lang_dir)
        for text in texts:
            # text = re.sub(whitespace, "", text)
            text = text.split(' ')
            # print(text)
            sub_ids = []
            # print(self.lexicon.word_table['dan4'])
            for txt in text:
                
                if txt in self.lexicon.word_table:
                    # print('in word table')
                    for t in self.lexicon.text_to_token_ids(txt).tolist():
                      # print(t)
                      for it in t:
                        sub_ids.append(it)
                else:
                    sub_ids.append(self.oov_id)
            # print('sub_ids:', sub_ids)
            ids.append(sub_ids)
        return ids

    def texts_to_ids_with_bpe(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts (which include chars and bpes)
           to a list-of-list of token IDs.

        Args:
          texts:
            It is a list of strings.
            An example containing two strings is given below:

                [['你', '好', '▁C', 'hina'], ['北','京', '▁', 'welcome', '您']
        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        for text in texts:
            text = text.split("/")
            sub_ids = [
                self.token_table[txt] if txt in self.token_table else self.oov_id
                for txt in text
            ]
            ids.append(sub_ids)
        return ids

    def compile(
        self,
        token_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list token IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
         modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(token_ids, modified=modified, device=self.device)
        return graph
