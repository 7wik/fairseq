# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool
import numpy as np
import torch
from fairseq import utils
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class ReplacementDictionary:
    """A mapping from symbols to corresponding replacement tokens and vice versa"""

    def __init__(
        self,
        # begin keyword-only arguments
        # bos="<s>",
        # pad="<pad>",
        # eos="</s>",
        # unk="<unk>",
        # extra_special_symbols=None,
    ):
        self.word_idxs = []#np.array([], dtype=np.int64)
        self.replacement_toks = []#np.array([], dtype=np.int64)
        self.cur_idx = 0
        self.unk_word = 0
        self.word2tok = None #{}
        self.tok2word = {}
        # self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        # self.symbols = []
        # self.count = []
        # self.indices = {}
        # self.bos_index = self.add_symbol(bos)
        # self.pad_index = self.add_symbol(pad)
        # self.eos_index = self.add_symbol(eos)
        # self.unk_index = self.add_symbol(unk)
        # if extra_special_symbols:
        #     for s in extra_special_symbols:
        #         self.add_symbol(s)
        # self.nspecial = len(self.symbols)

    # def __eq__(self, other):
    #     return self.indices == other.indices

    def __getitem__(self, idx):
        # return self.replacement_toks[idx]
        return self.word2tok[(idx)]
        # if idx < len(self.symbols):
        #     return self.symbols[idx]
        # return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.replacement_toks)

    # def __contains__(self, sym):
    #     return sym in self.ind

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        return self.word2tok[int(sym)]
        # if sym in self.indices:
        #     return self.indices[sym]
        # return self.unk_index

    def add_symbol(self, word, tok):
        """Adds a word to the dictionary"""
        # if word in self.word2tok.keys():
        #     return tok #self.word2tok[word]
        # else:
        try:
            self.word2tok[int(word)] = tok
        except:
            self.word2tok[self.cur_idx] = tok
        self.tok2word[tok] = (word)
        self.replacement_toks.append(tok)
        self.word_idxs.append(word)
        # self.replacement_toks = np.insert(self.replacement_toks, self.cur_idx,(tok))
        # self.word_idxs = np.insert(self.word_idxs, self.cur_idx, word)
        self.cur_idx+=1
        return tok
        # if word in self.indices and not overwrite:
        #     idx = self.indices[word]
        #     self.count[idx] = self.count[idx] + n
        #     return idx
        # else:
        #     idx = len(self.symbols)
        #     self.indices[word] = idx
        #     self.symbols.append(word)
        #     self.count.append(n)
        #     return idx

    # def update(self, new_dict):
    #     """Updates counts from new dictionary."""
    #     for word in new_dict.symbols:
    #         idx2 = new_dict.indices[word]
    #         if word in self.indices:
    #             idx = self.indices[word]
    #             self.count[idx] = self.count[idx] + new_dict.count[idx2]
    #         else:
    #             idx = len(self.symbols)
    #             self.indices[word] = idx
    #             self.symbols.append(word)
    #             self.count.append(new_dict.count[idx2])

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing replacement dictionary from a text file and adds its symbols
        to this instance.
        """
        # print(f,"+++++++++++++++++++++++++++++++")
        # bre
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        self.word2tok = np.zeros(len(lines), dtype=np.int64)
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            # try:
            word, field = line.rstrip().rsplit(" ", 1)
            # print(line,"))))))))))))))))))))))))))))))))))))))))))))))))",word, field)
            # if field == "#fairseq:overwrite":
            #     overwrite = True
            #     line, field = line.rsplit(" ", 1)
            # else:
            #     overwrite = False
            token = int(field)
            # word = line
            self.add_symbol(word, token)
            # except:
            #     pass
            # except ValueError:
            #     raise ValueError(
            #         "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
            #     )

    # def _save(self, f, kv_iterator):
    #     if isinstance(f, str):
    #         PathManager.mkdirs(os.path.dirname(f))
    #         with PathManager.open(f, "w", encoding="utf-8") as fd:
    #             return self.save(fd)
    #     for k, v in kv_iterator:
    #         print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    # def save(self, f):
    #     """Stores dictionary into a text file"""
    #     ex_keys, ex_vals = self._get_meta()
    #     self._save(
    #         f,
    #         zip(
    #             ex_keys + self.symbols[self.nspecial :],
    #             ex_vals + self.count[self.nspecial :],
    #         ),
    #     )

    # def dummy_sentence(self, length):
    #     t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
    #     t[-1] = self.eos()
    #     return t

    # def encode_line(
    #     self,
    #     line,
    #     line_tokenizer=tokenize_line,
    #     add_if_not_exist=True,
    #     consumer=None,
    #     append_eos=True,
    #     reverse_order=False,
    # ) -> torch.IntTensor:
    #     words = line_tokenizer(line)
    #     if reverse_order:
    #         words = list(reversed(words))
    #     nwords = len(words)
    #     ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

    #     for i, word in enumerate(words):
    #         if add_if_not_exist:
    #             idx = self.add_symbol(word)
    #         else:
    #             idx = self.index(word)
    #         if consumer is not None:
    #             consumer(word, idx)
    #         ids[i] = idx
    #     if append_eos:
    #         ids[nwords] = self.eos_index
    #     return ids

    # @staticmethod
    # def _add_file_to_dictionary_single_worker(
    #     filename, tokenize, eos_word, worker_id=0, num_workers=1
    # ):
    #     counter = Counter()
    #     with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
    #         size = os.fstat(f.fileno()).st_size
    #         chunk_size = size // num_workers
    #         offset = worker_id * chunk_size
    #         end = offset + chunk_size
    #         f.seek(offset)
    #         if offset > 0:
    #             safe_readline(f)  # drop first incomplete line
    #         line = f.readline()
    #         while line:
    #             for word in tokenize(line):
    #                 counter.update([word])
    #             counter.update([eos_word])
    #             # f.tell() returns only an opaque number which can
    #             # return to the position in the file via f.seek()
    #             # and does not necessarily represent a byte position
    #             # in the file. However, f.tell() is faithful to the
    #             # byte position _most of the time_. Thus we can just
    #             # check against the file size to prevent early exit.
    #             if f.tell() > end and f.tell() < size:
    #                 break
    #             line = f.readline()
    #     return counter

    # @staticmethod
    # def add_file_to_dictionary(filename, dict, tokenize, num_workers):
    #     def merge_result(counter):
    #         for w, c in sorted(counter.items()):
    #             dict.add_symbol(w, c)

    #     if num_workers > 1:
    #         pool = Pool(processes=num_workers)
    #         results = []
    #         for worker_id in range(num_workers):
    #             results.append(
    #                 pool.apply_async(
    #                     Dictionary._add_file_to_dictionary_single_worker,
    #                     (filename, tokenize, dict.eos_word, worker_id, num_workers),
    #                 )
    #             )
    #         pool.close()
    #         pool.join()
    #         for r in results:
    #             merge_result(r.get())
    #     else:
    #         merge_result(
    #             Dictionary._add_file_to_dictionary_single_worker(
    #                 filename, tokenize, dict.eos_word
    #             )
    #         )


class TruncatedDictionary(object):
    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {},
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
