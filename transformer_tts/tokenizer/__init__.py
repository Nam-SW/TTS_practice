import re

import numpy as np
import torch
from g2pk import G2p

from .g2p import ALL_TOKENS, graph2prono, readRules


class Tokenizer:
    def __init__(self, rulebook_path: str):
        self.set_g2p(rulebook_path)
        self.set_pattern()
        self.set_tokens()

    def set_g2p(self, rulebook_path: str):
        self.rulebook = readRules(rulebook_path)
        self.g2p_module = G2p()

    def set_pattern(self):
        self.pattern = re.compile(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣0-9…~!?,.]")

    def set_tokens(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        self.pad_token_id = 0
        self.unk_token_id = 1

        self.token_dict = {t: i + 2 for i, t in enumerate(ALL_TOKENS)}
        self.vocab_size = len(self.token_dict) + 2

    def _num_to_str(self, text: str):
        text = (
            text.replace("0", "영")
            .replace("1", "일")
            .replace("2", "이")
            .replace("3", "삼")
            .replace("4", "사")
            .replace("5", "오")
            .replace("6", "육")
            .replace("7", "칠")
            .replace("8", "팔")
            .replace("9", "구")
        )
        return text

    def _processing(self, text):
        # text = self._num_to_str(text)
        text = text.replace("...", "…")
        text = self.pattern.sub("", text)
        text = graph2prono(self.g2p_module(text), *self.rulebook)
        return text.split()

    def _encode_one(self, text):
        text_list = self._processing(text)
        encoded = [self.token_dict.get(w, 1) for w in text_list]
        return encoded

    def encode(self, text, max_length=None, return_tensors=None):
        if isinstance(text, str):
            text = [text]

        input_ids = [self._encode_one(t) for t in text]

        if max_length is not None:
            input_ids = self.pad(input_ids, max_length)

        if return_tensors == "np":
            input_ids = np.array(input_ids)
        elif return_tensors == "pt":
            input_ids = torch.tensor(input_ids)

        return input_ids

    def pad(self, ids_list, max_length):
        result = np.full(
            (len(ids_list), max_length),
            self.pad_token_id,
            dtype=int,
        )
        for i, ids in enumerate(ids_list):
            max_idx = min(max_length, len(ids))
            result[i, :max_idx] = ids[:max_idx]

        return result.tolist()

    def __call__(self, text, max_length=None, return_tensors=None):
        ids = self.encode(text, max_length, return_tensors)

        if return_tensors is None:
            attention_mask = np.array(ids) == self.pad_token_id
            attention_mask = attention_mask.tolist()
        elif return_tensors in ["np", "pt"]:
            attention_mask = ids == self.pad_token_id

        return {"input_ids": ids, "attention_mask": attention_mask}
