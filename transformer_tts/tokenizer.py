import json
import re

import numpy as np
import torch
from g2pk import G2p


class Tokenizer:
    def __init__(self, token_id_dict_path):
        f = open(token_id_dict_path, "r", encoding="utf-8")

        self.token_id_dict = json.load(f)
        self.id_token_dict = {v: k for k, v in self.token_id_dict.items()}
        self.vocab_size = len(self.token_id_dict)
        self.g2p = G2p()

        f.close()
        self.setup_pattern()
        self.set_special_tokens()

    def setup_pattern(self):
        self.pattern = re.compile(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z0-9…~!?,.]")

    def set_special_tokens(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        self.pad_token_id = self.token_id_dict["[PAD]"]
        self.unk_token_id = self.token_id_dict["[UNK]"]

    def _processing(self, text):
        text = text.replace("...", "…")
        text = self.pattern.sub("", text)
        text = self.g2p(text, descriptive=True)
        return text

    def _encode_one(self, text):
        text = self._processing(text)
        encoded = [self.token_id_dict.get(w, 1) for w in text]
        return encoded

    def _decode_one(self, ids):
        decoded = [self.id_token_dict[i] for i in ids]
        return decoded

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

    def decode(self, ids):
        assert isinstance(ids, list), "ids must be list type."

        if isinstance(ids[0], int):
            ids = [ids]

        decoded = [self._decode_one(i) for i in ids]

        return decoded

    def pad(self, ids_list, max_length):
        result = np.full(
            (len(ids_list), max_length),
            self.pad_token_id,
            dtype=int,
        )
        for i, ids in enumerate(ids_list):
            max_idx = min(max_length, len(ids))
            result[i, :max_idx] = ids[:max_idx]

        return result

    def __call__(self, s1, max_length=None):
        result = [self.encode(s) for s in s1]

        keys = result[0].keys()
        result_dict = {key: [row[key] for row in result] for key in keys}

        if isinstance(max_length, int):
            result_dict = {k: self.pad(v, max_length) for k, v in result_dict.items()}

        return result_dict
