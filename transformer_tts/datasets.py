import glob
import os
import random as rd
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset


class TextMelDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        tokenizer,
        max_ids_length: int = 128,
        max_spectrogram_length: int = 512,
    ):
        self.files = files
        self.tokenizer = tokenizer
        self.max_ids_length = max_ids_length
        self.max_spectrogram_length = max_spectrogram_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: slice):
        sample = torch.load(self.files[index])

        encoded = self.tokenizer(
            sample["text"],
            max_length=self.max_ids_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        mel = sample["mel_spectrogram"][:, : self.max_spectrogram_length]
        mel_mask_origin = sample["mel_mask"][:, : self.max_spectrogram_length]
        input_mel = torch.nn.functional.pad(
            torch.concat([torch.zeros((mel.size(0), 1)), mel[:, :-1]], dim=1),
            (0, self.max_spectrogram_length - mel.size(-1)),
        )
        mel_label = torch.nn.functional.pad(
            mel,
            (0, self.max_spectrogram_length - mel.size(-1)),
        )
        mel_mask = torch.full(input_mel.size(), True)
        mel_mask[:, : mel.size(-1)] = mel_mask_origin

        stop_label = torch.zeros(input_mel.size(-1), dtype=torch.float32)
        stop_label[mel.size(-1) - 1] = 1

        return (
            {
                "input_ids": input_ids,
                "input_mel": input_mel,
                "attention_mask": attention_mask,
                "decoder_attention_mask": mel_mask[0],
            },
            {"mel_label": mel_label, "stop_label": stop_label},
        )


def get_datasets(
    data_dir: Union[str, List[str]],
    tokenizer,
    max_ids_length: int = 128,
    max_spectrogram_length: int = 512,
    train_ratio: Optional[float] = 0.8,
    seed: Optional[int] = 42,
):
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    files = []
    for path in data_dir:
        files += glob.glob(os.path.join(path, "*.pt"))

    # rd.seed(seed)
    # rd.shuffle(files)

    if train_ratio is not None:
        train_size = int(len(files) * train_ratio)

        train_files = files[:train_size]
        eval_files = files[train_size:]

        train_dataset = TextMelDataset(
            train_files,
            tokenizer,
            max_ids_length,
            max_spectrogram_length,
        )
        eval_dataset = TextMelDataset(
            eval_files,
            tokenizer,
            max_ids_length,
            max_spectrogram_length,
        )

    else:
        train_dataset = TextMelDataset(
            files,
            tokenizer,
            max_ids_length,
            max_spectrogram_length,
        )
        eval_dataset = None

    return train_dataset, eval_dataset


def get_dataloader(dataset, *args, **kwargs):
    return DataLoader(dataset, *args, **kwargs)
