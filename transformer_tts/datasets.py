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

        input_ids = self.tokenizer.encode(
            sample["text"], max_length=self.max_ids_length
        )
        input_ids = torch.nn.functional.pad(
            input_ids, (0, self.max_ids_length - input_ids.size(-1))
        )
        spectrogram = sample["mel_spectrogram"]

        spectrogram = torch.nn.functional.pad(
            spectrogram,
            (0, self.max_spectrogram_length - spectrogram.size(-1)),
        )

        return input_ids, spectrogram


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

    rd.seed(seed)
    rd.shuffle(files)

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
