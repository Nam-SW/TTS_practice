import glob
import os
import random as rd
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset


class MelWavDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        data_length: int = 64,
        hop_length: Union[int, str] = 256,
    ):
        self.files = files
        self.data_length = data_length
        self.hop_length = hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: slice):
        sample = torch.load(self.files[index])

        wave = sample["waveform"]

        spectrogram = sample["mel_spectrogram"]

        if self.data_length != "MAX":
            start = rd.randint(0, len(spectrogram[0]) - self.data_length - 1)
            spectrogram = spectrogram[:, start : start + self.data_length]

            start *= self.hop_length
            wave = wave[start : start + (self.data_length * self.hop_length)]

        return spectrogram, wave.unsqueeze(0)


def get_datasets(
    data_dir: Union[str, List[str]],
    data_length: int,
    hop_length: Union[int, str],
    train_ratio: Optional[float] = 0.8,
    seed: Optional[int] = 42,
):
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    files = []
    for path in data_dir:
        files += [f for f in glob.glob(os.path.join(path, "*.pt"))]

    rd.seed(seed)
    rd.shuffle(files)

    if train_ratio is not None:
        train_size = int(len(files) * train_ratio)

        train_files = files[:train_size]
        eval_files = files[train_size:]

        train_dataset = MelWavDataset(train_files, data_length, hop_length)
        eval_dataset = MelWavDataset(eval_files, data_length, hop_length)

    else:
        train_dataset = MelWavDataset(files, data_length, hop_length)
        eval_dataset = None

    return train_dataset, eval_dataset


def get_dataloader(dataset, *args, **kwargs):
    if dataset is None:
        return None

    return DataLoader(dataset, *args, **kwargs)
