import os
import random
from pathlib import Path

import librosa
import pandas as pd
import torch
import audioread

import logging
logger = logging.getLogger(__name__)

from utils import convert_audio


def _resolve_paths_from_df(df: pd.DataFrame, csv_path: str, mode: str, audio_root: Path = None):
    # 规范列名与split取值
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()

    # 选择路径列
    candidate_cols = [
        "audio_filename", "audio_filepath", "audio_path",
        "filepath", "file_path", "path", "wav", "wav_path"
    ]
    path_col = None
    for c in candidate_cols:
        if c in df.columns:
            path_col = c
            break

    # 一列CSV（老格式）兼容
    if path_col is None:
        if df.shape[1] == 1:
            path_series = df.iloc[:, 0].astype(str)
        else:
            # 多列但未识别到路径列，回退到第一列
            path_series = df.iloc[:, 0].astype(str)
    else:
        # MAESTRO等多列表头格式
        if "split" in df.columns:
            if mode == "train":
                df = df[df["split"] == "train"]
            elif mode == "test":
                # 合并 val+test 作为 test，以接近 8:2
                df = df[df["split"].isin(["validation", "val", "test"])]
        path_series = df[path_col].astype(str)

    # 音频根目录
    if audio_root is None:
        audio_root = Path(csv_path).parent

    # 拼接路径（相对路径拼接audio_root；绝对路径保持）
    resolved = []
    for p in path_series.tolist():
        p = p.strip()
        pp = Path(p)
        if pp.is_absolute():
            resolved.append(str(pp))
        else:
            resolved.append(str(audio_root / pp))
    return pd.Series(resolved)


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        self.mode = mode

        # 选择CSV路径（两者可都指向同一个 MAESTRO CSV）
        csv_path = config.datasets.train_csv_path if mode == 'train' else config.datasets.test_csv_path

        # 允许可选 audio_root，用于多列CSV相对路径拼接
        audio_root = Path(getattr(config.datasets, "audio_root", Path(csv_path).parent))

        # 读取CSV并解析出音频绝对/相对路径列表（同时兼容老格式与MAESTRO格式）
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except pd.errors.EmptyDataError:
            raise RuntimeError(f"Empty CSV: {csv_path}")

        # 自动兼容表头/无表头
        if df.shape[1] > 1:
            # 多列：按表头处理
            self.audio_files = _resolve_paths_from_df(df, csv_path, mode, audio_root)
        else:
            # 单列：每行一个路径
            self.audio_files = df.iloc[:, 0].astype(str)

        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)

    def get(self, idx=None):
        """uncropped, untransformed getter with random sample feature"""
        if idx is not None and idx >= len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))

        path = self.audio_files.iloc[idx]
        try:
            logger.debug(f'Loading {path}')
            waveform, sample_rate = librosa.load(
                path,
                sr=self.sample_rate,
                mono=self.channels == 1
            )
        except (audioread.exceptions.NoBackendError, ZeroDivisionError, FileNotFoundError):
            logger.warning(f"Not able to load {path}, removing from dataset")
            # 从Series删除该索引并重取一条
            self.audio_files = self.audio_files.drop(index=idx).reset_index(drop=True)
            return self.get(None)

        # 转张量并处理通道
        waveform = torch.as_tensor(waveform)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            waveform = waveform.expand(self.channels, -1)
        return waveform, sample_rate

    def __getitem__(self, idx):
        waveform, sample_rate = self.get(idx)

        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0 and waveform.size(1) > self.tensor_cut:
            start = random.randint(0, waveform.size(1) - self.tensor_cut - 1)
            waveform = waveform[:, start:start + self.tensor_cut]
        return waveform, sample_rate


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = [waveform for waveform, _ in batch]
    tensors = pad_sequence(tensors)
    return tensors