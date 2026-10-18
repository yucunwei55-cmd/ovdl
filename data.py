"""CinC2017 WFDB loader with shared filters and post-normalization augmentation."""
import os
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from preprocessing import preprocess_ecg


class ECGDataset(Dataset):
    """Native 300 Hz, N/A/O/~ labels, record-level preprocessing then windowing.

    time_steps_list counts ECG windows; repeat_times expands each window into
    repeated SNN inputs. This is direct input encoding, not temporal resampling.
    """
    def __init__(self, phase, data_dir, label_csv, timestep_length=9000,
                 repeat_times=3, overlap_ratio=.2, tail_padding='zero',
                 enable_time_warp=False, leads='1', duration='variable',
                 preprocessing='legacy', tail_mode='pad', seed=42, augment=True):
        super().__init__()
        self.phase, self.data_dir = phase, os.path.normpath(data_dir)
        self.labels = pd.read_csv(label_csv, header=None, names=['patient_id', 'label'],
                                  dtype={'patient_id': str})
        self.classes, self.n_classes = ['N', 'A', 'O', '~'], 4
        if (self.labels.patient_id.isna().any() or self.labels.patient_id.duplicated().any()
                or not self.labels.label.isin(self.classes).all()):
            raise ValueError('Expected unique record IDs with N/A/O/~ labels')
        self.timestep_length, self.repeat_times = int(timestep_length), int(repeat_times)
        if min(self.timestep_length, self.repeat_times) < 1:
            raise ValueError('Window length and repeat times must be positive')
        if duration not in ('fixed', 'variable') or tail_padding != 'zero':
            raise ValueError('Invalid duration/tail padding')
        if preprocessing not in ('legacy', 'corrected') or tail_mode not in ('pad', 'anchor'):
            raise ValueError('Invalid preprocessing/tail mode')
        if enable_time_warp:
            raise ValueError('Time warp is disabled to match CPSC2018 preprocessing')
        self.duration, self.preprocessing, self.tail_mode = duration, preprocessing, tail_mode
        self.seed, self.epoch, self.augment = seed, 0, augment
        self._overlap_ratio = float(overlap_ratio)
        if not 0 <= self._overlap_ratio < 1:
            raise ValueError('Overlap must be in [0, 1)')
        self._stride = max(1, int(self.timestep_length * (1-self._overlap_ratio)))
        self.lead_indices = None if leads == 'all' else [int(x)-1 for x in leads.split(',')]
        if self.lead_indices is not None and (not self.lead_indices or min(self.lead_indices) < 0
                or len(set(self.lead_indices)) != len(self.lead_indices)):
            raise ValueError('Use unique one-based lead indices or all')
        self.record_paths, self.lengths, self.time_steps_list = [], [], []
        for pid in self.labels.patient_id:
            path = os.path.join(self.data_dir, os.path.splitext(str(pid))[0])
            h = wfdb.rdheader(path)
            if abs(h.fs-300.) > 1e-6:
                raise ValueError(f'{pid}: expected native 300 Hz, got {h.fs}; resample explicitly first')
            if h.sig_len < 2:
                raise ValueError(f'{pid}: invalid or too-short ECG')
            if self.lead_indices is not None and max(self.lead_indices) >= h.n_sig:
                raise ValueError(f'{pid}: selected leads missing')
            count = 1 if duration == 'fixed' or h.sig_len <= self.timestep_length else (
                1+(h.sig_len-self.timestep_length+self._stride-1)//self._stride)
            self.record_paths.append(path)
            self.lengths.append(h.sig_len)
            self.time_steps_list.append(count)
        self.window_counts = self.time_steps_list
        self.actual_time_steps_list = [n*self.repeat_times for n in self.time_steps_list]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.record_paths[index]
        x, fields = wfdb.rdsamp(path)
        if len(x) != self.lengths[index] or abs(float(fields['fs'])-300.) > 1e-6:
            raise ValueError(f'Record metadata changed: {path}')
        if self.lead_indices is not None:
            x = x[:, self.lead_indices]
        x = preprocess_ecg(x, self.duration, self.timestep_length, self.preprocessing,
                           self.phase == 'train' and self.augment,
                           (self.seed+self.epoch*100003+index*997) % (2**32), fs=300.,
                           augmentation_profile='cinc2017')
        chunks = []
        for j in range(self.window_counts[index]):
            start = 0 if self.duration == 'fixed' else j*self._stride
            if self.duration == 'variable' and self.tail_mode == 'anchor' and len(x) > self.timestep_length:
                start = min(start, len(x)-self.timestep_length)
            chunk = x[start:start+self.timestep_length]
            if len(chunk) < self.timestep_length:
                chunk = np.pad(chunk, ((0, self.timestep_length-len(chunk)), (0, 0)))
            chunks.append(chunk)
        sequence = np.repeat(np.stack(chunks).astype(np.float32), self.repeat_times, axis=0)
        assert len(sequence) == self.actual_time_steps_list[index]
        label = self.classes.index(self.labels.iloc[index]['label'])
        return torch.from_numpy(sequence.copy()), torch.tensor(label, dtype=torch.long)
