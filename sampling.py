"""精确时长分组：不丢弃尾 batch，不重复记录，不按类别加权采样。

保持 V2 的整组随机顺序。未改为混排时长 batch，避免改变已取得的结果。
每个 epoch 都从原始索引复制再打乱，恢复 RNG 后可准确续跑。
"""
import random
import numpy as np
import torch
from torch.utils.data import Sampler

class TimeStepGroupSampler(Sampler):
    def __init__(self, time_steps_list, batch_size, shuffle=True):
        self.time_steps_list = time_steps_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = {}
        for idx, ts in enumerate(time_steps_list):
            if ts not in self.groups:
                self.groups[ts] = []
            self.groups[ts].append(idx)
        self.group_indices = list(self.groups.values())

    def __iter__(self):
        group_indices = self.group_indices[:]
        if self.shuffle:
            np.random.shuffle(group_indices)
        for group in group_indices:
            g = group[:]
            if self.shuffle:
                np.random.shuffle(g)
            for i in range(0, len(g), self.batch_size):
                yield g[i:i + self.batch_size]

    def __len__(self):
        total = 0
        for group in self.group_indices:
            total += (len(group) + self.batch_size - 1) // self.batch_size
        return total

def custom_collate_fn(batch):
    datas, labels = zip(*batch)  # datas: [T,L,C]
    datas = torch.stack([d.contiguous() for d in datas])  # [B,T,L,C]
    labels = torch.stack(labels)
    return datas, labels

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
