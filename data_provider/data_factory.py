from data_provider.data_loader import TrainSegLoader, TrainSampleLoader
from torch.utils.data import DataLoader
from .read_data import data_info
from .batch_scheduler import BatchSchedulerSampler
import os
from torch.utils.data.sampler import SequentialSampler
import random


# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())


class SequentialSampler_Shuffle(SequentialSampler):
    def __init__(self, data_source, num_data=None):
        self.data_source = data_source
        self.index_list = list(range(len(self.data_source)))
        random.shuffle(self.index_list)
        if num_data is None:
            self.num_data = len(self.index_list)
        else:
            self.num_data = num_data
            self.index_list = self.index_list[:num_data]

    def __iter__(self):
        return iter(self.index_list)

    def __len__(self):
        return self.num_data


def DataProvider(root_path, datasets, batch_size, win_size=100, step=100, mode="test", percentage=1, sampler=False):
    if mode == "train":
        shuffle = True
    else: 
        shuffle = False
        percentage=1
    print(f"loading {datasets}({mode}) percentage: {percentage*100}% ...", end="")
    filenames, train_lens, file_nums, discrete_channels = data_info(root_path=root_path, dataset=datasets)
    assert file_nums == 1
    data_path = os.path.join(root_path, filenames[0])
    if sampler:
        data_set = TrainSegLoader(data_path, train_lens[0], win_size, step, mode, 1, discrete_channels)
        sampler = SequentialSampler_Shuffle(data_set, num_data=int(len(data_set)*percentage))
        data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=8, drop_last=False, sampler=sampler)
    else:
        data_set = TrainSegLoader(data_path, train_lens[0], win_size, step, mode, percentage, discrete_channels)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=False)
    print("done!")
    return data_set, data_loader


def AnormDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    anorm_loader = DataLoader(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return anorm_dataset, anorm_loader


def PretrainDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    norm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Norm", nums=nums)
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    norm_loader = DataLoader(
        dataset=norm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    anorm_loader = DataLoader(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return norm_loader, anorm_loader