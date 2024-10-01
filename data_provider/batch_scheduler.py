from torch.utils.data import RandomSampler, Sampler
from numpy import random


class BatchSchedulerSampler(Sampler):
    def __init__(self, dataset, batch_size, nums=-1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.dataset_size = [len(cur_dataset)
                             for cur_dataset in dataset.datasets]
        # self.len = max(self.dataset_size)*self.number_of_datasets
        if nums == -1:
            self.len = self.dataset.cumulative_sizes[-1]
        else:
            self.len = nums
        random.seed(1)

    def __len__(self):
        return self.len

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        # init child dataset sampler iterator
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        child_dataset_start_index = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        final_samples_index = []
        for _ in range(0, self.len, step):
            random_sampler_iterator_id = random.randint(
                low=0, high=self.number_of_datasets, size=self.number_of_datasets)
            for i in random_sampler_iterator_id:
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(self.batch_size):
                    try:
                        cur_sample_index = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_index + \
                            child_dataset_start_index[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # restart the iterator
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_index = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_index + \
                            child_dataset_start_index[i]
                        cur_samples.append(cur_sample)
                final_samples_index.extend(cur_samples)

        return iter(final_samples_index)
