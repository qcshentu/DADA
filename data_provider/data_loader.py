import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from .read_data import read_data

warnings.filterwarnings('ignore')


class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step, mode="train", percentage=0.1, discrete_channels=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = read_data(data_path)
        # 2.train
        train_data = data.iloc[:train_length, :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = data.iloc[train_length:, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        # 4.process
        if discrete_channels is not None:
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if mode == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        index = index * self.step
        if self.mode == "train":           
            return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_label[index: index + self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_label[index: index + self.win_size])
        elif self.mode == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        elif self.mode == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size: index// self.step * self.win_size+ self.win_size]), np.float32(self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
        
    
class TrainSampleLoader(Dataset):
    def __init__(self, data_path, datasets, win_size, step, type="Norm", nums=-1):
        datasets = datasets.split(",")
        self.samples_list = []
        for dataset in datasets:
            print(f"loading {dataset}({type})...", end=" ")
            step = 50
            file_path = f"{data_path}/AnomalyDatasets_TestSets/{dataset}/{type}/{win_size}_{step}"
            if dataset=="Monash":
                step = 50
                file_path = f"{data_path}/MonashSamples/{type}/{win_size}_{step}"
            if dataset=="Forecast_800M" or dataset=="AnomalyDatasets":
                step = 50
                _path = f"{data_path}/{dataset}/"
                file_paths = os.listdir(_path)
                for file_path in file_paths:
                    file_path = os.path.join(_path, file_path)
                    file_path = f"{file_path}/{type}/{win_size}_{step}"
                    filenames = os.listdir(file_path)
                    samplenames = [os.path.join(file_path, filename) for filename in filenames]
                    self.samples_list.extend(samplenames)
                print("done!")
                continue
            filenames = os.listdir(file_path)
            samplenames = [os.path.join(file_path, filename) for filename in filenames]
            self.samples_list.extend(samplenames)
            print("done!")
        
        if nums != -1:
            nums =  min(len(self.samples_list), nums)
            self.samples_list = self.samples_list[:nums]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):      
        data = np.load(self.samples_list[index])
        return data[0], data[1]
    