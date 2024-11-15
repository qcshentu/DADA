import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import argparse
from injection import inject_anomalies
from sklearn.preprocessing import StandardScaler


class GenMonashPlus:
    def __init__(self,
                 data_source = "./dataset/",
                 sample_save_path = "./pretrain_dataset",
                 max_anomaly_ratio = 0.10,
                 rng = None,
                 save_type = 'hdf5'
                ):
        self.data_source = data_source
        self.sample_save_path = sample_save_path
        self.max_anomaly_ratio = max_anomaly_ratio
        self.rng = rng
        self.save_type = save_type
        self.norm_count = 0
        self.anorm_count = 0

    def __call__(self, win_size, step=50, folder_name="Monash"):
        data_source = os.path.join(self.data_source, folder_name)
        self.norm_save_path = f"{self.sample_save_path}/{folder_name}+/Norm/{win_size}_{step}"
        self.anorm_save_path = f"{self.sample_save_path}/{folder_name}+/Anorm/{win_size}_{step}"
        if not os.path.exists(self.norm_save_path):
            os.makedirs(self.norm_save_path)
        if not os.path.exists(self.anorm_save_path):
            os.makedirs(self.anorm_save_path) 
        files = os.listdir(data_source)
        for name in tqdm(files):
            data_name = os.path.join(data_source, name)
            data =  pd.read_csv(data_name)
            series = data.iloc[:, 1].values
            scaler = StandardScaler()
            scaler.fit(series.reshape(-1, 1))
            series, series_label = inject_anomalies(series, max_anomaly_ratio=self.max_anomaly_ratio, rng=self.rng)
            series = series.reshape(-1, 1)
            series_label = series_label.reshape(-1, 1)
            series = scaler.transform(series)
            norm_samples, abnorm_samples = self._generate_sample(series, series_label, win_size=win_size, step=step)
            name = name.split('.')[0]
            norm_save_path = f"{self.norm_save_path}/{name}"
            anorm_save_path = f"{self.anorm_save_path}/{name}"
            if self.save_type == 'hdf5':
                with h5py.File(f"{norm_save_path}.h5", 'w') as hf:
                    hf.create_dataset('data', data=norm_samples, compression='gzip')
                with h5py.File(f"{anorm_save_path}.h5", 'w') as hf:
                    hf.create_dataset('data', data=abnorm_samples, compression='gzip')
            elif self.save_type == 'npy':
                np.save(norm_save_path, norm_samples)
                np.save(anorm_save_path, abnorm_samples)
            else:
                raise Exception("save_type is invaild.")

    def _generate_sample(self, series, series_label, win_size, step):
        n_samples = (len(series) - win_size) // step + 1
        norm_index = []
        abnorm_index = []
        data = np.zeros((n_samples, 2, win_size, 1))
        for i in range(n_samples):
            index = i * step
            sample = series[index: index + win_size]
            sample_label = series_label[index: index + win_size]
            is_abnomal = any(sample_label)
            data[i, 0] = sample
            data[i, 1] = sample_label
            if is_abnomal:
                abnorm_index.append(i)
                self.anorm_count += 1
            else:
                norm_index.append(i)
                self.norm_count += 1
        norm_samples = data[norm_index]
        abnorm_samples = data[abnorm_index]
        return norm_samples, abnorm_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument('--rand_seed', type=int, default=2024)
    parser.add_argument('--root_path', type=str, required=True, default='./dataset')
    parser.add_argument('--save_path', type=str, required=True, default='./dataset/pretrain_dataset')
    parser.add_argument('--data', type=str, required=True, default='Monash')
    parser.add_argument('--save_type', type=str, default='hdf5')
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--step', type=int, default=50)
    args = parser.parse_args()

    # gen Monash+
    rng = np.random.default_rng(args.rand_seed)
    g = GenMonashPlus(data_source=args.root_path, sample_save_path=args.save_path, rng=rng, save_type=args.save_type)
    g(folder_name=args.data, win_size=args.win_size, step=args.step)
