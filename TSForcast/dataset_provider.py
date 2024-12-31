import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

root_path = './dataset'


class Custom_Dataset(Dataset):

    def __init__(self, dataset_name, input_len, pred_len, is_normalized) -> None:
        super().__init__()
        self.dataset_name = os.path.join(root_path, dataset_name)
        self.input_len = input_len
        self.pred_len = pred_len
        df = pd.read_csv(self.dataset_name).drop(['wind_direction', "weather"], axis=1).iloc[:, 1:].select_dtypes(
            include=[float, int]) if "AIR" in dataset_name else pd.read_csv(self.dataset_name).iloc[:,
                                                                1:].select_dtypes(include=[float, int])

        self.dataset = torch.tensor(df.values, dtype=torch.float)


        if is_normalized:
            self.mean = torch.mean(self.dataset, dim=0)
            self.std = torch.std(self.dataset, dim=0)
            self.dataset = (self.dataset - self.mean) / self.std

    def __getitem__(self, index):
        x, y = self.dataset[index:index + self.input_len], self.dataset[
                                                           index + self.input_len:index + self.input_len + self.pred_len]
        return x, y

    def __len__(self):
        return len(self.dataset) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return (data * np.array(self.std)) + np.array(self.mean)
