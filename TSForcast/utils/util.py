import copy
import datetime
import logging
import os

import random

import numpy as np
import torch
from torch import nn
from typing import Any

mse_criterion = nn.MSELoss()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_params = None

    def __call__(self, val_loss, model, path=None, is_saved=False):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.__save_checkpoint(val_loss, model, path, is_saved)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.__save_checkpoint(val_loss, model, path, is_saved)
            self.counter = 0

    def __save_checkpoint(self, val_loss, model, path, is_save):
        if is_save:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
        self.best_model_params = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss





class Config(dict):
    """
    A configuration class that converts a dictionary into a class-like object
    where dictionary keys can be accessed as class attributes.

    Note: Attributes are dynamically created based on the input dictionary.
    """

    def __init__(self, data: dict):
        super().__init__()
        for key, value in data.items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def convert_items(data):
    new = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            new[key] = value.item()
        else:
            new[key] = value
    return new


def getLogger(path):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(time)
    logger.setLevel('INFO')
    os.makedirs(path, exist_ok=True)
    fileHander = logging.FileHandler(filename=f'{path}/{time}.txt', mode='w')
    streamHander = logging.StreamHandler()
    format = logging.Formatter(fmt='%(message)s')
    streamHander.setFormatter(format)
    logger.addHandler(streamHander)
    logger.addHandler(fileHander)
    return logger


class Train_Result:
    def __init__(self) -> None:
        super().__init__()
        self.result_dict = None
        self.count = 0
        self.is_first = True

    def add_dicts(self, dict):
        if self.result_dict:
            for key, value in dict.items():
                self.result_dict[key] = value + self.result_dict[key]

        else:
            self.result_dict = dict
        self.count += 1

    def clear(self):
        self.result_dict = None
        self.count = 0
        self.is_first = True

    def getResult(self):
        if self.is_first:
            if self.result_dict:
                for key, value in self.result_dict.items():
                    self.result_dict[key] = self.result_dict[key] / self.count
            self.is_first = False
        return self.result_dict


def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def calculate_pred_loss(pred, target):
    return  mse_criterion(pred, target)





