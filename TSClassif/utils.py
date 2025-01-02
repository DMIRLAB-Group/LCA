import argparse

import torch
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, log_dir, last_model, best_model):
    save_dict = {
        "last": last_model,
        "best": best_model
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")

    torch.save(save_dict, save_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


class simple_MLP(nn.Module):
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units // 2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units // 2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x


def calculate_risk(target_model, risk_dataloader, device):
    if type(risk_dataloader) == tuple:
        x_data = torch.cat((risk_dataloader[0].dataset.x_data, risk_dataloader[1].dataset.x_data), axis=0)
        y_data = torch.cat((risk_dataloader[0].dataset.y_data, risk_dataloader[1].dataset.y_data), axis=0)
    else:
        x_data = risk_dataloader.dataset.x_data
        y_data = risk_dataloader.dataset.y_data

    feat = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()


class DictAsObject:
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"'DictAsObject' object has no attribute '{name}'")


def merge_args(main_args, model_args):
    merged_args = argparse.Namespace()
    for arg in vars(main_args):
        setattr(merged_args, arg, getattr(main_args, arg))
    for arg in vars(model_args):
        setattr(merged_args, arg, getattr(model_args, arg))

    return merged_args