import argparse

import yaml

from configs import LCA_config

from utils.util import Config
from trainer import Trainer

configs_map = {
               "LCA": LCA_config}


def merge_args(main_args, model_args, dataset_dict):
    merged_args = argparse.Namespace()
    for arg in vars(main_args):
        setattr(merged_args, arg, getattr(main_args, arg))
    for arg in vars(model_args):
        setattr(merged_args, arg, getattr(model_args, arg))
    dataset_config = Config(dataset_dict)
    for arg in dataset_dict:
        setattr(merged_args, arg, getattr(dataset_config, arg))
    return merged_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train', allow_abbrev=False)

    parser.add_argument('--dataset', type=str, default='pems',
                        help='source domain train path')
    parser.add_argument('--input_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
    parser.add_argument("--model", type=str, default="LCA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--is_normalized", action='store_true')
    parser.add_argument("--src", type=str, default="domain1")
    parser.add_argument("--trg", type=str, default="domain2")
    parser.add_argument("--logger_root_path", type=str, default="./logger")
    parser.add_argument("--save_root_path", type=str, default="./result")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs='+', default=[2000, 1, 18])


    args, remaining_args = parser.parse_known_args()
    with open("./configs/dataset_config.yaml", 'r') as file:
        dataset_dict = yaml.safe_load(file)[args.dataset]

    model_parser = configs_map[args.model].get_model_a_parser()
    model_args = model_parser.parse_args(remaining_args)
    args = merge_args(args, model_args, dataset_dict)
    Trainer(args, model_args).run()
