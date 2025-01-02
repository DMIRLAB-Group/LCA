import utils
from configs.LCA_config import get_model_a_parser
from trainers.train import Trainer

import argparse

parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase', default='train', type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='test_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='EXP1', type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='LCA', type=str)

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'dataset', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda:0", type=str, help='cpu or cuda')

    # arguments
    args, remaining_args = parser.parse_known_args()

    # create trainier object
    trainer = Trainer(args)

    if args.da_method == "LCA":
        LCA_args = get_model_a_parser().parse_args(remaining_args)
        trainer.LCA_config = LCA_args
    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()
