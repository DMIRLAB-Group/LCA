import os
import time
from itertools import cycle

import numpy as np
import torch
from torch import device, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset_provider import Custom_Dataset
from algorithm.model_set import get_model_class
from utils.util import getLogger, convert_items, Train_Result, EarlyStopping, setSeed
from utils.metrics import metric
from utils.util import calculate_pred_loss


class Trainer:

    def __init__(self, config, train_config_data) -> None:
        super().__init__()

        self.config = config
        self.train_config_data = train_config_data
        self.device = device(
            self.config.device) if torch.cuda.is_available() and self.config.device != "gpu" else device("cpu")
        self.logger_path = os.path.join(self.config.logger_root_path, self.config.model, self.config.dataset)

        self.tgt_train_dl, self.tgt_val_dl, self.tgt_test_dl = None, None, None
        self.src_train_dl, self.src_val_dl, self.src_test_dl = None, None, None
        self.optimizer, self.criterion, self.model, self.early_stopping = None, None, None, None
        self.logger, self.current_seed, self.current_src, self.current_tgt = None, None, None, None

    def __get_dataset(self, doman_name):
        path = os.path.join(self.config.dataset, doman_name)

        train_dataset, val_dataset, test_dataset = Custom_Dataset(dataset_name=path + "_train.csv",
                                                                  input_len=self.config.input_len,
                                                                  pred_len=self.config.pred_len,
                                                                  is_normalized=self.config.is_normalized), \
            Custom_Dataset(dataset_name=path + "_valid.csv",
                           input_len=self.config.input_len,
                           pred_len=self.config.pred_len,
                           is_normalized=self.config.is_normalized), \
            Custom_Dataset(dataset_name=path + "_test.csv",
                           input_len=self.config.input_len,
                           pred_len=self.config.pred_len,
                           is_normalized=self.config.is_normalized)
        train_dl = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_dl = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_dl, val_dl, test_dl

    def __run_once_init(self):
        self.model = get_model_class(self.config.model)(self.config).float().to(self.config.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)

        self.early_stopping = EarlyStopping(patience=self.config.early_stop)

        self.criterion = nn.MSELoss()

    def run(self):

        result = []
        self.current_src, self.current_tgt = self.config.src, self.config.trg
        self.logger = getLogger(os.path.join(self.logger_path, f"{self.current_src}_{self.current_tgt}"))
        result_path = f"{self.config.save_root_path}/{self.config.model}/{self.config.dataset}"

        os.makedirs(result_path, exist_ok=True)
        self.result_file = f"{result_path}/{self.config.input_len}_{self.config.pred_len}.txt"
        f = open(self.result_file, 'a')
        f.write(f" {self.current_src}--{self.current_tgt}\n")
        f.write(f"train_param:{self.config}\n")
        f.close()

        for seed in self.config.seeds:
            self.current_seed = seed
            setSeed(self.current_seed)
            once_result = self.run_once()
            result.append(once_result)

        mean_tgt_rmse, mean_tgt_mse, mean_tgt_mae = np.array(result).mean(
            axis=0)
        f = open(self.result_file, 'a')

        self.logger.info(
            '--tgt--:mean_rmse:{} ,mean_mse:{}, mean_mae:{}'.format(mean_tgt_rmse, mean_tgt_mse, mean_tgt_mae))
        f.write('--tgt--:mean_rmse:{} ,mean_mse:{}, mean_mae:{}\n'.format(mean_tgt_rmse, mean_tgt_mse, mean_tgt_mae))
        f.write('\n\n')
        f.close()

    def run_once(self):
        self.src_train_dl, self.src_val_dl, self.src_test_dl = self.__get_dataset(self.current_src)
        self.tgt_train_dl, self.tgt_val_dl, self.tgt_test_dl = self.__get_dataset(self.current_tgt)
        self.__run_once_init()
        self.logger.info(f"\n-------{self.config.dataset}: {self.current_src}-->{self.current_tgt}---------")
        self.logger.info(f"-------seed:{self.current_seed}---------")

        max_len = max(len(self.src_train_dl), len(self.tgt_train_dl))
        if len(self.src_train_dl) < len(self.tgt_train_dl):
            self.src_train_dl = cycle(self.src_train_dl)
        else:
            self.tgt_train_dl = cycle(self.tgt_train_dl)
        train_result = Train_Result()

        for epoch in range(1, self.config.train_epochs + 1):
            self.logger.info(f'\n--------------epoch:{epoch}---------------')
            train_result.clear()
            self.model.train()
            time_s = time.time()
            for i, (src_data, tgt_data) in enumerate(zip(self.src_train_dl, self.tgt_train_dl)):
                sx, sy = src_data
                tx, ty = tgt_data

                sx, sy = sx.to(self.config.device), sy.to(self.config.device)

                tx, ty = tx.to(self.config.device), ty.to(self.config.device)

                self.optimizer.zero_grad()
                output = self.model(sx, tx, sy)

                output["total_loss"].backward()

                self.optimizer.step()
                train_result.add_dicts(convert_items(output))

            self.logger.info(f"train:{train_result.getResult()}")
            tgt_val_loss = self.val(self.tgt_val_dl, "tgt_val loss")
            time_e = time.time()
            self.logger.info(f"time {time_e - time_s}")
            self.early_stopping(tgt_val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        tgt_mae, tgt_mse, tgt_rmse, tgt_mape, tgt_mspe = self.test(self.tgt_test_dl)

        self.logger.info("\n--------------Test result---------------")
        self.logger.info('tgt---rmse:{} mse:{}, mae:{}'.format(tgt_rmse, tgt_mse, tgt_mae))

        f = open(self.result_file, 'a')
        f.write(f"current_seed:{self.current_seed}\n")
        f.write('--tgt--:rmse:{} ,mse:{}, mae:{}\n'.format(tgt_rmse, tgt_mse, tgt_mae))
        f.write('\n')
        f.close()
        return [tgt_rmse, tgt_mse, tgt_mae]

    def test(self, dl):

        self.model.load_state_dict(self.early_stopping.best_model_params)
        self.model.eval()
        output_list = []
        ground_truth = []
        with torch.no_grad():
            for i, batch in enumerate(dl):
                x, y = batch
                x = x.to(self.config.device)
                output = self.model.inference(x)

                output_list.append(output.cpu())
                ground_truth.append(y)

            output = np.array(torch.cat(output_list, dim=0))
            ground_truth = np.array(torch.cat(ground_truth, dim=0))

            if self.config.is_normalized:
                output = dl.dataset.inverse_transform(output)
                ground_truth = dl.dataset.inverse_transform(ground_truth)

            mae, mse, rmse, mape, mspe = metric(output, ground_truth)
        return mae, mse, rmse, mape, mspe

    def val(self, dl, name):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, batch in enumerate(dl):
                x, y = batch
                x = x.to(self.config.device)
                y = y.to(self.config.device)

                output = self.model.inference(x)

                loss = calculate_pred_loss(output, y).item()

                loss_list.append(loss)
        mean_loss = np.mean(loss_list)
        self.logger.info(f"{name}:{mean_loss}")
        return mean_loss
