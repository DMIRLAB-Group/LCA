import itertools
from copy import deepcopy

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from models.models import classifier, CNN
from models.LCA_models import Base_Net, NPTransitionPrior, init_weights
from utils import merge_args


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class LCA(torch.nn.Module):

    def __init__(self, model_config, configs, hparams, device):
        super(LCA, self).__init__()
        config = merge_args(configs, model_config)

        self.hparams = hparams
        self.config = config
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss()
        self.d_std = 0.1
        config.z_dim = configs.input_channels
        new_configs = deepcopy(configs)
        new_configs.input_channels = config.z_dim
        self.feature_extractor = CNN(new_configs)
        self.classifier = classifier(new_configs)
        self.rec_criterion = nn.MSELoss()

        self.z_net = Base_Net(input_len=config.sequence_len, out_len=config.sequence_len,
                              input_dim=configs.input_channels,
                              out_dim=config.z_dim, layer_nums=config.layer_nums, c_type=config.type,
                              hidden_dim=config.d_model // 2, layer_norm=config.is_ln, activation=config.activation,
                              drop_out=config.dropout_rate)

        self.rec_net = Base_Net(input_len=config.sequence_len, out_len=config.sequence_len, input_dim=config.z_dim,
                                activation=config.activation,
                                out_dim=config.input_channels, layer_nums=config.layer_nums + 1, c_type=config.type,
                                is_mean_std=False, hidden_dim=config.d_model, layer_norm=False)

        # self.adaptive=nn.AdaptiveAvgPool1d(1)
        #
        # self.d_classifier = nn.Sequential(nn.Linear(config.input_channels,config.input_channels*32),nn.BatchNorm1d(config.input_channels*32),nn.ReLU(),
        #                                   nn.Linear(config.input_channels*32,config.num_classes),nn.Softmax(dim=1)
        #                                   )

        self.threa = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        #

        self.register_buffer('base_dist_mean', torch.zeros(config.z_dim))
        self.register_buffer('base_dist_var', torch.eye(config.z_dim) * self.d_std)
        self.transition_prior_fix = NPTransitionPrior(lags=1, latent_size=config.z_dim, num_layers=2, hidden_dim=16)
        self.z_net.apply(init_weights)
        # self.feature_extractor.apply(weights_init)
        # self.classifier.apply(weights_init)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.lr,
            weight_decay=1e-4
        )

    def update(self, src_loader, trg_loader, avg_meter, logger, value_method):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            metic = value_method(is_train=False)[1]
            logger.debug(f'trg_value f1 is {metic}')
            logger.debug(f'-------------------------------------')

        last_model = self.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            (src_z_mean, src_z_std, src_z), src_rec, src_pred_class = self.get_features(src_x)
            (tgt_z_mean, tgt_z_std, tgt_z), tgt_rec, tgt_pred_class = self.get_features(trg_x)
            class_loss, rec_loss, sparsity_loss, kld_loss, structure_loss \
                = self.__loss_function(src_z_mean, src_z_std, src_z, src_x, src_rec, src_pred_class,
                                       tgt_z_mean, tgt_z_std, tgt_z, trg_x, tgt_rec, tgt_pred_class, src_y,
                                       epoch, no_kl=self.config.No_prior)

            # _,_,pred=self.get_features(src_x)
            # loss=self.cross_entropy(pred,src_y)
            loss = (
                    class_loss * self.config.class_weight + rec_loss * self.config.rec_weight + sparsity_loss * self.config.sparsity_weight
                    + kld_loss * self.config.z_kl_weight + structure_loss * self.config.structure_weight)
            losses = {"total_loss": loss.item(), "Src_cls_loss": class_loss.item(), "rec_loss": rec_loss.item(),
                      "sparsity_loss": sparsity_loss.item(), "structure_loss": structure_loss.item(),
                      "kld_loss": kld_loss.item()}
            # losses = {"total_loss": loss.item(), "Src_cls_loss": loss.item()}

            # zero grad
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def soft_quantile(self, tensor, quantile, temperature=1):
        sorted_tensor, _ = torch.sort(tensor)

        weights = F.softmax(
            torch.linspace(0, 1, tensor.size(0), device=tensor.device)
            .sub(quantile)
            .abs()
            .mul(-temperature),
            dim=0
        )
        return (weights * sorted_tensor).sum()

    def get_features(self, x, is_train=True):

        # x = self.normalize(x, "norm")
        z_mean, z_std = self.z_net(x.permute(0, 2, 1))
        z = self.__reparametrize(z_mean, z_std) if is_train else z_mean
        # z = z_mean
        pred = self.classifier(self.feature_extractor(z.permute(0, 2, 1)))
        rec_x = self.rec_net(z)
        # x=self.adaptive(x)
        # pred=self.d_classifier(x.reshape(x.size(0), -1))
        # return None,None,pred

        return (z_mean, z_std, z_std), rec_x, pred

    #
    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * self.d_std
        z = mu + std * eps
        return z

    def inference(self, x):
        _, _, pred = self.get_features(x, is_train=False)
        return pred

    def __loss_function(self, src_z_mean, src_z_std, src_z, src_x, src_rec, src_class, tgt_z_mean,
                        tgt_z_std, tgt_z, tgt_x, tgt_rec, tgt_class, src_label, epoch, no_kl=True):

        x = torch.cat((src_x, tgt_x), dim=0).permute(0, 2, 1)
        rec = torch.cat((src_rec, tgt_rec), dim=0)
        class_loss = self.cross_entropy(src_class, src_label)

        pseudo_cls_loss = 0
        if epoch > self.config.start_psuedo_step:
            tat_p = F.softmax(tgt_class, dim=-1)
            prob, pseudo_label = tat_p.max(dim=-1)
            conf_mask = (prob > self.config.tar_psuedo_thre)
            if conf_mask.any().item():
                pseudo_cls_loss = self.cross_entropy(tgt_class[conf_mask], pseudo_label[conf_mask])
            class_loss = class_loss + pseudo_cls_loss * 0.5
        if no_kl:
            return (class_loss, torch.tensor(0, device=x.device), torch.tensor(0, device=x.device),
                    torch.tensor(0, device=x.device), torch.tensor(0, device=x.device))
        rec_loss = self.rec_criterion(x, rec) / x.shape[0]
        z_mean = torch.cat((src_z_mean, tgt_z_mean), dim=0)
        z_std = torch.cat((src_z_std, tgt_z_std), dim=0)
        z = torch.cat((src_z, tgt_z), dim=0)

        b, length, z_dim = z_mean.shape
        rate = b * length * z_dim
        q_dist = D.Normal(z_mean, torch.exp(z_std / 2) * self.d_std)

        log_qz = q_dist.log_prob(z)

        p_dist = D.Normal(torch.zeros_like(z_mean[:, :self.config.lags]),
                          torch.ones_like(z_std[:, :self.config.lags]) * self.d_std)
        log_pz_normal = torch.sum(p_dist.log_prob(z[:, :self.config.lags]), dim=[-2, -1])
        log_qz_normal = torch.sum(log_qz[:, :self.config.lags], dim=[-2, -1])
        kld_normal = (torch.abs(log_qz_normal - log_pz_normal) / self.config.lags).sum()

        log_qz_laplace = log_qz[:, self.config.lags:]

        residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(z)

        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
        kld_future = (torch.abs(torch.sum(log_qz_laplace, dim=[-2, -1]) - log_pz_laplace) / (
                length - self.config.lags)).sum()

        kld_loss = (kld_normal + kld_future) / rate

        structure_loss = torch.tensor(0, device=x.device)
        sparsity_loss = torch.tensor(0, device=x.device)
        for jac in hist_jac:
            sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, :self.config.lags * self.config.z_dim],
                                                      torch.zeros_like(
                                                          jac[:, 0, :self.config.lags * self.config.z_dim]),
                                                      reduction='sum')
            src_jac, trg_jac = torch.chunk(jac, dim=0, chunks=2)
            # threshold = torch.quantile(src_jac, self.config.threshold)
            threshold = self.soft_quantile(src_jac.flatten(), self.threa)
            #
            # I_J1_src = (src_jac > threshold).bool()
            # I_J1_trg = (trg_jac > threshold).bool()
            I_J1_src = ((src_jac > threshold).float() - threshold).detach() + threshold
            I_J1_trg = ((trg_jac > threshold).float() - threshold).detach() + threshold

            # mask = torch.bitwise_xor(I_J1_src, I_J1_trg)
            mask = torch.abs(I_J1_src - I_J1_trg)
            # structure_loss = structure_loss + torch.sum((src_jac[mask].detach() - trg_jac[mask]) ** 2)
            structure_loss = structure_loss + torch.sum((src_jac * mask.detach() - trg_jac * mask) ** 2)
        sparsity_loss = sparsity_loss / rate
        structure_loss = structure_loss / rate
        return class_loss, rec_loss, sparsity_loss, kld_loss, structure_loss

    @property
    def base_dist(self):
        # Noise density function

        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
