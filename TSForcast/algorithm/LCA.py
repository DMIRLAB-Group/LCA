import torch
import torch.distributions as D
from einops import rearrange
from torch import nn
from torch.jit import fork, wait
from torch.nn import functional as F
from layer.StandardNorm import Normalize
from utils.util import calculate_pred_loss


class Embedding_Net(nn.Module):

    def __init__(self, patch_size, input_len, out_len, emb_dim) -> None:
        super().__init__()
        self.patch_size = patch_size if patch_size <= input_len else input_len
        self.stride = self.patch_size // 2
        self.out_len = out_len

        self.num_patches = int((input_len - self.patch_size) / self.stride + 1)

        self.net1 = MLP(1, in_dim=self.patch_size, out_dim=emb_dim)
        self.net2 = MLP(1, emb_dim * self.num_patches, out_dim=self.out_len)

    def forward(self, x):
        B, L, M = x.shape
        if self.num_patches != 1:
            x = rearrange(x, 'b l m -> b m l')
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, 'b m n p -> (b m) n p')
        else:
            x = rearrange(x, 'b l m -> (b m) 1 l')
        x = self.net1(x)
        outputs = self.net2(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b  l m', b=B)
        return outputs


class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_norm - 2):
                net.append(nn.Linear(in_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.PReLU())

            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.PReLU())
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NPTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.L = lags

        gs = [NLayerLeakyMLP(in_features=lags * latent_size + 1,
                             out_features=1,
                             num_layers=num_layers,
                             hidden_dim=hidden_dim) for _ in range(latent_size)]

        self.gs = nn.ModuleList(gs)

    def forward(self, x):
        batch_size, length, input_dim = x.shape
        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L + 1, input_dim)

        yy, xx = x[:, -1:], x[:, :-1]
        xx = xx.reshape(-1, self.L * input_dim)

        residuals = []

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([xx] + [yy[:, :, i]], dim=-1)

            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = torch.vmap(torch.func.jacfwd(self.gs[i]))(inputs)

            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))

            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac


class Base_Net(nn.Module):
    def __init__(self, input_len, out_len, input_dim, out_dim, is_mean_std=True, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len
        self.c_type = c_type
        self.out_dim = out_dim
        self.c_type = "type1" if out_dim != input_dim and c_type == "None" else c_type
        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=out_len * 2,
                           activation=activation,
                           layer_norm=layer_norm)
        elif self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=self.out_dim * 2,
                           out_dim=self.out_dim * self.radio,
                           layer_norm=layer_norm, activation=activation)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=self.out_dim * 2 * input_len,
                           activation=activation,
                           out_dim=self.out_dim * input_len * self.radio, layer_norm=layer_norm)

        self.dropout_net = nn.Dropout(drop_out)

    def forward(self, x):
        if self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim)

        elif self.c_type == "None":
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout_net(x)
        if self.radio == 2:
            dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
            x = torch.chunk(x, dim=dim, chunks=2)
        return x


class Hidden_Pre_Net(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.z_pred_net = MLP(2, config.input_len, config.pred_len * 2, config.pred_len * 2)

    def forward(self, x):
        mean, std = torch.chunk(self.z_pred_net(x.permute(0, 2, 1)), dim=-1, chunks=2)
        mean, std = mean.permute(0, 2, 1), std.permute(0, 2, 1)
        return mean, std


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        config.z_dim=config.feature_dim
        self.config = config

        self.emb_net = Embedding_Net(config.patch_size, config.input_len, config.input_len, config.emb_dim)

        self.encoder = Base_Net(self.config.input_len, config.input_len, config.feature_dim,
                                config.z_dim,
                                layer_norm=config.is_ln, activation=config.activation,
                                drop_out=config.dropout, layer_nums=self.config.layer_nums)
        self.hidden_Pre_Net = Base_Net(config.input_len, config.pred_len, config.z_dim, config.z_dim,
                                       layer_norm=config.is_ln, activation=config.activation,
                                       drop_out=config.dropout, layer_nums=self.config.layer_nums)

        self.rec_decoder = Base_Net(config.input_len, config.input_len, config.z_dim, config.feature_dim,
                                    c_type=config.type, layer_norm=config.is_ln, activation=config.activation,
                                    drop_out=config.dropout, layer_nums=self.config.layer_nums, is_mean_std=False)
        self.pred_decoder = Base_Net(config.pred_len, config.pred_len, config.z_dim, config.feature_dim,
                                     c_type=config.type, layer_norm=config.is_ln, activation=config.activation,
                                     drop_out=config.dropout, layer_nums=self.config.layer_nums, is_mean_std=False)

        self.threa = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.register_buffer('base_dist_mean', torch.zeros(self.config.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.config.z_dim))
        self.transition_prior_fix = NPTransitionPrior(lags=1,
                                                      latent_size=self.config.z_dim,
                                                      num_layers=1,
                                                      hidden_dim=4)
        self.criterion = nn.MSELoss()
        self.normalize = Normalize(num_features=config.feature_dim, affine=True, non_norm=not self.config.is_norm)

    def forward(self, xs, xt, ys):
        (src_rec_x, src_pred_x), (src_z_rec_mean, src_z_rec_std, src_z_rec), (
            src_z_pred_mean, src_z_pred_std, src_z_pred) = self.__get_pred(xs)
        (trg_rec_x, trg_pred_x), (trg_z_rec_mean, trg_z_rec_std, trg_z_rec), (
            trg_z_pred_mean, trg_z_pred_std, trg_z_pred) = self.__get_pred(xt)

        src_z_mean = torch.cat([src_z_rec_mean, src_z_pred_mean], dim=1)
        src_z_std = torch.cat([src_z_rec_std, src_z_pred_std], dim=1)
        src_z = torch.cat([src_z_rec, src_z_pred], dim=1)
        trg_z = torch.cat([trg_z_rec, trg_z_pred], dim=1)
        rec_loss, pre_loss, sparsity_loss, kld_loss, structure_loss = self.__loss_function(xs, src_rec_x, ys,
                                                                                           src_pred_x, src_z_mean,
                                                                                           src_z_std, src_z, xt,
                                                                                           trg_rec_x, trg_z,
                                                                                           no_kl=self.config.is_no_prior)

        rec_loss = rec_loss * self.config.rec_weight
        kld_loss = kld_loss * self.config.z_kl_weight
        sparsity_loss = sparsity_loss * self.config.sparsity_weight
        structure_loss = structure_loss * self.config.structure_weight
        total_loss = rec_loss + pre_loss + structure_loss + kld_loss + sparsity_loss
        total_loss = {"total_loss": total_loss, "pred_loss": pre_loss, "rec_loss": rec_loss,
                      "sparsity_loss": sparsity_loss, "structure_loss": structure_loss, "kld_loss": kld_loss}

        return total_loss

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

    def __get_pred(self, x, is_train=True):
        std_x = self.normalize(x, "norm")
        emb = self.emb_net(std_x)
        z_rec_mean, z_rec_std = self.encoder(emb)
        z_rec = self.__reparametrize(z_rec_mean, z_rec_std) if is_train else z_rec_mean
        z_pre_mean, z_pre_std = self.hidden_Pre_Net(z_rec)
        z_pred = self.__reparametrize(z_pre_mean, z_pre_std) if is_train else z_pre_mean

        rec_x = self.rec_decoder(z_rec)

        rec_x = self.normalize(rec_x, "denorm")
        pred_x = self.pred_decoder(z_pred)

        pred_x = self.normalize(pred_x, "denorm")

        return (rec_x, pred_x), (z_rec_mean, z_rec_std, z_rec), (z_pre_mean, z_pre_std, z_pred)

    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def inference(self, x):
        (_, pred_x), _, _ = self.__get_pred(x, is_train=False)

        return pred_x

    @property
    def base_dist(self):

        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def __loss_function(self, src_x, src_rec_x, src_y, src_pre_y, src_mus, src_logvars, src_z, trg_x, trg_rec_x, trg_z,
                        no_kl=False):

        rec_loss = self.criterion(src_rec_x, src_x) + self.criterion(trg_rec_x, trg_x)

        # pre_loss = calculate_pred_loss(src_pre_y, src_y, self.config.pred_type == "m_s")
        pre_loss = calculate_pred_loss(src_pre_y, src_y)
        if no_kl:
            return rec_loss, pre_loss, 0, 0, 0

        batch_size, length, z_dim = src_mus.shape
        rate = batch_size * length * z_dim

        q_dist = D.Normal(src_mus, torch.exp(src_logvars / 2))

        log_qz = q_dist.log_prob(src_z)

        p_dist = D.Normal(torch.zeros_like(src_mus[:, :self.config.lags]),
                          torch.ones_like(src_logvars[:, :self.config.lags]))
        log_pz_normal = torch.sum(p_dist.log_prob(src_z[:, :self.config.lags]), dim=[-2, -1])
        log_qz_normal = torch.sum(log_qz[:, :self.config.lags], dim=[-2, -1])
        kld_normal = (torch.abs(log_qz_normal - log_pz_normal) / self.config.lags).sum()

        log_qz_laplace = log_qz[:, self.config.lags:]

        residuals, logabsdet, src_hist_jac = self.transition_prior_fix.forward(src_z)
        _, _, trg_hist_jac = self.transition_prior_fix.forward(trg_z)

        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
        kld_future = (torch.abs(torch.sum(log_qz_laplace, dim=[-2, -1]) - log_pz_laplace) / (
                length - self.config.lags)).sum()

        kld_loss = (kld_normal + kld_future) / rate

        structure_loss = 0
        sparsity_loss = 0
        for src_jac, trg_jac in zip(src_hist_jac, trg_hist_jac):
            sparsity_loss = sparsity_loss + F.l1_loss(src_jac[:, 0, :self.config.lags * self.config.z_dim],
                                                      torch.zeros_like(
                                                          src_jac[:, 0, :self.config.lags * self.config.z_dim]),
                                                      reduction='sum')

            threshold = self.soft_quantile(src_jac.flatten(), self.threa)

            I_J1_src = ((src_jac > threshold).float() - threshold).detach() + threshold
            I_J1_trg = ((trg_jac > threshold).float() - threshold).detach() + threshold

            mask = torch.abs(I_J1_src - I_J1_trg)

            structure_loss = structure_loss + torch.sum((src_jac * mask.detach() - trg_jac * mask) ** 2)
        sparsity_loss = sparsity_loss / rate
        structure_loss = structure_loss / rate
        return rec_loss, pre_loss, sparsity_loss, kld_loss, structure_loss
