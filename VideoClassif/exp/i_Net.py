from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import torch

from torch import nn
import torch.distributions as D
from torch.nn import functional as F


class BackBone(nn.Module):
    def __init__(self, configs):
        super(BackBone, self).__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout_rate,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout_rate,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.configs = configs
        self.enc_embedding = nn.Linear(configs.feature_dim, configs.d_model)

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        if not self.configs.No_encoder:
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out


class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        elif activation == "leakyReLU":
            a_f = nn.LeakyReLU()
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


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class NPTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.L = lags
        gs = [MLP(in_dim=lags * latent_size + 1,
                  out_dim=1,
                  layer_nums=num_layers,
                  hid_dim=hidden_dim, activation="leakyReLU", layer_norm=False) for _ in range(latent_size)]

        self.gs = nn.ModuleList(gs)

    def forward(self, x):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # prepare data

        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)

        x = x.reshape(-1, self.L + 1, input_dim)
        yy, xx = x[:, -1:], x[:, :-1]
        xx = xx.reshape(-1, self.L * input_dim)
        # get residuals and |J|
        residuals = []

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([xx] + [yy[:, :, i]], dim=-1)

            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = torch.func.vmap(torch.func.jacfwd(self.gs[i]))(inputs)
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config

        self.backbone = BackBone(config)

        self.z_net = Base_Net(input_len=config.num_segments, out_len=config.num_segments, input_dim=config.d_model,
                              out_dim=config.z_dim, layer_nums=config.layer_nums, c_type="type2",
                              hidden_dim=config.d_model // 2, layer_norm=config.is_ln, activation=config.activation,
                              drop_out=config.dropout_rate)
        self.class_net = Base_Net(input_len=config.num_segments, out_len=1, input_dim=config.z_dim,
                                  out_dim=config.num_class, layer_nums=config.layer_nums, c_type="type2",
                                  activation=config.activation, drop_out=config.dropout_rate,
                                  is_mean_std=False, hidden_dim=config.z_dim * 2, layer_norm=config.is_ln)
        self.rec_net = Base_Net(input_len=config.num_segments, out_len=config.num_segments, input_dim=config.z_dim,
                                activation=config.activation,
                                out_dim=config.feature_dim, layer_nums=config.layer_nums + 1, c_type="type2",
                                is_mean_std=False, hidden_dim=config.d_model, layer_norm=False)
        # self.threa = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.rec_criterion = nn.MSELoss()
        self.register_buffer('base_dist_mean', torch.zeros(config.z_dim))
        self.register_buffer('base_dist_var', torch.eye(config.z_dim))
        self.transition_prior_fix = NPTransitionPrior(lags=1,
                                                      latent_size=config.z_dim,
                                                      num_layers=2,
                                                      hidden_dim=16)

        # self.apply(init_weights)

    def forward(self, source_data, target_data, src_label, epoch):
        (src_z_mean, src_z_std, src_z), src_rec, src_pred_class = self.__get_features(source_data)
        (tgt_z_mean, tgt_z_std, tgt_z), tgt_rec, tgt_pred_class = self.__get_features(target_data)
        class_loss, rec_loss, sparsity_loss, kld_loss, structure_loss \
            = self.__loss_function(src_z_mean, src_z_std, src_z, source_data, src_rec, src_pred_class,
                                   tgt_z_mean, tgt_z_std, tgt_z, target_data, tgt_rec, tgt_pred_class, src_label,
                                   epoch, no_kl=self.config.No_prior)
        loss = (
                class_loss * self.config.class_weight + rec_loss * self.config.rec_weight + sparsity_loss * self.config.sparsity_weight
                + kld_loss * self.config.z_kl_weight + structure_loss * self.config.structure_weight)
        return {"total_loss": loss, "c_loss": class_loss, "rec_loss": rec_loss,
                "sparsity_loss": sparsity_loss, "structure_loss": structure_loss, "kld_loss": kld_loss}

    #
    def __get_features(self, x, is_train=True):
        out = self.backbone(x)
        z_mean, z_std = self.z_net(out)
        z = self.__reparametrize(z_mean, z_std) if is_train else z_mean
        rec = self.rec_net(z)

        pred = torch.squeeze(self.class_net(z.reshape(x.shape[0], -1)))

        return (z_mean, z_std, z), rec, pred

    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def inference(self, x):
        _, _, pred = self.__get_features(x, is_train=False)
        return pred

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

    def __loss_function(self, src_z_mean, src_z_std, src_z, src_x, src_rec, src_class, tgt_z_mean,
                        tgt_z_std, tgt_z, tgt_x, tgt_rec, tgt_class, src_label, epoch, no_kl=True):

        x = torch.cat((src_x, tgt_x), dim=0)
        rec = torch.cat((src_rec, tgt_rec), dim=0)
        class_loss = self.config.criterion_src(src_class, src_label)

        pseudo_cls_loss = 0
        if epoch > self.config.start_psuedo_step:
            tat_p = F.softmax(tgt_class, dim=-1)
            prob, pseudo_label = tat_p.max(dim=-1)
            conf_mask = (prob > self.config.tar_psuedo_thre)
            if conf_mask.any().item():
                print("entering conf_mask")
                pseudo_cls_loss = self.config.criterion_src(tgt_class[conf_mask], pseudo_label[conf_mask])
            class_loss = class_loss + pseudo_cls_loss
        if no_kl:
            return (class_loss, torch.tensor(0, device=x.device), torch.tensor(0, device=x.device),
                    torch.tensor(0, device=x.device), torch.tensor(0, device=x.device))
        rec_loss = self.rec_criterion(x, rec) / x.shape[0]
        z_mean = torch.cat((src_z_mean, tgt_z_mean), dim=0)
        z_std = torch.cat((src_z_std, tgt_z_std), dim=0)
        z = torch.cat((src_z, tgt_z), dim=0)

        b, length, _ = z_mean.shape
        q_dist = D.Normal(z_mean, torch.exp(z_std / 2))

        log_qz = q_dist.log_prob(z)

        p_dist = D.Normal(torch.zeros_like(z_mean[:, :self.config.lags]),
                          torch.ones_like(z_std[:, :self.config.lags]))
        log_pz_normal = torch.sum(p_dist.log_prob(z[:, :self.config.lags]), dim=[-2, -1])
        log_qz_normal = torch.sum(log_qz[:, :self.config.lags], dim=[-2, -1])

        kld_normal = (torch.abs(log_qz_normal - log_pz_normal) / self.config.lags).sum()
        log_qz_laplace = log_qz[:, self.config.lags:]

        residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(z)

        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
        kld_future = ((torch.sum(log_qz_laplace, dim=[-2, -1]) - log_pz_laplace) / (
                length - self.config.lags)).mean()

        kld_loss = (kld_normal + kld_future) / self.config.z_dim


        structure_loss = torch.tensor(0, device=x.device)
        sparsity_loss = torch.tensor(0, device=x.device)
        for jac in hist_jac:
            sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, :self.config.lags * self.config.z_dim],
                                                      torch.zeros_like(
                                                          jac[:, 0, :self.config.lags * self.config.z_dim]),
                                                      reduction='sum')
            src_jac, trg_jac = torch.chunk(jac, dim=0, chunks=2)
            threshold = torch.quantile(src_jac, self.config.threshold)
            I_J1_src = (src_jac > threshold).bool()
            I_J1_trg = (trg_jac > threshold).bool()

            mask = torch.bitwise_xor(I_J1_src, I_J1_trg)
            structure_loss = structure_loss + torch.sum((src_jac[mask].detach() - trg_jac[mask]) ** 2)

        sparsity_loss = sparsity_loss / self.config.batch_size
        structure_loss = structure_loss
        return class_loss, rec_loss, sparsity_loss, kld_loss, structure_loss

    @property
    def base_dist(self):
        # Noise density function

        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)


class Base_Net(nn.Module):
    def __init__(self, input_len, out_len, input_dim, out_dim, hidden_dim, is_mean_std=True, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len
        self.c_type = c_type
        self.out_dim = out_dim
        self.c_type = "type1" if out_dim != input_dim and c_type == "None" else c_type
        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=hidden_dim,
                           activation=activation,
                           layer_norm=layer_norm)
        elif self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=hidden_dim,
                           out_dim=self.out_dim * self.radio,
                           layer_norm=layer_norm, activation=activation)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=hidden_dim * 2 * out_len,
                           activation=activation,
                           out_dim=self.out_dim * out_len * self.radio, layer_norm=layer_norm)

        self.dropout_net = nn.Dropout(drop_out)

    def forward(self, x):
        if self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim * self.radio)

        elif self.c_type == "None":
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout_net(x)
        if self.radio == 2:
            dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
            x = torch.chunk(x, dim=dim, chunks=2)
        return x
