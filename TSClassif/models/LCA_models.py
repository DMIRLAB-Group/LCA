import numpy as np
import torch
from einops import rearrange
from torch import nn
from math import sqrt
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

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







def init_weights(m):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            nn.init.zeros_(m.weight)
            dim = min(m.weight.size(0), m.weight.size(1))
            eye = torch.eye(dim)
            m.weight[:dim, :dim] = eye
            nn.init.zeros_(m.bias)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


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
        self.enc_embedding = nn.Linear(configs.input_channels, configs.d_model)

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
        batch_size, length, input_dim = x.shape

        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)

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
                pdd = torch.func.vmap(torch.func.jacfwd(self.gs[i]))(inputs)

            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


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


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False, max_min=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        self.max_min = max_min
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.max_min:
            self.mean = torch.min(x, dim=1, keepdim=True)[0].detach()
            self.stdev = (torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0]).detach()
        else:
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x