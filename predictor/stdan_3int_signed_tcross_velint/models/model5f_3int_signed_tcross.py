from __future__ import annotations

import math

import torch as t
import torch.nn.functional as F
from einops import repeat
from torch import nn


class GDEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.n_head = args["n_head"]
        self.att_out = args["att_out"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.f_length = args["f_length"]
        self.relu_param = args["relu"]
        self.traj_linear_hidden = args["traj_linear_hidden"]
        self.use_maneuvers = args["use_maneuvers"]
        self.use_elu = args["use_elu"]
        self.use_spatial = args["use_spatial"]
        self.dropout = args["dropout"]

        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)

        self.qff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.first_glu = GLU(self.n_head * self.att_out, self.lstm_encoder_size, self.dropout)
        self.second_glu = GLU(self.n_head * self.att_out, self.lstm_encoder_size, self.dropout)

        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)

    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):
        if self.f_length == 5:
            hist = t.cat((hist, cls, va), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva), -1)
        elif self.f_length == 6:
            hist = t.cat((hist, cls, va, lane), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva, nbrslane), -1)

        hist_enc = self.activation(self.linear1(hist))
        hist_hidden_enc, _ = self.lstm(hist_enc)
        hist_hidden_enc = hist_hidden_enc.permute(1, 0, 2)

        nbrs_enc = self.activation(self.linear1(nbrs))
        nbrs_hidden_enc, _ = self.lstm(nbrs_enc)
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, "b g s -> t b g s", t=self.in_length)
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)

        query = self.qff(hist_hidden_enc)
        _, _, embed_size = query.shape
        query = t.cat(t.split(t.unsqueeze(query, 2), int(embed_size / self.n_head), -1), 1)
        keys = t.cat(t.split(self.kff(soc_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values = t.cat(t.split(self.vff(soc_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 2, 3)
        att = t.matmul(query, keys) / math.sqrt(self.lstm_encoder_size)
        att = t.softmax(att, -1)
        values = t.matmul(att, values)
        values = t.cat(t.split(values, int(hist.shape[0]), 1), -1).squeeze(2)

        spa_values, _ = self.first_glu(values)
        spa_values = self.addAndNorm(hist_hidden_enc, spa_values)

        qt = t.cat(t.split(self.qt(spa_values), int(embed_size / self.n_head), -1), 0)
        kt = t.cat(t.split(self.kt(spa_values), int(embed_size / self.n_head), -1), 0).permute(0, 2, 1)
        vt = t.cat(t.split(self.vt(spa_values), int(embed_size / self.n_head), -1), 0)
        att = t.matmul(qt, kt) / math.sqrt(self.lstm_encoder_size)
        att = t.softmax(att, -1)
        values = t.matmul(att, vt)
        values = t.cat(t.split(values, int(hist.shape[1]), 0), -1)
        time_values, _ = self.second_glu(values)

        if self.use_spatial:
            values = self.addAndNorm(hist_hidden_enc, spa_values, time_values)
        else:
            values = self.addAndNorm(hist_hidden_enc, time_values)
        return values


def outputActivation(x):
    mu_x = x[:, :, 0:1]
    mu_y = x[:, :, 1:2]
    sig_x = t.exp(x[:, :, 2:3])
    sig_y = t.exp(x[:, :, 3:4])
    rho = t.tanh(x[:, :, 4:5])
    return t.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=2)


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            return self.normalize(t.add(t.add(x1, x2), x3))
        return self.normalize(t.add(x1, x2))


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.relu_param = args["relu"]
        self.use_elu = args["use_elu"]
        self.use_maneuvers = args["use_maneuvers"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.encoder_size = args["lstm_encoder_size"]
        self.device = args["device"]
        self.cat_pred = args["cat_pred"]
        self.use_mse = args["use_mse"]
        self.num_intentions = args["num_intentions"]
        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)

        self.lstm = nn.LSTM(self.encoder_size, self.encoder_size)
        self.linear1 = nn.Linear(self.encoder_size, 2 if self.use_mse else 5)
        self.dec_linear = nn.Linear(self.encoder_size + self.num_intentions, self.encoder_size)

    def forward(self, dec, intent_enc):
        if self.use_maneuvers or self.cat_pred:
            intent_enc = intent_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = self.dec_linear(t.cat((dec, intent_enc), -1))
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)
        return fut_pred if self.use_mse else outputActivation(fut_pred)


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.n_head = args["n_head"]
        self.att_out = args["att_out"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.relu_param = args["relu"]
        self.train_flag = args["train_flag"]
        self.use_maneuvers = args["use_maneuvers"]
        self.num_intentions = args["num_intentions"]
        self.use_elu = args["use_elu"]
        self.use_true_man = args["use_true_man"]

        self.Decoder = Decoder(args=args)
        self.mu_fc1 = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.mu_fc = nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.op_intent = nn.Linear(self.lstm_encoder_size, self.num_intentions)
        self.t_cross_head = nn.Linear(self.lstm_encoder_size, 1)

        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)
        self.normalize = nn.LayerNorm(self.lstm_encoder_size)
        self.mapping = nn.Parameter(t.Tensor(self.in_length, self.out_length, self.num_intentions))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)

    def forward(self, values, intent_enc):
        maneuver_state = values[:, -1, :]
        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        intent_pred = F.softmax(self.op_intent(maneuver_state), dim=-1)
        t_cross_pred = t.tanh(self.t_cross_head(maneuver_state))

        if self.train_flag:
            if self.use_true_man:
                selected_intent = intent_enc
            else:
                intent_idx = t.argmax(intent_pred, dim=-1).detach().unsqueeze(1)
                selected_intent = t.zeros_like(intent_pred).scatter_(1, intent_idx, 1)
            index = selected_intent.permute(-1, 0)
            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)
            decoder_intent = intent_enc if self.use_maneuvers else intent_pred
            fut_pred = self.Decoder(dec, decoder_intent)
            return fut_pred, intent_pred, t_cross_pred

        out = []
        for i in range(self.num_intentions):
            intent_tmp = t.zeros_like(intent_enc)
            intent_tmp[:, i] = 1
            index = intent_tmp.permute(-1, 0)
            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)
            out.append(self.Decoder(dec, intent_tmp))
        return out, intent_pred, t_cross_pred


class GLU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated
