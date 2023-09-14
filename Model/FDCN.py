import numpy as np
import torch
from torch import nn
from torch.nn import init
from utils import EarlyStopping
import wandb
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
import glob
from utils.df_utils import save_data_df, save_dict
import datetime
import copy
from torch.cuda.amp import autocast, GradScaler
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
                               smart_optimizer,
                               torch_distributed_zero_first)
import math
from utils.dcn import DeformConv1D
import time


class blank_insert(nn.Module):
    def __init__(self):
        super(blank_insert, self).__init__()

    def forward(self, x):
        return x


def max_norm(model, max_val=3.0, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


class frequency_SELayer(nn.Module):
    def __init__(self, fifter_size, reduction=16):
        super(frequency_SELayer, self).__init__()
        self.transpose_1 = self.transpose_frequency2channel
        self.transpose_2 = self.transpose_channel2frequency
        self.SElayer = SELayer(fifter_size, reduction)

    def transpose_frequency2channel(self, x):
        return x.permute(0, 2, 1, 3)

    def transpose_channel2frequency(self, x):
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.transpose_1(x)
        x = self.SElayer(x)
        out = self.transpose_2(x)
        return out


class channel_SELayer(nn.Module):
    def __init__(self, channel_size, reduction=16):
        super(channel_SELayer, self).__init__()
        self.transpose_1 = self.transpose_spat2channel
        self.transpose_2 = self.transpose_channel2spat
        self.SElayer = SELayer(channel_size, reduction)

    def transpose_spat2channel(self, x):
        return x.permute(0, 3, 2, 1)

    def transpose_channel2spat(self, x):
        return x.permute(0, 3, 2, 1)

    def forward(self, x):
        x = self.transpose_1(x)
        x = self.SElayer(x)
        out = self.transpose_2(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction) + 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction) + 1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        i = y.expand_as(x)
        return x * y.expand_as(x)


class CNN2CWT(nn.Module):
    def __init__(self, in_chans, sfreq, fifter_size=None, frequency_max=38, frequency_min=8,
                 noliner=nn.ELU):
        super(CNN2CWT, self).__init__()
        # 计算卷积核大小

        kernel_size_min = sfreq // frequency_max
        kernel_size_max = sfreq // frequency_min
        if fifter_size == "auto":  # 自动计算fif的size
            if kernel_size_min % 2 == 0:
                kernel_size_min = kernel_size_min - 1
            if kernel_size_max % 2 == 0:
                kernel_size_max = kernel_size_max + 1
            fifter_size = (kernel_size_max - kernel_size_min) // 2 + 1

        kernel_size_list = np.linspace(kernel_size_min, kernel_size_max, num=fifter_size, endpoint=True,
                                       dtype=np.float16)
        kernel_num = kernel_size_list.size
        self.kernel_num = kernel_num
        k_size_list = []
        for i in range(kernel_num):
            min = math.floor(kernel_size_list[i])
            k_size_list.append(min)
            # max = math.ceil(kernel_size_list[i])
            #
            # if min % 2 != 0:
            #     k_size_list.append(min)
            # else:
            #     k_size_list.append(min + 1)

        # self.transpose_chanle_to_spat = transpose_chanle_to_spat
        self.layer_num = kernel_num
        self.CWT_layer = []
        for i in range(kernel_num):
            self.CWT_layer.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_chans, out_channels=in_chans,
                              kernel_size=(1, k_size_list[i]),
                              stride=(1, 1), padding="same", groups=in_chans),
                    noliner()
                )
            )
        self.CWT_model = nn.Sequential(*list([m for m in self.CWT_layer]))

    def forward(self, x):
        x = self.transpose_chanle_to_spat(x)
        out = [self.CWT_model[0](x)]
        for i in range(1, self.layer_num):
            out.append(self.CWT_model[i](x))
        out = torch.cat(out, dim=2)
        return out

    @staticmethod
    def transpose_chanle_to_spat(x):
        return x.permute(0, 2, 1, 3)


def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 2, 3, 1)


def transpose_time_to_spat2(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 1, 3, 2)


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


def identity(x):
    return x


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__ +
                "(expression=%s) " % expression_str
        )


class DeformConv_1Dto2D(nn.Module):
    def __init__(self, in_chan, out_channel, kernel_size):
        super(DeformConv_1Dto2D, self).__init__()
        self.in_chan = in_chan
        self.out_channel = out_channel
        self.DeformConv_list = []
        # for _ in range(in_chan):
        #     self.DeformConv_list.append(
        #         DeformConv1D(inc=1, outc=out_channel, kernel_size=kernel_size, padding=False, modulation=True,
        #                      stride=1, bias=True))
        self.DeformConv_list_model = DeformConv1D(inc=1, outc=out_channel, kernel_size=kernel_size,
                                                  padding=False,
                                                  modulation=True, stride=1, bias=True)

    def forward(self, x):
        out = [self.DeformConv_list_model(x[:, :, :, 0]).unsqueeze(dim=-1)]
        for i in range(1, self.in_chan):
            out.append(self.DeformConv_list_model(x[:, :, :, i]).unsqueeze(dim=-1))
        out = torch.cat(out, dim=-1)
        return out


def mean_out_put(x):
    x = x.squeeze()
    shape = x.shape
    outputs = torch.mean(x, dim=-1)
    outputs = outputs.squeeze()
    return outputs


def max_out_put(x):
    x = x.squeeze()
    outputs = torch.max(x, dim=-1)[0]
    outputs = outputs.squeeze()
    return outputs


def Nothing(x):
    return x


def T_Squeeze(x):
    x = x.squeeze()
    return x


class fdcn(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_window_samples=None,
                 n_filters_time=40,
                 filter_time_length=25,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_conv_length=30,
                 frequency_max=38,
                 frequency_min=4,
                 fifter_size=100,
                 frequency_SE_layer_feature=100,
                 channel_SE_layer_feature=100,
                 frequency_SE=True,
                 sfreq=250,
                 channel_SE=True,
                 conv_nonlin=square,
                 pool_mode="mean",
                 pool_nonlin=safe_log,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5,
                 lable=1,
                 last_pool="mean",
                 log=False
                 ):
        super(fdcn, self).__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.init = False
        self.lable = lable
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.last_pool = last_pool
        self.log = log
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        # layer

        if fifter_size == "auto":
            self.CWT_space = CNN2CWT(in_chans=in_chans, sfreq=sfreq, fifter_size=fifter_size,
                                     frequency_max=frequency_max,
                                     frequency_min=frequency_min)
            fifter_size = self.CWT_space.kernel_num
        else:
            self.CWT_space = CNN2CWT(in_chans=in_chans, sfreq=sfreq, fifter_size=fifter_size,
                                     frequency_max=frequency_max,
                                     frequency_min=frequency_min)
        if frequency_SE == True:
            frequency_SE_layer_reduction = fifter_size / frequency_SE_layer_feature
            self.F_SElayer = frequency_SELayer(fifter_size=fifter_size,
                                               reduction=frequency_SE_layer_reduction)
        else:
            self.F_SElayer = blank_insert()

        if channel_SE == True:
            channel_SE_layer_reduction = in_chans / channel_SE_layer_feature
            self.C_SElayer = channel_SELayer(channel_size=in_chans, reduction=channel_SE_layer_reduction)
        else:
            self.C_SElayer = blank_insert()

        self.conv_frequency = nn.Conv2d(self.in_chans, self.in_chans, kernel_size=(fifter_size, 1),
                                        stride=(fifter_size, 1), groups=self.in_chans)

        self.change_time_and_space = Expression(transpose_time_to_spat)
        self.change_time_and_space2 = Expression(transpose_time_to_spat2)
        self.conv_time_1 = DeformConv_1Dto2D(in_chan=self.in_chans, out_channel=self.n_filters_time,
                                             kernel_size=self.filter_time_length, )
        self.conv_time_2 = DeformConv_1Dto2D(in_chan=self.in_chans, out_channel=self.n_filters_time,
                                             kernel_size=self.filter_time_length, )
        self.conv_spat = nn.Conv2d(self.n_filters_time, self.n_filters_spat, (1, self.in_chans),
                                   stride=(1, 1),
                                   bias=not self.batch_norm, )
        n_filters_conv = self.n_filters_spat
        self.bnorm = nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True)
        if self.conv_nonlin == "ELU":
            self.conv_nonlin_exp = nn.ELU()
        else:
            self.conv_nonlin_exp = Expression(square)

        self.pool = pool_class(kernel_size=(self.pool_time_length, 1), stride=(1, 1), )
        if self.pool_nonlin == "ELU":
            self.pool_nonlin_exp = nn.ELU()
        else:
            self.pool_nonlin_exp = Expression(safe_log)

        self.drop = nn.Dropout2d(p=self.drop_prob)
        out = self(
            torch.from_numpy(
                np.ones((1, 1, self.in_chans, self.input_window_samples), dtype=np.float32, )))
        n_out_time = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto" or self.final_conv_length >= n_out_time:
            self.final_conv_length = n_out_time
            self.init = True
            self.output_layer = Expression(T_Squeeze)
        else:
            self.init = True
            if self.last_pool == "mean":
                self.output_layer = Expression(mean_out_put)
            if self.last_pool == "max":
                self.output_layer = Expression(max_out_put)
        self.conv_classifier = nn.Conv2d(n_filters_conv, self.n_classes, (self.final_conv_length, 1),
                                         dilation=(self.pool_time_stride, 1),
                                         bias=True)
        if self.log:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = Expression(Nothing)
        # init
        self.output_layer = Expression(T_Squeeze)
        # torch.nn.init.kaiming_normal_(self.conv_time.weight)
        # maybe no bias in case of no split layer and batch norm
        # if self.split_first_layer or (not self.batch_norm):
        #     init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            torch.nn.init.kaiming_normal_(self.conv_spat.weight)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        torch.nn.init.kaiming_normal_(self.conv_classifier.weight)
        init.constant_(self.conv_classifier.bias, 0)

    def forward(self, x):
        max_norm(self.CWT_space, max_val=2.0)
        x_f = self.CWT_space(x)
        x_f = self.F_SElayer(x_f)
        max_norm(self.conv_frequency, max_val=1.0)
        x_f = self.conv_frequency(x_f)
        x_f = self.change_time_and_space(x_f)
        x_t = self.change_time_and_space2(x)
        max_norm(self.conv_time_1, max_val=2.0)
        max_norm(self.conv_time_2, max_val=2.0)
        x_t = self.conv_time_1(x_t)
        x_f = self.conv_time_2(x_f)
        max_norm(self.conv_spat, max_val=0.5)
        x_o = self.conv_spat(x_t + x_f)
        x_o = self.bnorm(x_o)
        x_o = self.conv_nonlin_exp(x_o)
        x_o = self.pool(x_o)
        x_o = self.pool_nonlin_exp(x_o)
        x_o = self.drop(x_o)
        if self.init == False:
            return x_o
        max_norm(self.conv_classifier, max_val=0.5)
        x_c = self.conv_classifier(x_o)
        x_c = self.softmax(x_c)
        out = self.output_layer(x_c)

        return out

