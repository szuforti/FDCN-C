from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
from prefetch_generator import BackgroundGenerator
import numpy as np
import scipy.io
from scipy import signal
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import os
import gc
import glob
import scipy.io as scio
# from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
import os
from PIL import Image
import numpy as np
import glob
import torch
import random
import gc
from multiprocessing import Pool
import time
from io import BytesIO
import io
import copy
import yaml
from utils import butter_bandpass_filter, butter_lowpass_filter
from preprocessing.config import CONSTANT
from preprocessing.HGD import raw
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

CONSTANT = CONSTANT['HGD']
raw_path = CONSTANT['raw_path']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

# eventDescription = {  # '276': "eyesOpen", '277': "eyesClosed", '768': "startTrail",
#     '769': "CueOnsetLeft", '770': "CueOnsetRight", '771': "CueOnsetFoot",
#     '772': "CueOnsetTongue", '783': "cueUnknown",
#
#     # '1023': "rejected", '1072': 'EyeMovements','32766':"startNewRun"
# }

label_name = ['Feet', 'Left Hand', 'Rest', 'Right Hand']  # 标签
# 把数据处理放在分割完数据集之后，不然会导致数据集的输入不一样，
# 反思、。。。。
label_int = [1, 2, 3, 4]

label_dict = {}
for idx, name in enumerate(label_int):
    label_dict[name] = idx

min_max_scaler = preprocessing.MinMaxScaler()  # 默认为范围0~1，拷贝操作
abs_max_scaler = preprocessing.MaxAbsScaler()  # 绝对值归一化
from sklearn.preprocessing import StandardScaler


# scale = StandardScaler()


class filter_fun(object):
    def __init__(self, filter_type, frequency_min, frequency_max, freq, order):
        super(filter_fun, self).__init__()
        self.filter_type = filter_type
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.freq = freq
        self.order = order

    def __call__(self, data):
        out_put = copy.deepcopy(data)
        if self.filter_type == "bandpass":
            out_put = butter_bandpass_filter(data=data, lowcut=self.frequency_min,
                                             highcut=self.frequency_max,
                                             order=self.order, fs=self.freq)
        if self.filter_type == "lowpass":
            out_put = butter_lowpass_filter(data=data, lowcut=self.frequency_max, fs=self.freq,
                                            order=self.order)
        return out_put


class scale_fun(object):
    def __init__(self, mean, std):
        super(scale_fun, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        out_put = copy.deepcopy((data - np.expand_dims(np.array(self.mean), axis=-1)) / np.expand_dims(
            np.array(self.std), axis=-1))
        return out_put


class vertical_inversion_fun(object):
    def __init__(self, rate):
        super(vertical_inversion_fun, self).__init__()
        self.rate = rate

    def __call__(self, data):
        random_i = random.random()
        if random_i < self.rate:
            plt.plot(data[0, :])
            plt.show()
            out_put = np.flip(data, axis=2)
            plt.plot(out_put[0, :])
            plt.show()
        else:
            out_put = data
        return out_put


class vertical_Negative_fun(object):
    def __init__(self, rate):
        super(vertical_Negative_fun, self).__init__()
        self.rate = rate

    def __call__(self, data):
        random_i = random.random()
        if random_i < self.rate:
            # plt.plot(data[0, :])
            # plt.show()
            out_put = -1 * data
            # plt.plot(out_put[0, :])
            # plt.show()
        else:
            out_put = data
        return out_put


class add_noise_fun(object):
    def __init__(self, rate, snr):
        self.rate = rate
        self.snr = snr

    def __call__(self, data):
        random_i = random.random()
        out_put = copy.deepcopy(data)
        if random_i < self.rate:
            [channel_num, data_num] = data.shape
            for channel in range(channel_num):
                Ps = np.sum(abs(data[channel, :]) ** 2) / len(data[channel, :])
                Pn = Ps / (10 ** ((self.snr / 10)))
                noise = np.random.normal(size=len(data[channel, :])) * np.sqrt(Pn)
                out_put[channel, :] = out_put[channel, :] + noise

        return out_put


class cut_cat_fun(object):
    def __init__(self, rate, cut_cat_time, data, label):
        super(cut_cat_fun, self).__init__()
        self.rate = rate
        self.cut_cat_time = cut_cat_time
        self.data = data
        self.label = label
        self.data_len = self.data.shape[-1]

    def __call__(self, data, label, index):
        random_i = random.random()
        if random_i < self.rate:
            # 先确定换的片段
            use_data_index = [i for i, j in enumerate(self.label) if j == label and i != index]
            be_change_index = random.choice(use_data_index)
            # 然后选择
            input_time = random.randint(0, self.data_len - self.cut_cat_time)
            # 进行替换
            be_change_data = self.data[be_change_index]
            data[:, input_time:input_time + self.cut_cat_time] = be_change_data[:,
                                                                 input_time:input_time + self.cut_cat_time]
        return data


from utils.Augmented import ft_surrogate, gaussian_noise, smooth_time_mask


class ft_surrogate_fun(object):
    def __init__(self, rate, random_state, phase_noise_magnitude, channel_indep):
        super(ft_surrogate_fun, self).__init__()
        self.rate = rate
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.rng = check_random_state(random_state)

    def __call__(self, data, label):
        random_i = random.random()
        if random_i < self.rate:
            # plt.plot(data[0,0,:].numpy())
            # plt.show()
            Out_X, out_y = ft_surrogate(X=data, y=label,
                                        phase_noise_magnitude=self.phase_noise_magnitude,
                                        channel_indep=self.channel_indep,
                                        random_state=self.rng,
                                        )
            out_data = Out_X
            # plt.plot(out_data[0,0,:].numpy())
            # plt.show()
        else:
            out_data = data
        return out_data


class gaussian_noise_fun(object):
    def __init__(self, rate, random_state, std):
        super(gaussian_noise_fun, self).__init__()
        self.rate = rate
        self.std = std
        self.rng = check_random_state(random_state)

    def __call__(self, data, label):
        random_i = random.random()
        if random_i < self.rate:
            # plt.plot(data[0,0,:].numpy())
            # plt.show()
            transformed_X, y = gaussian_noise(X=data, y=label, std=self.std, random_state=self.rng)
            out_data = transformed_X
            # plt.plot(out_data[0,0,:].numpy())
            # plt.show()
        else:
            out_data = data
        return out_data


class smooth_time_mask_fun(object):
    def __init__(self, rate, random_state, mask_len_samples):
        super(smooth_time_mask_fun, self).__init__()
        self.rate = rate
        self.mask_len_samples = mask_len_samples
        self.rng = check_random_state(random_state)

    def __call__(self, data, label):
        random_i = random.random()
        if random_i < self.rate:
            seq_length = data.shape[2]
            mask_len_samples = self.mask_len_samples
            mask_start = torch.as_tensor(self.rng.uniform(
                low=0, high=1, size=1,
            ), device=data.device) * (seq_length - mask_len_samples)
            # plt.plot(data[0,0,:].numpy())
            # plt.show()
            Out_X, y = smooth_time_mask(X=data, y=label, mask_start_per_sample=mask_start,
                                        mask_len_samples=mask_len_samples)
            # plt.plot(Out_X[0,0,:].numpy())
            # plt.show()
        else:
            Out_X = data
        return Out_X


class DataLoaderX(DataLoader):  # 加速用的

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def default_loader(data):
    output = torch.unsqueeze(torch.from_numpy(data), dim=0)
    # output = torch.from_numpy(data)
    return output


def load_HGD(PATH, subjects, new_smp_freq, num_class, id_chosen_chs, start=0, stop=4):
    # start = CONSTANT['MI']['start']  # 2
    # stop = CONSTANT['MI']['stop']  # 6
    X_train, y_tr, X_test, y_te = raw.load_crop_data(
        PATH=PATH, subjects=subjects, start=start, stop=stop, new_smp_freq=new_smp_freq,
        id_chosen_chs=id_chosen_chs)
    return X_train, y_tr, X_test, y_te


class sour_data_with_FastICA():
    def __init__(self,
                 source_list,
                 start_time=0,
                 time=4,
                 freq=250,
                 subjects=[1],
                 frequency_max=38,
                 frequency_min=8,
                 sel_chs=None,
                 bands_fif=True,
                 scale=True,
                 filter_type="bandpass",
                 order=5):
        sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
        # id_chosen_chs = raw.chanel_selection(sel_chs)
        self.start_time = start_time
        self.end_time = start_time + time
        self.start_point = int(start_time * freq)
        self.end_point = int(self.end_time * freq)
        self.frequency_max = frequency_max
        self.frequency_min = frequency_min
        self.order = order
        self.freq = freq
        num_class = 4
        self.subjects = subjects
        self.sel_chs = sel_chs
        self.data_name = 'HDG'

        X_train_all, y_train_all, X_test_all, y_test_all = load_HGD(source_list, subjects, freq,
                                                                    num_class,
                                                                    sel_chs, start=start_time,
                                                                    stop=time + start_time)
        # 时间选取：
        [num, chanel, time] = X_train_all.shape
        if self.end_point > time:
            print(f"数据长度不够，现在最大的数据点为{time},原本需要长度为{self.end_point}")
            self.end_point = time

        # self.X_train_all = X_train_all[:, :, self.start_point:self.end_point].squeeze()
        self.X_train_all = X_train_all.squeeze()
        self.y_train_all = y_train_all.squeeze()
        # self.X_test_all = X_test_all[:, :, self.start_point:self.end_point].squeeze()
        self.X_test_all = X_test_all.squeeze()
        self.y_test_all = y_test_all.squeeze()
        [num, channel_num, time] = X_train_all.shape
        self.num = num
        mean = []
        std = []
        if bands_fif:
            self.filter = filter_fun(filter_type=filter_type, frequency_min=frequency_min,
                                     frequency_max=frequency_max, freq=freq, order=order)
        self.scale = scale
        if scale:
            for channel in range(channel_num):
                mean_tmp = np.empty(shape=(0, 1))
                std_tmp = np.empty(shape=(0, 1))
                # data_tmp = np.empty(shape=(0, 1))
                for time in range(num):
                    use_data = X_train_all[time, channel, :].copy()
                    use_data = self.filter(use_data)  # 进行带通滤波 7-30Hz 8阶滤波器
                    mean_tmp = np.append(mean_tmp, np.mean(use_data))
                    std_tmp = np.append(std_tmp, np.std(use_data))
                    # data_tmp = np.append(data_tmp, use_data)
                mean.append(float(np.mean(mean_tmp)))
                std.append(float(np.mean(std_tmp)))
        # mean.append(float(np.mean(data_tmp)))
        # std.append(float(np.std(data_tmp)))

        mean_std = {"mean": mean, "std": std}
        self.mean_std = mean_std


class KZDDataset_pic(Dataset):

    def __init__(self, source_data, ki=0, K=0, type='train',
                 loader=default_loader,
                 output_size=1000,
                 bands_fif=True,
                 filter_type="bandpass",
                 freq=250,
                 frequency_max=38,
                 frequency_min=8,
                 order=8,
                 scale=True,
                 windows_size=100,
                 windows_overlapping=20,
                 shallow_windows=False,
                 windows_keeplast=False,
                 vertical_inversion=False,
                 vertical_inversion_rate=0.2,
                 add_noise=True,
                 add_noise_rate=0.2,
                 res=0.01,
                 cut_cat=True,
                 cut_cat_rate=0.3,
                 cut_cat_times=100,
                 load_data=False,
                 data_path=None,
                 label_path=None,

                 ft_furrogate=True,
                 ft_furrogate_rate=0.5,
                 ft_furrogate_phase_noise_magnitude=0.5,
                 ft_furrogate_channel_indep=False,

                 gaussian_noise=True,
                 gaussian_noise_rate=0.5,
                 gaussian_noise_std=0.1,

                 smooth_time_mask=True,
                 smooth_time_mask_rate=0.5,
                 smooth_time_mask_len_samples=200,

                 ):

        super(KZDDataset_pic, self).__init__()
        # 原始数据
        X_train_all = source_data.X_train_all
        y_train_all = source_data.y_train_all
        X_test_all = source_data.X_test_all
        y_test_all = source_data.y_test_all

        self.frequency_max = source_data.frequency_max
        self.frequency_min = source_data.frequency_min
        self.order = source_data.order
        self.freq = source_data.freq
        self.subjects = source_data.subjects
        self.sel_chs = source_data.sel_chs
        self.mean_std = source_data.mean_std
        self.type = type
        self.output_size = output_size  # 输出数据的大小
        self.shallow_windows = shallow_windows
        self.loader = loader

        #################
        if load_data:
            assert data_path is not None
            assert label_path is not None
            self.data = np.load(data_path)
            self.label = np.load(label_path)
            print("please check the data and lable set", "\ndata:", self.data.shape, "\nlabel",
                  self.label.shape)
            self.bands_fif = bands_fif
            if bands_fif:
                self.filter = filter_fun(filter_type=filter_type, frequency_min=frequency_min,
                                         frequency_max=frequency_max, freq=freq, order=order)
            self.scale = scale
            if scale:
                self.normalizer = scale_fun(source_data.mean_std["mean"], source_data.mean_std["std"])

            self.add_noise = add_noise
            if add_noise:
                self.noise = add_noise_fun(add_noise_rate, res)

            return
            ######################

        if K != 0 and (type == "T" or type == "val"):
            skf = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)
            for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all)):
                if fold == ki:
                    break
        if type == 'val':
            choose_index = val_index
            self.data = X_train_all[val_index]
            self.label = y_train_all[val_index]

            print(type + ":")
            print("index_ID:", end=" ")
            for im in choose_index:
                print(im, end=" ")
            print("")
        # print(every_z_len * ki,every_z_len * (ki + 1))
        elif type == 'T':
            print(type + ":")
            choose_index = train_index
            self.data = X_train_all[train_index]
            self.label = y_train_all[train_index]
            print("index_ID:", end=" ")
            for im in choose_index:
                print(im, end=" ")
            print("")
        elif type == "T_all":
            choose_index = range(X_train_all.shape[0])
            self.data = X_train_all
            self.label = y_train_all
            print("index_ID:", end=" ")
            for im in choose_index:
                print(im, end=" ")
            print("")
        elif type == "E":
            choose_index = range(X_test_all.shape[0])
            self.data = X_test_all
            self.label = y_test_all
            print("index_ID:", end=" ")
            for im in choose_index:
                print(im, end=" ")
            print("")
        elif type == "all":
            choose_index = range(X_test_all.shape[0] + X_train_all.shape[0])
            self.data = np.concatenate((X_train_all, X_test_all), axis=0)
            self.label = np.concatenate((y_train_all, y_test_all), axis=0)
            print("index_ID:", end=" ")
            for im in choose_index:
                print(im, end=" ")
            print("")
        print("choose_num:", len(choose_index))
        # 获得数据的上一层
        # 进行数据增强
        if shallow_windows:
            [epoch_num, channel_num, data_num] = self.data.shape
            moving_size = windows_size - windows_overlapping
            shallow_windows_num = ((data_num - windows_size) // moving_size) + 1
            data_mu_size = shallow_windows_num
            last_size = data_num - shallow_windows_num * moving_size
            if last_size > moving_size // 4 and last_size > 0 and windows_keeplast == True:
                data_mu_size = shallow_windows_num + 1
            epoch_num = epoch_num * data_mu_size
            new_data = np.empty(shape=(epoch_num, channel_num, self.output_size))
            new_label = np.empty(shape=(epoch_num,))
            i = 0
            for data_index in range(len(choose_index)):
                windows_start = 0
                windows_end = windows_size
                tmp_label = int(self.label[data_index])  # 标签
                for iter_i in range(shallow_windows_num):
                    tmp_epoch_data = self.data[data_index, :, windows_start:windows_end]  # 数据
                    new_data[i, :, :] = tmp_epoch_data
                    new_label[i] = tmp_label
                    i = i + 1
                    windows_start += moving_size
                    windows_end += moving_size
                if windows_keeplast:  # 要保留最后一个
                    last_size = data_num - shallow_windows_num * moving_size
                    if last_size > moving_size // 4 and last_size > 0:  # 大于10个数据点才会刘希
                        tmp_epoch_data = self.data[data_index, :, -1 * windows_size - 1:-1]
                        new_data[i, :, :] = tmp_epoch_data
                        new_label[i] = tmp_label
                        i = i + 1
            self.data = new_data
            self.label = new_label

        self.vertical_inversion = vertical_inversion
        if vertical_inversion:
            self.vertical_inversion_fun = vertical_Negative_fun(vertical_inversion_rate)
        self.cut_cat = cut_cat
        self.cut_cat_rate = cut_cat_rate
        self.cut_cat_times = cut_cat_times
        if cut_cat:
            self.cut_cat_fun = cut_cat_fun(cut_cat_rate, cut_cat_times, self.data, self.label)

        print("please check the data and lable set", "\ndata:", self.data.shape, "\nlabel",
              self.label.shape)
        self.bands_fif = bands_fif
        if bands_fif:
            self.filter = filter_fun(filter_type=filter_type, frequency_min=frequency_min,
                                     frequency_max=frequency_max, freq=freq, order=order)
        self.scale = scale
        if scale:
            self.normalizer = scale_fun(source_data.mean_std["mean"], source_data.mean_std["std"])

        self.add_noise = add_noise
        if add_noise:
            self.noise = add_noise_fun(add_noise_rate, res)

        self.ft_furrogate = ft_furrogate
        if ft_furrogate:
            self.ft_furrogate_fun = ft_surrogate_fun(rate=ft_furrogate_rate, random_state=20222202,
                                                     phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                                     channel_indep=ft_furrogate_channel_indep)
        self.smooth_time_mask = smooth_time_mask
        if smooth_time_mask:
            self.smooth_time_mask_fun = smooth_time_mask_fun(rate=smooth_time_mask_rate,
                                                             random_state=20222202,
                                                             mask_len_samples=smooth_time_mask_len_samples)

        self.gaussian_noise = gaussian_noise
        if gaussian_noise:
            self.gaussian_noise_fun = gaussian_noise_fun(rate=gaussian_noise_rate,
                                                         std=gaussian_noise_std, random_state=20222202)
        # 均值放在后面去求

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index][:, 0:self.output_size])
        if self.cut_cat:
            data = self.cut_cat_fun(data=data, index=index, label=self.label[index])
        if self.vertical_inversion:
            data = self.vertical_inversion_fun(data)
        if self.bands_fif:
            data = self.filter(data)
        if self.scale:
            data = self.normalizer(data)
        if self.add_noise:
            data = self.noise(data)

        output_data = self.loader(data)

        # 有一些要在tensor上进行数据增强的
        im_label = copy.deepcopy(self.label[index])
        if self.ft_furrogate:
            output_data = self.ft_furrogate_fun(output_data, im_label)
        if self.smooth_time_mask:
            output_data = self.smooth_time_mask_fun(output_data, im_label)
        if self.gaussian_noise:
            output_data = self.gaussian_noise_fun(output_data, im_label)
        return output_data, im_label

    def __len__(self):
        [epoch_data, channel, time] = self.data.shape
        return epoch_data

    def save_data(self, path):
        data_type = self.type
        subject_name = "subject_" + "_".join(str(ii) for ii in self.subjects)
        if_shallow_windows = "shallow_windows" if self.shallow_windows else ""
        if_cut_cat = "cut_cat" if self.cut_cat else ""
        data_name = data_type + subject_name + if_shallow_windows + if_cut_cat
        save_data_name = path + "\\" + data_name + "_data.npy"
        save_label_name = path + "\\" + data_name + "_label.npy"
        np.save(save_data_name, self.data)
        np.save(save_label_name, self.label)


class HGD_dataset():
    def __init__(self, data_list, subjects, start_time=0.0, time=4, sel_chs=None,
                 shallow_windows=False, windows_overlapping=1000, windows_keeplast=False,
                 output_size=1000,
                 bands_fif=True, freq=250, frequency_max=38, frequency_min=8, filter_type="bandpass",
                 order=8,
                 scale=True,
                 vertical_inversion=False, vertical_inversion_rate=0.2,
                 add_noise=False, add_noise_rate=0.2, res=1000,
                 cut_cat=True, cut_cat_times=100, cut_cat_rate=0.2,
                 ft_furrogate=True, ft_furrogate_rate=0.5, ft_furrogate_phase_noise_magnitude=0.5,
                 ft_furrogate_channel_indep=False,
                 gaussian_noise=True, gaussian_noise_rate=0.5, gaussian_noise_std=0.1,
                 smooth_time_mask=True, smooth_time_mask_rate=0.5, smooth_time_mask_len_samples=200,
                 ):

        self.data_list = data_list
        self.subjects = subjects
        self.start_time = start_time
        self.time = time
        self.shallow_windows = shallow_windows
        self.windows_overlapping = windows_overlapping
        self.windows_keeplast = windows_keeplast
        self.output_size = output_size
        self.windows_size = output_size
        self.vertical_inversion = vertical_inversion
        self.add_noise = add_noise
        self.add_noise_rate = add_noise_rate
        self.res = res
        self.filter_type = filter_type
        self.frequency_max = frequency_max
        self.frequency_min = frequency_min
        self.freq = freq
        self.order = order
        self.sel_chs = sel_chs
        self.bands_fif = bands_fif
        self.scale = scale
        self.cut_cat_rate = cut_cat_rate
        self.cut_cat = cut_cat
        self.vertical_inversion_rate = vertical_inversion_rate
        self.cut_cat_times = cut_cat_times
        self.ft_furrogate = ft_furrogate
        self.ft_furrogate_rate = ft_furrogate_rate
        self.ft_furrogate_phase_noise_magnitude = ft_furrogate_phase_noise_magnitude
        self.ft_furrogate_channel_indep = ft_furrogate_channel_indep
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_rate = gaussian_noise_rate
        self.gaussian_noise_std = gaussian_noise_std
        self.smooth_time_mask = smooth_time_mask
        self.smooth_time_mask_rate = smooth_time_mask_rate
        self.smooth_time_mask_len_samples = smooth_time_mask_len_samples

        self.sourcedata = sour_data_with_FastICA(data_list, subjects=subjects, start_time=start_time,
                                                 time=time,
                                                 freq=freq, frequency_max=frequency_max,
                                                 filter_type=self.filter_type,
                                                 frequency_min=frequency_min, sel_chs=self.sel_chs, )

    def train_loader(self, data_augmentation=True, ki=2, K=5, batch_size=64, drop_last=True,
                     save_data=False,
                     save_data_path=None, num_workers=0,
                     load_data=False, data_path=None, label_path=None, ):
        self.K = K
        self.Ki = ki
        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = None
            windows_overlapping = None
            windows_keeplast = None
            vertical_inversion = False
            add_noise = False
            res = None
            cut_cat = False
            cut_cat_times = None
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate

        train_dataset = KZDDataset_pic(self.sourcedata, ki=self.Ki, K=self.K, type='T',
                                       shallow_windows=shallow_windows,
                                       windows_size=windows_size,
                                       windows_overlapping=windows_overlapping,
                                       windows_keeplast=windows_keeplast,
                                       output_size=output_size,

                                       vertical_inversion=vertical_inversion,
                                       vertical_inversion_rate=vertical_inversion_rate,

                                       add_noise=add_noise,
                                       res=res,
                                       add_noise_rate=add_noise_rate,

                                       cut_cat=cut_cat,
                                       cut_cat_rate=cut_cat_rate,
                                       cut_cat_times=cut_cat_times,

                                       bands_fif=bands_fif,
                                       filter_type=filter_type,
                                       freq=freq,
                                       frequency_max=frequency_max,
                                       frequency_min=frequency_min,
                                       order=order,

                                       scale=scale,

                                       load_data=load_data,
                                       data_path=data_path,
                                       label_path=label_path,

                                       ft_furrogate=ft_furrogate,
                                       ft_furrogate_rate=ft_furrogate_rate,
                                       ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                       ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                       gaussian_noise=gaussian_noise,
                                       gaussian_noise_rate=gaussian_noise_rate,
                                       gaussian_noise_std=gaussian_noise_std,
                                       smooth_time_mask=smooth_time_mask,
                                       smooth_time_mask_rate=smooth_time_mask_rate,
                                       smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                       )  # 测试机
        if save_data:
            train_dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
            prefetch_factor = 5
        else:
            persistent_workers = False
            prefetch_factor = 2

        loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor)
        return loader

    def train_all_loader(self, data_augmentation=True, batch_size=64, drop_last=True, save_data=False,
                         save_data_path=None, num_workers=0,
                         load_data=False, data_path=None, label_path=None, ):
        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = False
            windows_overlapping = False
            windows_keeplast = False
            vertical_inversion = False
            add_noise = False
            res = False
            cut_cat = False
            cut_cat_times = False
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate

        train_dataset = KZDDataset_pic(self.sourcedata, type='T_all',
                                       shallow_windows=shallow_windows,
                                       windows_size=windows_size,
                                       windows_overlapping=windows_overlapping,
                                       windows_keeplast=windows_keeplast,
                                       output_size=output_size,

                                       vertical_inversion=vertical_inversion,
                                       vertical_inversion_rate=vertical_inversion_rate,

                                       add_noise=add_noise,
                                       res=res,
                                       add_noise_rate=add_noise_rate,

                                       cut_cat=cut_cat,
                                       cut_cat_rate=cut_cat_rate,
                                       cut_cat_times=cut_cat_times,

                                       bands_fif=bands_fif,
                                       filter_type=filter_type,
                                       freq=freq,
                                       frequency_max=frequency_max,
                                       frequency_min=frequency_min,
                                       order=order,

                                       scale=scale,

                                       load_data=load_data,
                                       data_path=data_path,
                                       label_path=label_path,
                                       ft_furrogate=ft_furrogate,
                                       ft_furrogate_rate=ft_furrogate_rate,
                                       ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                       ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                       gaussian_noise=gaussian_noise,
                                       gaussian_noise_rate=gaussian_noise_rate,
                                       gaussian_noise_std=gaussian_noise_std,
                                       smooth_time_mask=smooth_time_mask,
                                       smooth_time_mask_rate=smooth_time_mask_rate,
                                       smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                       )  # 测试机
        if save_data:
            train_dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=2)
        return loader

    def all_loader(self, data_augmentation=True, batch_size=64, drop_last=True, save_data=False,
                   save_data_path=None, num_workers=0,
                   load_data=False, data_path=None, label_path=None, ):
        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = None
            windows_overlapping = None
            windows_keeplast = None
            vertical_inversion = False
            add_noise = False
            res = None
            cut_cat = False
            cut_cat_times = None
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate

        train_dataset = KZDDataset_pic(self.sourcedata, type='all',
                                       shallow_windows=shallow_windows,
                                       windows_size=windows_size,
                                       windows_overlapping=windows_overlapping,
                                       windows_keeplast=windows_keeplast,
                                       output_size=output_size,

                                       vertical_inversion=vertical_inversion,
                                       vertical_inversion_rate=vertical_inversion_rate,

                                       add_noise=add_noise,
                                       res=res,
                                       add_noise_rate=add_noise_rate,

                                       cut_cat=cut_cat,
                                       cut_cat_rate=cut_cat_rate,
                                       cut_cat_times=cut_cat_times,

                                       bands_fif=bands_fif,
                                       filter_type=filter_type,
                                       freq=freq,
                                       frequency_max=frequency_max,
                                       frequency_min=frequency_min,
                                       order=order,

                                       scale=scale,

                                       load_data=load_data,
                                       data_path=data_path,
                                       label_path=label_path,

                                       ft_furrogate=ft_furrogate,
                                       ft_furrogate_rate=ft_furrogate_rate,
                                       ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                       ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                       gaussian_noise=gaussian_noise,
                                       gaussian_noise_rate=gaussian_noise_rate,
                                       gaussian_noise_std=gaussian_noise_std,
                                       smooth_time_mask=smooth_time_mask,
                                       smooth_time_mask_rate=smooth_time_mask_rate,
                                       smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                       )  # 测试机
        if save_data:
            train_dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=2)
        return loader

    def train_all_loader(self, data_augmentation=True, batch_size=64, drop_last=True, save_data=False,
                         save_data_path=None, num_workers=0,
                         load_data=False, data_path=None, label_path=None, ):
        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = None
            windows_overlapping = None
            windows_keeplast = None
            vertical_inversion = False
            add_noise = False
            res = None
            cut_cat = False
            cut_cat_times = None
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate
        train_dataset = KZDDataset_pic(self.sourcedata, type='T_all',
                                       shallow_windows=shallow_windows,
                                       windows_size=windows_size,
                                       windows_overlapping=windows_overlapping,
                                       windows_keeplast=windows_keeplast,
                                       output_size=output_size,

                                       vertical_inversion=vertical_inversion,
                                       vertical_inversion_rate=vertical_inversion_rate,

                                       add_noise=add_noise,
                                       res=res,
                                       add_noise_rate=add_noise_rate,

                                       cut_cat=cut_cat,
                                       cut_cat_rate=cut_cat_rate,
                                       cut_cat_times=cut_cat_times,

                                       bands_fif=bands_fif,
                                       filter_type=filter_type,
                                       freq=freq,
                                       frequency_max=frequency_max,
                                       frequency_min=frequency_min,
                                       order=order,

                                       scale=scale,

                                       load_data=load_data,
                                       data_path=data_path,
                                       label_path=label_path,
                                       ft_furrogate=ft_furrogate,
                                       ft_furrogate_rate=ft_furrogate_rate,
                                       ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                       ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                       gaussian_noise=gaussian_noise,
                                       gaussian_noise_rate=gaussian_noise_rate,
                                       gaussian_noise_std=gaussian_noise_std,
                                       smooth_time_mask=smooth_time_mask,
                                       smooth_time_mask_rate=smooth_time_mask_rate,
                                       smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                       )  # 测试机
        if save_data:
            train_dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=2)
        return loader

    def val_loader(self, data_augmentation=True, ki=2, K=5, batch_size=64, drop_last=True,
                   save_data=False,
                   save_data_path=None, num_workers=0,
                   load_data=False, data_path=None, label_path=None, ):
        self.K = K
        self.Ki = ki

        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = None
            windows_overlapping = None
            windows_keeplast = None
            vertical_inversion = False
            add_noise = False
            res = None
            cut_cat = False
            cut_cat_times = None
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate
        dataset = KZDDataset_pic(self.sourcedata, ki=self.Ki, K=self.K, type='val',
                                 shallow_windows=shallow_windows,
                                 windows_size=windows_size,
                                 windows_overlapping=windows_overlapping,
                                 windows_keeplast=windows_keeplast,
                                 output_size=output_size,

                                 vertical_inversion=vertical_inversion,
                                 vertical_inversion_rate=vertical_inversion_rate,

                                 add_noise=add_noise,
                                 res=res,
                                 add_noise_rate=add_noise_rate,

                                 cut_cat=cut_cat,
                                 cut_cat_rate=cut_cat_rate,
                                 cut_cat_times=cut_cat_times,

                                 bands_fif=bands_fif,
                                 filter_type=filter_type,
                                 freq=freq,
                                 frequency_max=frequency_max,
                                 frequency_min=frequency_min,
                                 order=order,

                                 scale=scale,

                                 load_data=load_data,
                                 data_path=data_path,
                                 label_path=label_path,

                                 ft_furrogate=ft_furrogate,
                                 ft_furrogate_rate=ft_furrogate_rate,
                                 ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                 ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                 gaussian_noise=gaussian_noise,
                                 gaussian_noise_rate=gaussian_noise_rate,
                                 gaussian_noise_std=gaussian_noise_std,
                                 smooth_time_mask=smooth_time_mask,
                                 smooth_time_mask_rate=smooth_time_mask_rate,
                                 smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                 )  # 测试机
        if save_data:
            dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=2)
        return loader

    def test_loader(self, data_augmentation=True, batch_size=64, drop_last=True, save_data=False,
                    save_data_path=None, num_workers=0,
                    load_data=False, data_path=None, label_path=None, ):

        if data_augmentation:
            shallow_windows = self.shallow_windows
            windows_overlapping = self.windows_overlapping
            windows_keeplast = self.windows_keeplast
            vertical_inversion = self.vertical_inversion
            res = self.res
            add_noise = self.add_noise
            cut_cat = self.cut_cat
            cut_cat_times = self.cut_cat_times
            ft_furrogate = self.ft_furrogate
            gaussian_noise = self.gaussian_noise
            smooth_time_mask = self.smooth_time_mask


        else:
            shallow_windows = None
            windows_overlapping = None
            windows_keeplast = None
            vertical_inversion = False
            add_noise = False
            res = None
            cut_cat = False
            cut_cat_times = None
            ft_furrogate = False
            gaussian_noise = False
            smooth_time_mask = False

        ft_furrogate_rate = self.ft_furrogate_rate
        ft_furrogate_phase_noise_magnitude = self.ft_furrogate_phase_noise_magnitude
        ft_furrogate_channel_indep = self.ft_furrogate_channel_indep
        gaussian_noise_rate = self.gaussian_noise_rate
        gaussian_noise_std = self.gaussian_noise_std
        smooth_time_mask_rate = self.smooth_time_mask_rate
        smooth_time_mask_len_samples = self.smooth_time_mask_len_samples
        windows_size = self.windows_size
        output_size = self.output_size
        bands_fif = self.bands_fif
        filter_type = self.filter_type
        freq = self.freq
        frequency_max = self.frequency_max
        frequency_min = self.frequency_min
        order = self.order
        scale = self.scale

        vertical_inversion_rate = self.vertical_inversion_rate
        add_noise_rate = self.add_noise_rate

        cut_cat_rate = self.cut_cat_rate
        dataset = KZDDataset_pic(self.sourcedata, type='E',
                                 shallow_windows=shallow_windows,
                                 windows_size=windows_size,
                                 windows_overlapping=windows_overlapping,
                                 windows_keeplast=windows_keeplast,
                                 output_size=output_size,

                                 vertical_inversion=vertical_inversion,
                                 vertical_inversion_rate=vertical_inversion_rate,

                                 add_noise=add_noise,
                                 res=res,
                                 add_noise_rate=add_noise_rate,

                                 cut_cat=cut_cat,
                                 cut_cat_rate=cut_cat_rate,
                                 cut_cat_times=cut_cat_times,

                                 bands_fif=bands_fif,
                                 filter_type=filter_type,
                                 freq=freq,
                                 frequency_max=frequency_max,
                                 frequency_min=frequency_min,
                                 order=order,

                                 scale=scale,

                                 load_data=load_data,
                                 data_path=data_path,
                                 label_path=label_path,
                                 ft_furrogate=ft_furrogate,
                                 ft_furrogate_rate=ft_furrogate_rate,
                                 ft_furrogate_phase_noise_magnitude=ft_furrogate_phase_noise_magnitude,
                                 ft_furrogate_channel_indep=ft_furrogate_channel_indep,
                                 gaussian_noise=gaussian_noise,
                                 gaussian_noise_rate=gaussian_noise_rate,
                                 gaussian_noise_std=gaussian_noise_std,
                                 smooth_time_mask=smooth_time_mask,
                                 smooth_time_mask_rate=smooth_time_mask_rate,
                                 smooth_time_mask_len_samples=smooth_time_mask_len_samples,
                                 )  # 测试机
        if save_data:
            dataset.save_data(save_data_path)
        if num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            drop_last=drop_last, num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            prefetch_factor=2)
        return loader


if __name__ == '__main__':
    # gc.enable()
    use_time = time.time()
    data_list = r"D:\BCI_DATA\HGD\raw"
    for subject_index in range(1, 15):
        subject = [subject_index]
        Source_data = HGD_dataset(data_list=data_list, subjects=subject, start_time=0.0, time=4,
                                  sel_chs=None,
                                  shallow_windows=False, windows_overlapping=1000,
                                  windows_keeplast=False,
                                  output_size=1000,
                                  bands_fif=True, freq=250, frequency_max=38, frequency_min=8,
                                  filter_type="lowpass", order=8,
                                  scale=True,
                                  vertical_inversion=True, vertical_inversion_rate=1,
                                  add_noise=True, add_noise_rate=1, res=1000,
                                  cut_cat=True, cut_cat_times=100, cut_cat_rate=1,
                                  ft_furrogate=True, ft_furrogate_rate=1,
                                  ft_furrogate_phase_noise_magnitude=0.5,
                                  ft_furrogate_channel_indep=False,
                                  gaussian_noise=True, gaussian_noise_rate=1,
                                  gaussian_noise_std=0.1,
                                  smooth_time_mask=True, smooth_time_mask_rate=1,
                                  smooth_time_mask_len_samples=200,
                                  )

        val_loader = Source_data.all_loader(data_augmentation=True, batch_size=64, drop_last=True,
                                            save_data=False,
                                            num_workers=0, )

        train_loader = Source_data.train_all_loader(data_augmentation=True, batch_size=64,
                                                    drop_last=True,
                                                    save_data=False,
                                                    num_workers=0, )

        test_loader = Source_data.test_loader(data_augmentation=True, batch_size=64, drop_last=True,
                                              save_data=False,
                                              num_workers=0, )

        print(len(train_loader))
        print(len(val_loader))
        print(len(test_loader))

        # for i, data in enumerate(train_loader):
        #     print("pre_use_time = ", time.time() - use_time)
        #     use_time = time.time()
        #     print(data[0].shape)
        #
        for i, data in enumerate(val_loader):
            print("pre_use_time = ", time.time() - use_time)
            use_time = time.time()
            print(data[0].shape)
    #
    # for i, data in enumerate(train_loader):
    #     print("pre_use_time = ", time.time() - use_time)
    #     use_time = time.time()
    #     print(data[0].shape)
