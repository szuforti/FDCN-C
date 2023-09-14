import numpy as np
import scipy.io as sio
from utils import resampling
from preprocessing.config import CONSTANT
from sklearn.model_selection import train_test_split
import h5py
import mne
from braindecode.datasets.bbci import BBCIDataset
from matplotlib import pyplot as plt

CONSTANT = CONSTANT['HGD']


def read_raw(PATH, subject, start=0, end=4):
    Train_mat_file_name = PATH + '\\Train\\' + str(subject) + ".mat"
    wanted_chan_inds, wanted_sensor_names = BBCIDataset(filename=Train_mat_file_name,
                                                        load_sensor_names=None)._determine_sensors()
    train_cnt = BBCIDataset(filename=Train_mat_file_name, load_sensor_names=None).load()
    Test_mat_file_name = PATH + '\\Test\\' + str(subject) + ".mat"
    test_cnt = BBCIDataset(filename=Test_mat_file_name, load_sensor_names=None).load()

    train_events, _ = mne.events_from_annotations(train_cnt, )
    train_picks = mne.pick_types(train_cnt.info, meg=False, eeg=True, stim=False, eog=False,
                                 exclude='bads')
    train_epochs = mne.Epochs(train_cnt, train_events, _, start, end, proj=True, picks=train_picks,
                              baseline=None,
                              preload=True)
    NO_valid_trial = 0
    size = len(train_epochs)
    n_trials = size
    n_chs = 128
    window_len = 4 * 500
    raw_train_data = np.zeros((n_trials, n_chs, window_len))
    label_train_data = np.zeros(n_trials)

    for trial in range(0, size):
        # num_class = 2: picked only class 1 (left hand) and class 2 (right hand) for our propose
        tmp_epoch = train_epochs[NO_valid_trial]
        tmp_data = tmp_epoch.get_data()[:, :, 1:]
        raw_train_data[NO_valid_trial, :, :] = tmp_data  # selected merely motor cortices region
        label_train_data[NO_valid_trial] = list(tmp_epoch.event_id.values())[0] - 1
        NO_valid_trial += 1
    raw_train_data = raw_train_data[0:NO_valid_trial, 0:128, :]

    test_events, _ = mne.events_from_annotations(test_cnt, )
    test_picks = mne.pick_types(test_cnt.info, meg=False, eeg=True, stim=False, eog=False,
                                exclude='bads')
    test_epochs = mne.Epochs(test_cnt, test_events, _, start, end, proj=True, picks=test_picks,
                             baseline=None,
                             preload=True)
    NO_valid_trial = 0
    size = len(test_epochs)
    n_trials = size
    raw_test_data = np.zeros((n_trials, n_chs, window_len))
    label_test_data = np.zeros(n_trials)
    for trial in range(0, size):
        tmp_epoch = test_epochs[NO_valid_trial]
        tmp_data = tmp_epoch.get_data()[:, :, 1:]
        # plt.plot(tmp_data[0, 0, :])
        # plt.show()
        raw_test_data[NO_valid_trial, :, :] = tmp_data  # selected merely motor cortices region
        label_test_data[NO_valid_trial] = list(tmp_epoch.event_id.values())[0] - 1
        NO_valid_trial += 1
    raw_test_data = raw_test_data[0:NO_valid_trial, 0:128, :]
    return raw_train_data, label_train_data, raw_test_data, label_test_data, wanted_chan_inds, \
           wanted_sensor_names


def load_crop_data(PATH, subjects, start, stop, new_smp_freq, id_chosen_chs):
    start_time = int(start * new_smp_freq)  # 2*
    stop_time = int(stop * new_smp_freq)  # 6*
    n_subjs = len(subjects)
    orig_smp_freq = CONSTANT['orig_smp_freq']
    if id_chosen_chs is not None:
        chs_len = len(id_chosen_chs)
    else:
        chs_len = len(CONSTANT["orig_chs"])
    MI_len = stop - start
    trial_len = stop - start
    X_train_all = np.empty(shape=(0, chs_len, int(MI_len * new_smp_freq)))
    y_tr_all = np.empty(shape=(0,))
    X_test_all = np.empty(shape=(0, chs_len, int(MI_len * new_smp_freq)))
    y_te_all = np.empty(shape=(0,))
    for s in subjects:
        X_train, y_tr, X_test, y_te, id, name = read_raw(PATH, s, start, stop)
        chs_id = chanel_selection(name, id_chosen_chs)
        X_train = X_train[:, chs_id, :]
        X_test = X_test[:, chs_id, :]
        if new_smp_freq < orig_smp_freq:
            X_train = resampling(X_train, new_smp_freq, trial_len)
            X_test = resampling(X_test, new_smp_freq, trial_len)
            print("Verify dimension training {} and testing {}".format(X_train.shape, X_test.shape))
        X_train_all = np.concatenate((X_train_all, X_train), axis=0)
        y_tr_all = np.append(y_tr_all, y_tr, axis=0)
        X_test_all = np.append(X_test_all, X_test, axis=0)
        y_te_all = np.append(y_te_all, y_te, axis=0)
    return X_train_all, y_tr_all, X_test_all, y_te_all


def chanel_selection(orig_chs, sel_chs):
    if sel_chs is None:
        chs_id = orig_chs
        # for ch_id, name_ch in enumerate(chs_id):
        # print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    else:
        chs_id = []
        for name_ch in sel_chs:
            ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
            chs_id.append(ch_id)
            # print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id


if __name__ == "__main__":
    data_list = r"D:\BCI_DATA\HGD\raw"
    id_chosen_chs = CONSTANT['sel_chs']
    X_train_all, y_tr_all, X_test_all, y_te_all = load_crop_data(data_list, [2], 0, 4, 500,
                                                                 id_chosen_chs)
    print(X_train_all.shape)
    print(X_test_all.shape)
