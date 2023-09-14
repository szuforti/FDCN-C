from utils import PATH

CONSTANT = {

    'BCI_CI_2A': {
        'raw_path': 'datasets/BCI_CI_2A/raw',  # raw data path 'raw_path': 'datasets/BCIC2a'
        'n_subjs': 9,
        'n_trials': 144,
        'n_trials_per_class': 72,
        'n_chs': 22,
        'orig_smp_freq': 250,  # Original sampling frequency (Hz)
        'trial_len': 4,  # 7s
        'MI': {
            'start': 0,  # start at time = 2 s
            'stop': 4,  # stop at time = 6 s
            'len': 4,  # 4s
        },
        'orig_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                     'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                     'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', "?1", "?2"],
        'sel_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2', "?1", "?2"]
    },
    'HGD': {
        'raw_path': 'D:/BCI_DATA/OpenBMI/raw',  # raw data path
        'n_subjs': 14,
        'n_chs': 128,
        'orig_smp_freq': 500,  # Original sampling frequency  (Hz)
        'trial_len': 4,  # 8s (cut-off)
        'MI': {
            'start': 0,  # start at time = 0 s
            'stop': 4,  # stop at time = 0 s
            'len': 4,  # 4s
        },
        'orig_chs': ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1',
                     'T7',
                     'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4',
                     'P8', 'POz', 'O1', 'Oz', 'O2',
                     'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1',
                     'C2', 'C6', 'CP3', 'CPz',
                     'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7',
                     'TP8', 'PO7', 'PO8', 'FT9', 'FT10',
                     'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h',
                     'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
                     'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h',
                     'CPP4h', 'CPP6h', 'PPO1',
                     'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h',
                     'FFC2h', 'FFT8h', 'FTT9h',
                     'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h',
                     'TPP7h', 'CPP1h',
                     'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h',
                     'POO10h', 'OI1h', 'OI2h'],

        'sel_chs': ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                    'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                    'C6',
                    'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                    'FCC5h',
                    'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                    'CPP5h',
                    'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                    'CCP1h',
                    'CCP2h', 'CPP1h', 'CPP2h']
    }

}
