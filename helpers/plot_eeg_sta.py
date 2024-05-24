import numpy as np
import yaml
import mne
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import align_peaks

EEG_CHANNEL_LIST = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
                    'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                    'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1',
                    'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4',
                    'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2',
                    'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
                    'EXG1', 'EXG2', # Left and right ear  # 64, 65
                    'EXG3', 'EXG4', # Left and right HEOG for subj01, top and bottom VEOG for subj02, subj03  # 66, 67
                    # 'EXG5', 'EXG6', 'EXG7', 'EXG8', # not connected
                    'Status']

RIGHT_HAND_ELECTRODES = ['FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3']
LEFT_HAND_ELECTRODES = ['FC6', 'FC4', 'C6', 'C4', 'CP6', 'CP4']

EEG_FS = 8192.

def get_source_id(group, gesture, rep):
    for contraction in group:
        if contraction[0] == gesture and contraction[1] == rep:
            return contraction[2]
    return -1

def get_raw_eeg(order_info, params):
    eeg_data = []
    for contraction in order_info[params['subject']][params['session']]:
        eeg_fragment = np.load(f"{params['eeg_path']}eeg_proc_s0{params['subject']}_sess{params['session']}_{contraction[0]}_{contraction[1]}.npy")
        # Downsample EEG to 4 kHz
        eeg_data.append(eeg_fragment[:, ::2])
    eeg_sess = np.hstack(eeg_data)
    n_samples = eeg_data[0].shape[1]

    info = mne.create_info(EEG_CHANNEL_LIST[:64], EEG_FS/2, ch_types='eeg')
    raw_array = mne.io.RawArray(eeg_sess, info)

    return raw_array, n_samples

def get_emg_events(order_info, params, n_samples):
    source_info = np.load(f"{params['info_path']}emg_s0{params['subject']}_sess{params['session']}_{params['info_gesture']}_common.npy")
    events_raw = []
    for event_id, mu_group in enumerate(source_info):
        for offset_id, contraction in enumerate(order_info[params['subject']][params['session']]):
            source_id = get_source_id(mu_group, contraction[0], contraction[1])
            if source_id < 0: continue
            emg_peaks = np.load(f"{params['results_path']}emg_proc_s0{params['subject']}_sess{params['session']}_{contraction[0]}_{contraction[1]}_results.npy")[source_id]['peaks']

            # Align peaks
            emg_data = np.load(f"{params['emg_path']}emg_proc_s0{params['subject']}_sess{params['session']}_{contraction[0]}_{contraction[1]}.npy")
            emg_peaks = align_peaks(emg_data, emg_peaks, 75)

            partial_event = np.zeros((len(emg_peaks), 3), dtype=int)
            partial_event[:, 0] = (n_samples * offset_id) + emg_peaks
            partial_event[:, 2] = event_id + 1
            events_raw.append(partial_event)
    return np.vstack(events_raw)

if __name__ == "__main__":

    with open('plot_eeg_sta.yaml') as f:
        params = yaml.load(f)

    order_info = np.load('../data/order_info.npy').item()

    raw_eeg, n_samples = get_raw_eeg(order_info, params)
    events = get_emg_events(order_info, params, n_samples)

    #mne.viz.plot_events(events, sfreq=raw_eeg.info['sfreq'], first_samp=raw_eeg.first_samp)

    # event_repeated='merge'
    epochs = mne.Epochs(raw_eeg, events, tmin=-1.0, tmax=0.2, baseline=(None, -0.8), event_repeated='drop')

    data = epochs['1'].get_data(picks=RIGHT_HAND_ELECTRODES)

    x_ticks = np.linspace(-1.0, 0.2, data.shape[-1])

    data_flat = data.reshape((data.shape[0] * data.shape[1], data.shape[-1]))
    data_global_mean = np.mean(data_flat, axis=0)
    data_global_std = np.std(data_flat, axis=0)
    plt.figure(1)
    plt.plot(x_ticks, data_global_mean, color='blue')
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')
    plt.fill_between(x_ticks, data_global_mean - data_global_std, data_global_mean + data_global_std, color='blue', alpha=0.4)

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    upper_bounds = data_mean + data_std
    lower_bounds = data_mean - data_std

    fig, axs = plt.subplots(3, 2, sharey=True)
    axs = axs.flat

    for ax, d_mean, upper, lower, ch in zip(axs, data_mean, upper_bounds, lower_bounds, RIGHT_HAND_ELECTRODES):
        ax.set_title(ch)
        ax.plot(x_ticks, d_mean, color='blue')
        ax.axhline(0, color='black', linestyle='dotted')
        ax.axvline(0, color='black', linestyle='dotted')
        #upper = d_mean + d_std
        #lower = d_mean - d_std
        ax.fill_between(x_ticks, lower, upper, color='blue', alpha=0.4)
    
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    plt.show()

    # Try one source
    #epochs['1'].average().plot(picks=RIGHT_HAND_ELECTRODES)
    #epochs['1'].plot_image(picks=RIGHT_HAND_ELECTRODES, combine='mean')