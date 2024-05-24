import argparse
import numpy as np
import mne
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
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

RIGHT_HAND_ELECTRODES_SHORT = ['FC5', 'FC3', 'C5', 'C3']
LEFT_HAND_ELECTRODES_SHORT = ['FC6', 'FC4', 'C6', 'C4']

CHANNEL_REFS = {'FC5': ['F5', 'FC3', 'C5', 'FT7'],
                'FC3': ['F3', 'FC1', 'C3', 'FC5'],
                'C5':  ['FC5', 'C3', 'CP5', 'T7'],
                'C3':  ['FC3', 'C1', 'CP3', 'C5'],
                'FC6': ['F6', 'FT8', 'C6', 'FC4'],
                'FC4': ['F4', 'FC6', 'C4', 'FC2'],
                'C6':  ['FC6', 'T8', 'C4'],
                'C4':  ['FC4', 'C6', 'CP4', 'C2']}

BAD_CHANNELS = {4: [],  # unchecked for now; note: EXG1 fell at some point of the experiment
                5: ['PO8'],  # from notes
                6: [],  # unchecked for now
                7: ['CP6'],  # from plots
                8: []  # unchecked for now
                }

EEG_FS = 4096.

def get_source_id(group, gesture, rep):
    for contraction in group:
        if contraction[0] == gesture and int(contraction[1]) == rep:
            return int(contraction[2])
    return -1

def get_raw_eeg(order_info, params):
    eeg_data = []
    for contraction in order_info[params.subject][params.session]:
        eeg_fragment = np.load(f"{params.eeg_path}eeg_proc_s0{params.subject}_sess{params.session}_{contraction[0]}_{contraction[1]}.npy", allow_pickle=True)
        if params.downsample:
            # Downsample EEG to 4 kHz
            eeg_data.append(eeg_fragment[:, ::2])
        else:
            eeg_data.append(eeg_fragment)
    eeg_sess = np.hstack(eeg_data)
    n_samples = eeg_data[0].shape[1]

    # Convert from microvolts to volts
    eeg_sess = eeg_sess / pow(10, 6)

    info = mne.create_info(EEG_CHANNEL_LIST[:64], EEG_FS, ch_types='eeg')
    raw_array = mne.io.RawArray(eeg_sess, info)

    return raw_array, n_samples

def get_emg_events(order_info, params, n_samples):
    source_info = np.load(f"{params.data_path}decomposition/tracking/emg_s0{params.subject}_sess{params.session}_all_common.npy", allow_pickle=True)
    events_raw = []
    for event_id, mu_group in enumerate(source_info):
        for offset_id, contraction in enumerate(order_info[params.subject][params.session]):
            source_id = get_source_id(mu_group, contraction[0], contraction[1])
            if source_id < 0: continue
            emg_peaks = np.load(f"{params.data_path}decomposition/final/emg_proc_s0{params.subject}_sess{params.session}_{contraction[0]}_{contraction[1]}_results.npy", allow_pickle=True)[source_id]['peaks']

            # Align peaks
            emg_data = np.load(f"{params.data_path}processed/emg_proc_s0{params.subject}_sess{params.session}_{contraction[0]}_{contraction[1]}.npy", allow_pickle=True)
            emg_peaks = align_peaks(emg_data, emg_peaks, 75)

            partial_event = np.zeros((len(emg_peaks), 3), dtype=int)
            partial_event[:, 0] = (n_samples * offset_id) + emg_peaks
            partial_event[:, 2] = event_id + 1
            events_raw.append(partial_event)
    return np.vstack(events_raw)

def plot_and_save(data, electrodes, group_id, side, type_name, options):
    x_ticks = np.linspace(-1.0, 0.2, data.shape[-1])
    fsize = (16, 8)

    data_flat = data.reshape((data.shape[0] * data.shape[1], data.shape[-1]))
    data_global_mean = np.mean(data_flat, axis=0)
    data_global_stderr = np.std(data_flat, axis=0) / np.sqrt(data_flat.shape[0])
    global_up_bound = data_global_mean + data_global_stderr
    global_low_bound = data_global_mean - data_global_stderr

    data_mean = np.mean(data, axis=0)
    data_stderr = np.std(data, axis=0) / np.sqrt(data.shape[0])
    up_bounds = data_mean + data_stderr
    low_bounds = data_mean - data_stderr

    plt.figure(figsize=fsize)
    plt.plot(x_ticks, data_global_mean, color='blue')
    plt.title(f'N={data.shape[0]}')
    for d_mean in data_mean:
        plt.plot(x_ticks, d_mean, color='black', alpha=0.2, linewidth=0.1)
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\mu$V')
    plt.fill_between(x_ticks, global_low_bound, global_up_bound, color='blue', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_{side}_{type_name}_mean.pdf', dpi=options.dpi)

    _, axs = plt.subplots(3, 2, sharey=False, sharex=True, figsize=fsize)
    axs = axs.flat

    for i, (ax, d_mean, up_bound, low_bound, ch) in enumerate(zip(axs, data_mean, up_bounds, low_bounds, electrodes)):
        ax.set_title(f'{ch} (N={data.shape[0]})')
        #for epoch in data[:, i, :]:
            #ax.plot(x_ticks, epoch, color='black', alpha=0.2, linewidth=0.1)
        ax.plot(x_ticks, d_mean, color='blue')
        ax.axhline(0, color='black', linestyle='dotted')
        ax.axvline(0, color='black', linestyle='dotted')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\mu$V')
        ax.fill_between(x_ticks, low_bound, up_bound, color='blue', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_{side}_{type_name}_channels.pdf', dpi=options.dpi)

def plot_both_sides(data_left, data_right, group_id, type_name, options):
    x_ticks = np.linspace(-1.0, 0.2, data_left.shape[-1])
    fsize = (16, 8)

    data_left_flat = data_left.reshape((data_left.shape[0] * data_left.shape[1], data_left.shape[-1]))
    data_right_flat = data_right.reshape((data_right.shape[0] * data_right.shape[1], data_right.shape[-1]))
    left_mean = np.mean(data_left_flat, axis=0)
    right_mean = np.mean(data_right_flat, axis=0)
    left_stderr = np.std(data_left_flat, axis=0) / np.sqrt(data_left_flat.shape[0])
    right_stderr = np.std(data_right_flat, axis=0) / np.sqrt(data_right_flat.shape[0])
    left_up_bound = left_mean + left_stderr
    right_up_bound = right_mean + right_stderr
    left_low_bound = left_mean - left_stderr
    right_low_bound = right_mean - right_stderr

    plt.figure(figsize=fsize)
    plt.plot(x_ticks, left_mean, color='blue', label='left hand')
    plt.plot(x_ticks, right_mean, color='orange', label='right hand')
    plt.title(f'N = {data_left.shape[0]}')
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\mu$V')
    plt.legend(loc='upper left')
    plt.fill_between(x_ticks, left_low_bound, left_up_bound, color='blue', alpha=0.2)
    plt.fill_between(x_ticks, right_low_bound, right_up_bound, color='orange', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_{type_name}_mean.pdf', dpi=options.dpi)

    left_mean_ch = np.mean(data_left, axis=0)
    right_mean_ch = np.mean(data_right, axis=0)
    left_stderr_ch = np.std(data_left, axis=0) / np.sqrt(data_left.shape[0])
    right_stderr_ch = np.std(data_right, axis=0) / np.sqrt(data_right.shape[0])
    left_up_bounds = left_mean_ch + left_stderr_ch
    right_up_bounds = right_mean_ch + right_stderr_ch
    left_low_bounds = left_mean_ch - left_stderr_ch
    right_low_bounds = right_mean_ch - right_stderr_ch

    _, axs = plt.subplots(3, 2, sharey=False, sharex=True, figsize=fsize)
    axs = axs.flat

    for ax, l_mean, r_mean, l_low_bound, r_low_bound, l_up_bound, r_up_bound, left_ch, right_ch in zip(axs, left_mean_ch, right_mean_ch, left_low_bounds, right_low_bounds, left_up_bounds, right_up_bounds, LEFT_HAND_ELECTRODES, RIGHT_HAND_ELECTRODES):
        ax.set_title(f'{left_ch} and {right_ch} (N={data_left.shape[0]})')
        ax.plot(x_ticks, l_mean, color='blue', label=left_ch)
        ax.plot(x_ticks, r_mean, color='orange', label=right_ch)
        ax.axhline(0, color='black', linestyle='dotted')
        ax.axvline(0, color='black', linestyle='dotted')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\mu$V')
        ax.legend(loc='upper left')
        ax.fill_between(x_ticks, l_low_bound, l_up_bound, color='blue', alpha=0.2)
        ax.fill_between(x_ticks, r_low_bound, r_up_bound, color='orange', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_{type_name}_channels.pdf', dpi=options.dpi)

def plot_refs(data, group_id, type_name, options):
    fsize = (16, 8)
    fig1, axs1 = plt.subplots(2, 2, sharey=False, sharex=True, figsize=fsize)
    fig2, axs2 = plt.subplots(2, 2, sharey=False, sharex=True, figsize=fsize)
    fig3, axs3 = plt.subplots(2, 2, sharey=False, sharex=True, figsize=fsize)

    axs1 = axs1.flat
    axs2 = axs2.flat
    axs3 = axs3.flat
    for ax1, ax2, ax3, left_ch, right_ch in zip(axs1, axs2, axs3, LEFT_HAND_ELECTRODES_SHORT, RIGHT_HAND_ELECTRODES_SHORT):
        left_val, left_err = get_ref_value(data, left_ch)
        right_val, right_err = get_ref_value(data, right_ch)
        size = int(len(left_val) * 0.5)
        left_val = left_val[size:]
        right_val = right_val[size:]
        left_err = left_err[size:]
        right_err = right_err[size:]
        x_ticks = np.linspace(-0.4, 0.2, size)

        ax1.set_title(f'{left_ch} and {right_ch} (N={len(data)})')
        ax1.plot(x_ticks, left_val, color='blue', label=left_ch)
        ax1.plot(x_ticks, right_val, color='orange', label=right_ch)
        ax1.axhline(0, color='black', linestyle='dotted')
        ax1.axvline(0, color='black', linestyle='dotted')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(r'$\mu$V')
        ax1.legend(loc='upper left')
        ax1.fill_between(x_ticks, left_val - left_err, left_val + left_err, color='blue', alpha=0.2)
        ax1.fill_between(x_ticks, right_val - right_err, right_val + right_err, color='orange', alpha=0.2)

        ax2.set_title(f'{left_ch} (N={len(data)})')
        ax2.plot(x_ticks, left_val, color='blue')
        ax2.axhline(0, color='black', linestyle='dotted')
        ax2.axvline(0, color='black', linestyle='dotted')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel(r'$\mu$V')
        ax2.fill_between(x_ticks, left_val - left_err, left_val + left_err, color='blue', alpha=0.2)

        ax3.set_title(f'{right_ch} (N={len(data)})')
        ax3.plot(x_ticks, right_val, color='orange')
        ax3.axhline(0, color='black', linestyle='dotted')
        ax3.axvline(0, color='black', linestyle='dotted')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'$\mu$V')
        ax3.fill_between(x_ticks, right_val - right_err, right_val + right_err, color='orange', alpha=0.2)

    fig1.tight_layout()
    fig1.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_{type_name}_refs.pdf', dpi=options.dpi)
    fig2.tight_layout()
    fig2.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_left_{type_name}_refs.pdf', dpi=options.dpi)
    fig3.tight_layout()
    fig3.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{group_id}_right_{type_name}_refs.pdf', dpi=options.dpi)

def get_ref_value(data, channel):
    # Microvolts are more appropriate here
    main_ch = data.get_data(picks=channel) * pow(10, 6)

    # From (n_epochs, n_channels, window) to (n_epochs, window) - only 1 channel
    main_ch = main_ch.reshape((main_ch.shape[0], main_ch.shape[-1]))

    # Convert to microvolts as well
    # (n_epochs, n_channels, window size)
    ref_chs = data.get_data(picks=CHANNEL_REFS[channel]) * pow(10, 6)

    # (n_epochs, window)
    main_ch_corrected = main_ch - np.mean(ref_chs, axis=1)
    result_mean = np.mean(main_ch_corrected, axis=0)
    result_err = np.std(main_ch_corrected, axis=0) / np.sqrt(main_ch_corrected.shape[0])

    return result_mean, result_err

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, dest='data_path')
    parser.add_argument('--eeg', type=str, dest='eeg_path')
    parser.add_argument('--ds', dest='downsample', action='store_true', default=False)
    parser.add_argument('-s', type=int, dest='subject')
    parser.add_argument('--sess', type=int, dest='session')
    parser.add_argument('--types', nargs='+')
    parser.add_argument('--dpi', type=int, dest='dpi', default=100)
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

if __name__ == "__main__":
    mne.set_log_level(False)

    parser = get_parser()
    options = parser.parse_args()

    order_info = np.load('../../data/order_info.npy', allow_pickle=True).item()

    raw_eeg, n_samples = get_raw_eeg(order_info, options)
    events = get_emg_events(order_info, options, n_samples)

    epochs = mne.Epochs(raw_eeg, events, tmin=-1.0, tmax=0.2, baseline=(None, -0.8), event_repeated='drop')
    epochs.set_montage('biosemi64')

    for epoch_id, type_name in enumerate(options.types):
        epoch_name = str(epoch_id + 1)
        
        # Temporarily disabled
        #data_left = epochs[epoch_name].get_data(picks=LEFT_HAND_ELECTRODES)
        #plot_and_save(data_left, LEFT_HAND_ELECTRODES, epoch_name, 'left', type_name, options)
        #data_right = epochs[epoch_name].get_data(picks=RIGHT_HAND_ELECTRODES)
        #plot_and_save(data_right, RIGHT_HAND_ELECTRODES, epoch_name, 'right', type_name, options)
        #plot_both_sides(data_left, data_right, epoch_name, type_name, options)

        #plot_refs(epochs[epoch_name], epoch_name, type_name, options)

        electrodes = LEFT_HAND_ELECTRODES + RIGHT_HAND_ELECTRODES

        # Drop channels only if there's anything to drop.
        # Otherwise MNE throws an error.
        if set(BAD_CHANNELS[options.subject]) & set(electrodes):
            fig1 = epochs[epoch_name].average(picks=electrodes).drop_channels(BAD_CHANNELS[options.subject]).crop(-0.4, 0.2).plot_joint(show=False)
        else:
            fig1 = epochs[epoch_name].average(picks=electrodes).crop(-0.4, 0.2).plot_joint(show=False)

        fig1.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{epoch_name}_{type_name}_joint.pdf', dpi=options.dpi)

        #fig2 = epochs[epoch_name].average(picks=LEFT_HAND_ELECTRODES).drop_channels(BAD_CHANNELS[options.subject]).crop(-0.4, 0.2).plot_joint(show=False)
        #fig2.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{epoch_name}_left_{type_name}_joint.pdf', dpi=options.dpi)
        #fig3 = epochs[epoch_name].average(picks=RIGHT_HAND_ELECTRODES).drop_channels(BAD_CHANNELS[options.subject]).crop(-0.4, 0.2).plot_joint(show=False)
        #fig3.savefig(f'{options.out_path}eeg_sta_S0{options.subject}_sess{options.session}_MU{epoch_name}_right_{type_name}_joint.pdf', dpi=options.dpi)