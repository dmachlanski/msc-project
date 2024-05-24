import numpy as np
import matplotlib.pyplot as plt

def get_source_id(group, gesture, rep):
    for contraction in group:
        if contraction[0] == gesture and int(contraction[1]) == rep:
            return int(contraction[2])
    return -1

def get_emg_events(order_info, data_path, subject, session, fs):
    source_info = np.load(f"{data_path}decomposition/tracking/emg_s0{subject}_sess{session}_all_common.npy", allow_pickle=True)
    trains = []
    for mu_group in source_info:
        train = []
        for offset_id, contraction in enumerate(order_info[subject][session]):
            source_id = get_source_id(mu_group, contraction[0], contraction[1])
            if source_id < 0: continue
            source = np.load(f"{data_path}decomposition/final/emg_proc_s0{subject}_sess{session}_{contraction[0]}_{contraction[1]}_results.npy", allow_pickle=True)[source_id]
            emg_peaks = source['peaks']
            n_samples = len(source['ipt'])
            train.append(((n_samples * offset_id) + emg_peaks) / fs)
        trains.append(np.hstack(train))
    return trains

if __name__ == "__main__":
    order_info = np.load('../../data/order_info.npy', allow_pickle=True).item()

    data_path = '../../data/'
    subject = 4
    session = 2
    fs = 4096
    fsize = 10

    events = get_emg_events(order_info, data_path, subject, session, fs)

    plt.eventplot(events, linelengths=0.8, linewidths=0.4, colors=['blue', 'orange', 'green', 'green'])
    plt.tight_layout()
    plt.yticks(np.arange(0, len(events)), labels=[f'MU{i+1}' for i in range(len(events))])
    plt.ylabel('Motor units', fontsize=fsize)
    plt.xlabel('Time [s]', fontsize=fsize)
    legend = plt.legend(['pinch', 'mixed', 'grip'], fontsize=fsize, loc=2)
    for handle in legend.legendHandles:
        handle.set_linewidth(3.0)

    plt.show()