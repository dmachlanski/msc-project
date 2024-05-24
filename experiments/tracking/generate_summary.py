import numpy as np
import pandas as pd

data_path = '../../data/decomposition/'

subjects = range(4, 8)
sessions = range(1, 6)
gestures = ['finger', 'fist']
gesture_rename = {'finger': 'pinch', 'fist': 'grip'}

table = []

for subject in subjects:
    for session in sessions:
        try:
            groups = np.load(f'{data_path}tracking/emg_s0{subject}_sess{session}_all_common.npy')
            
            for group in groups:
                peak_count = 0
                is_finger = False
                is_fist = False
                for trial in group:
                    peaks = np.load(f'{data_path}final/emg_proc_s0{subject}_sess{session}_{trial[0]}_{trial[1]}_results.npy')[int(trial[2])]['peaks']
                    peak_count += len(peaks)
                    if trial[0] == 'finger': is_finger = True
                    if trial[0] == 'fist': is_fist = True
                source_type = 'mixed'
                if is_finger and is_fist: source_type = 'mixed'
                elif is_finger: source_type = 'pinch'
                else: source_type = 'grip'

                table.append({'Subject': f'S0{subject}', 'Source Type': source_type, 'Pulses': peak_count})
        except:
            continue

df_table = pd.DataFrame.from_records(table)
df_table.to_csv('tracking_results.csv', index=False)