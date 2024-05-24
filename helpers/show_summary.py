import numpy as np
import pandas as pd
from collections import defaultdict

data_path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/emg_proc/final/filtered/'

subjects = range(4, 8)
sessions = range(1, 6)
gestures = ['finger', 'fist']
gesture_rename = {'finger': 'pinch', 'fist': 'grip'}
repetitions = range(1, 7)
fs = 4096

sources_total = 0
max_count = 0
sources_per_subject = defaultdict(lambda: defaultdict(int))
sources_per_gesture = defaultdict(int)

t_subject = []

for subject in subjects:
    for session in sessions:
        for gesture in gestures:
            for repetition in repetitions:
                try:
                    results = np.load(f'{data_path}emg_proc_s0{subject}_sess{session}_{gesture}_{repetition}_results.npy')
                    count = len(results)
                    sources_total += count
                    if count > max_count: max_count = count
                    sources_per_subject[subject]['total'] += count
                    sources_per_subject[subject][gesture] += count
                    sources_per_gesture[gesture] += count

                    for result in results:
                        t_subject.append({'Subject': f'S0{subject}', 'Gesture': gesture_rename[gesture], 'Pulses': len(result['peaks']), 'CoV': result['cov'], 'MDR': len(result['peaks']) / (len(result['ipt']) / fs)})
                except:
                    continue

print(f'Total number of sources = {sources_total}')
for subject in subjects:
    print(f"Subject {subject} = {sources_per_subject[subject]['total']}")
    for gesture in gestures:
        print(f'\t{gesture} = {sources_per_subject[subject][gesture]}')

for gesture in gestures:
    print(f'Total for {gesture} = {sources_per_gesture[gesture]}')

df_results = pd.DataFrame.from_records(t_subject)
df_results.to_csv('decomposition_results.csv', index=False)

print(f'Max count: {max_count}')