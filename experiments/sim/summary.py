import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from utils import eval_peaks_alt

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', type=str, dest='results_path')
    parser.add_argument('-g', type=str, dest='ground_truth_path')
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    fs = 4096.
    n_samples = 16 * fs
    n_subjects = 15
    mvc_levels = [10, 30, 50]

    all_scores = []
    all_times = []

    for subject_id in range(n_subjects):
        mvc = mvc_levels[subject_id % len(mvc_levels)]
        file_template = f'S{subject_id + 1}_{mvc}MVC'
        if options.verbose > 0: print(file_template)
        ground_truth = np.load(f'{options.ground_truth_path}{file_template}_peaks.npy', allow_pickle=True)
        trains = np.load(f'{options.results_path}{file_template}_trains.npy', allow_pickle=True)
        times = np.load(f'{options.results_path}{file_template}_times.npy', allow_pickle=True)

        for attempt, (pred_per_try, time_per_try) in enumerate(zip(trains, times)):
            scores = list(eval_peaks_alt(ground_truth, pred_per_try, n_samples, fs, n=2).values())
            for score in scores:
                score['subject'] = subject_id + 1
                score['mvc'] = mvc
                score['attempt'] = attempt + 1
            all_scores += scores

            time_entry = {'time': time_per_try, 'subject': subject_id + 1, 'mvc': mvc, 'attempt': attempt + 1}
            all_times.append(time_entry)
    
    df_scores = pd.DataFrame.from_records(all_scores)
    df_times = pd.DataFrame.from_records(all_times)

    df_scores.to_csv(f'{options.out_path}summary_scores.csv')
    df_times.to_csv(f'{options.out_path}summary_times.csv')