# A study of spike-triggered EEG waveforms based on automated surface EMG decomposition of muscle activity comprising primal and fine hand movements

## CE901: MSc Dissertation Project

## Team
- Damian Machlanski (student)
- Dr Ana Matran-Fernandez (supervisor)
- Prof. Luca Citi (co-supervisor)

Project managed through [Jira](https://cseejira.essex.ac.uk/secure/RapidBoard.jspa?projectKey=C901P19031&rapidView=2504).

## Info
Some additional information about the code. A brief contents:
- Algorithms (base.py, kmckc.py, ckc.py).
- Running the decomposition (run_exp_mul.py).
- Running the tracking (track_sources.py).
- Validation of the KmCKC replication (experiments/sim/).
- EEG ERPs generation (experiments/eeg_sta/).
- Utility methods (utils.py, tracking.py).
- Experimental modifications (kmckc_mod.py, hybrid.py).
- Lots of helper scripts under 'helpers' directory (visualisations, manual testing, etc.).
- Some unit test under 'test' folder.
- The ISCTEST method (the matlab code) is under 'isctest' directory.
- Some useful scripts under 'experiments' in general. 'emg_proc/final' and 'emg_proc/final_detop' contain scripts used for the final decomposition.

## Data
Due to the volume of the data used, none of them were uploaded here. All of the recordings and results currently live on the 'detop' machine. Most of the heavy computing was done there.