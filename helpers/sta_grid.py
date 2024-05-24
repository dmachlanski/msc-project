import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo, elephant
from yaml import load, Loader

EMG_CHANNEL_LIST = ['MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'MA7', 'MA8', 'MA9',
                    'MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'MB6', 'MB7', 'MB8', 'MB9',
                    'MC1', 'MC2', 'MC3', 'MC4', 'MC5', 'MC6', 'MC7', 'MC8', 'MC9',
                    'MD1', 'MD2', 'MD3', 'MD4', 'MD5', 'MD6', 'MD7', 'MD8', 'MD9',
                    'ME1', 'ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7', 'ME8', 'ME9',
                    'MF1', 'MF2', 'MF3', 'MF4', 'MF5', 'MF6', 'MF7', 'MF8', 'MF9',
                    'MG1', 'MG2', 'MG3', 'MG4', 'MG5', 'MG6', 'MG7', 'MG8', 'MG9',
                    'MH1', 'MH2', 'MH3', 'MH4', 'MH5', 'MH6', 'MH7', 'MH8', 'MH9',
                    'MI1', 'MI2', 'MI3', 'MI4', 'MI5', 'MI6', 'MI7', 'MI8', 'MI9',
                    'MJ1', 'MJ2', 'MJ3', 'MJ4', 'MJ5', 'MJ6', 'MJ7', 'MJ8', 'MJ9',
                    'MK1', 'MK2', 'MK3', 'MK4', 'MK5', 'MK6', 'MK7', 'MK8', 'MK9',
                    'ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9',
                    'MM1', 'MM2', 'MM3', 'MM4', 'MM5', 'MM6', 'MM7', 'MM8', 'MM9',
                    'MN1', 'MN2', 'MN3', 'MN4', 'MN5', 'MN6', 'MN7', 'MN8', 'MN9']

with open('sta_grid.yaml') as f:
    params = load(f, Loader=Loader)

rec = np.load(params['rec_path'], allow_pickle=True)
trains = np.load(params['train_path'], allow_pickle=True)
t = trains[params['mu_id']]

train = np.array(t)/params['sampling_rate']
sig_stop = len(rec[0])/params['sampling_rate']
window_half = params['window']/2

spiketrain = neo.SpikeTrain(train, units='sec', t_stop=sig_stop)

fig, axs = plt.subplots(params['n_rows'], params['n_cols'], sharey=not params['scale'])

for ax, r, ch in zip(axs.flat, rec, EMG_CHANNEL_LIST):
    signal = neo.AnalogSignal(r.T, units=pq.mV, sampling_rate=params['sampling_rate'] * pq.Hz)
    sta = elephant.sta.spike_triggered_average(signal, spiketrain, (-window_half * pq.ms, window_half * pq.ms))

    if params['scale']: ax.set_ylim([params['y_min'], params['y_max']])
    if params['titles']: ax.set_title(ch, fontsize='x-small', loc='left')
    ax.axis('off')
    ax.plot(sta, color='blue')

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()