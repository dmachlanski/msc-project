import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import yaml

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

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

with open('plot_all_stas.yaml') as f:
    params = yaml.load(f)

rec = np.load(params['emg_path'])
if params['downsample']: rec = rec[:, ::2]
rs = rec if params['use_all'] else rec[params['ids']]

results = np.load(params['result_path'])
result = results[params['source_id']]

plot_id = 1

if params['is_ipt']:
    if params['new_format']:
        ipt = result['ipt']
    else:
        ipt = result

    if params['thr_mode'] == 'median':
        thr = params['thr_mul'] * np.median(np.abs(ipt) / 0.6745)
    elif params['thr_mode'] == 'squared':
        itp = ipt**2
        thr = params['thr_mul'] * np.std(ipt)
    else:
        thr = params['thr_mul'] * np.std(ipt)

    source, _ = find_peaks(ipt, height=thr)
    if params['plot_ipt']:
        plt.figure(plot_id)
        plot_id += 1
        plt.plot(result)
        plt.plot(source, result[source], 'x')
        plt.axhline(thr, color='red', linewidth=2)
else:
    if params['count'] > 0:
        source = result[params['start_id'] : params['start_id'] + params['count']]
    else:
        if params['new_format']:
            source = result['peaks']
        else:
            source = result

if params['plot_source']:
    plt.figure(plot_id)
    plot_id += 1
    plt.eventplot(source, linelengths=0.8, linewidths=0.8)

print(f'Number of peaks: {len(source)}')

if params['plot_rate']:
    plt.figure(plot_id)
    plot_id += 1
    bin_sec = np.bincount(source//params['freq'])
    bin_avg = np.mean(bin_sec)
    plt.bar(range(len(bin_sec)), bin_sec)
    plt.axhline(bin_avg, color='green', linewidth=1)
    plt.fill_between(range(len(bin_sec)), params['l_bound'], params['u_bound'], color='green', alpha=0.1)

# half before the spike and another half after the spike
window_half = params['window_half']

x = np.arange(((window_half * 2) + 1))

fig, axs = plt.subplots(params['n_rows'], params['n_cols'], sharey=not params['scale'])
axs = trim_axs(axs, len(rs))

for ax, r, ch in zip(axs, rs, EMG_CHANNEL_LIST):
    data = np.zeros((len(source), ((window_half * 2) + 1)))

    for i, idx in enumerate(source):
        # Make sure the window is within the signal range
        if idx < window_half or idx+window_half > len(r): continue

        if params['optimise']:
            search_start = idx - params['search_window']
            search_end = idx + params['search_window'] + 1
            max_id = np.argmax(r[search_start:search_end])

            if max_id > params['search_window']:
                new_idx = idx + max_id - params['search_window']
            elif max_id < params['search_window']:
                new_idx = idx - params['search_window'] + max_id
            else:
                new_idx = idx

            start = new_idx - window_half
            end = new_idx + window_half + 1
        else:
            start = idx - window_half
            end = idx + window_half + 1

        if params['sub_mean']:
            if params['mean_window'] > 0:
                start_w = np.mean(r[start : start + params['mean_window']])
                end_w = np.mean(r[end - params['mean_window'] : end])
                data[i, :] = r[start:end] - ((start_w + end_w)/2.0)
            else:
                data[i, :] = r[start:end] - np.mean(r[start:end])
        else:
            data[i, :] = r[start:end]

    avg = np.mean(data, axis=0)
    
    if params['show_snr']:
        if params['rms_snr']:
            power = data**2
            signal = np.sqrt(np.mean(power[params['noise_window']:-params['noise_window']]))
            noise = (np.sqrt(np.mean(power[:params['noise_window']])) + np.sqrt(np.mean(power[-params['noise_window']:])))/2.0
            #snr = 10.0 * np.log10(signal / noise)
            snr = signal / noise
        else:
            # 10log10(power / variance)
            power = avg**2
            peak_val = np.max(power)
            peak_id = np.argmax(power)
            variance = np.var(data, axis=0)
            snr = 10.0 * np.log10(peak_val / variance[peak_id])

        ax.set_title(f'SNR={snr:.2f}')
    elif params['titles']:
        ax.set_title(ch, fontsize='small', loc='left')

    if params['scale']: ax.set_ylim([params['y_min'], params['y_max']])
    if not params['axis']: ax.axis('off')

    ax.plot(avg, color='blue')

    if params['std_dev']:
        stdev = np.std(data, axis=0)
        upper_bound = avg + stdev
        lower_bound = avg - stdev
        ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.4)

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()