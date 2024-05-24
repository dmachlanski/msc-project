import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('trains_per_subject.yaml') as f:
    params = yaml.load(f)

gestures = ['finger', 'fist']
means = [[], []]

for sess in range(1, params['sessions'] + 1):
    for i, g in enumerate(gestures):
        filename = f"emg_proc_s0{params['subject']}_sess{sess}_{g}_unique.npy"
        data = np.load(f"{params['path']}{filename}")
        means[i].append(len(data))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

x = np.arange(params['sessions'])  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means[0], width, label=gestures[0])
rects2 = ax.bar(x + width/2, means[1], width, label=gestures[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Unique trains')
ax.set_title('Unique trains by session and gesture type')
ax.set_xticks(x)
ax.set_xticklabels(range(1, params['sessions'] + 1))
ax.legend()

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()