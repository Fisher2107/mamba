import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load GPU stats
gpu_stats = pd.read_csv('../../checkpoints/gpu/mamba2_10_recy5_gpu_stats_14-07_15-36.csv')
gpu_stats['Timestamp'] = pd.to_datetime(gpu_stats['Timestamp'])

# Load events
events = pd.read_csv('../../checkpoints/gpu/mamba2_10_recy5_events_14-07_15-36.csv')
events['Timestamp'] = pd.to_datetime(events['Timestamp'])

# Calculate seconds from start
start_time = min(gpu_stats['Timestamp'].min(), events['Timestamp'].min())
gpu_stats['Seconds'] = (gpu_stats['Timestamp'] - start_time).dt.total_seconds()
events['Seconds'] = (events['Timestamp'] - start_time).dt.total_seconds()



# Set up the plot
plt.figure(figsize=(20, 15))
sns.set(style="darkgrid")

# Plot GPU Utilization
ax1 = plt.subplot(3, 1, 1)
sns.lineplot(x='Seconds', y='GPU Utilization', data=gpu_stats, ax=ax1)
ax1.set_title('GPU Utilization Over Time')
ax1.set_xlabel('')

# Plot Memory Usage
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
sns.lineplot(x='Seconds', y='Memory Usage', data=gpu_stats, ax=ax2)
ax2.set_title('GPU Memory Usage Over Time')
ax2.set_xlabel('')

# Plot Power Usage
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
sns.lineplot(x='Seconds', y='Power Usage (W)', data=gpu_stats, ax=ax3)
ax3.set_title('GPU Power Usage Over Time')
ax3.set_xlabel('Time (seconds)')

# Function to determine color based on event text
def get_event_color(event_text):
    return 'red' if 'start' in event_text.lower() else 'black'

# Add event markers to all plots
for ax in [ax1, ax2, ax3]:
    for _, event in events.iterrows():
        color = get_event_color(event['Event'])
        if True:#if color == 'red':#
            ax.axvline(x=event['Seconds'], color=color, linestyle='--', alpha=0.5)
            ax.text(event['Seconds'], ax.get_ylim()[1], event['Event'], 
                rotation=90, verticalalignment='top', fontsize=8, color=color)

# Set x-axis to show time in seconds
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, max(gpu_stats['Seconds'].max(), events['Seconds'].max()))
    ax.set_xlim(22,28)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}s"))

plt.tight_layout()
plt.savefig('gpu_stats2_with_events_zoom.pdf')
plt.show()