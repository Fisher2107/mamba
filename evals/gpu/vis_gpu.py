import pandas as pd
import matplotlib.pyplot as plt

# Load GPU stats
gpu_stats = pd.read_csv('gpu_stats.csv')
gpu_stats['Timestamp'] = pd.to_datetime(gpu_stats['Timestamp'])

# Load events
events = pd.read_csv('training_events.csv')
events['Timestamp'] = pd.to_datetime(events['Timestamp'])

# Set up the plot
plt.style.use('bmh')
plt.figure(figsize=(20, 10))

# Plot GPU Utilization
ax1 = plt.subplot(2, 1, 1)
sns.lineplot(x='Timestamp', y='GPU Utilization', data=gpu_stats, ax=ax1)
ax1.set_title('GPU Utilization Over Time')
ax1.set_xlabel('')

# Plot Memory Usage
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
sns.lineplot(x='Timestamp', y='Memory Usage', data=gpu_stats, ax=ax2)
ax2.set_title('GPU Memory Usage Over Time')

# Function to determine color based on event text
def get_event_color(event_text):
    return 'red' if 'start' in event_text.lower() else 'black'

# Add event markers to both plots
for ax in [ax1, ax2]:
    for _, event in events.iterrows():
        color = get_event_color(event['Event'])
        ax.axvline(x=event['Timestamp'], color=color, linestyle='--', alpha=0.5)
        ax.text(event['Timestamp'], ax.get_ylim()[1], event['Event'], 
                rotation=90, verticalalignment='top', fontsize=8, color=color)

plt.tight_layout()
plt.savefig('gpu_stats_with_events.png', dpi=300)
plt.show()

print("Visualization completed. The plot has been saved as 'gpu_stats_with_events.png'.")