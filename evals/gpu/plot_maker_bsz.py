import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def generate_plot(csv_file):
    x_values = [10, 20, 100, 150,200]
    y_values = []
    
    for csv_file in csv_file:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values.append(memory_usage)
    

    #least squares fit
    '''z64 = np.polyfit(x_values, y_values_64, 2)
    x_range = np.arange(0, 200, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='quad fit dim = 64')

    print('quad fit dim = 16:', z16)'''

    #plot
    plt.plot(x_values, y_values, marker='o', label='dim = 64')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by Batch Size')
    plt.legend()
    #plt.savefig('memory_usage_bsz.pdf')
    plt.show()

# Example usage
csv_file = ['../../checkpoints/gpu/bsz/mamba2_50_bsz10_gpu_stats_15-07_16-13.csv',
              '../../checkpoints/gpu/bsz/mamba2_50_bsz20_gpu_stats_15-07_16-22.csv', 
              '../../checkpoints/gpu/bsz/mamba2_50_bsz100_gpu_stats_15-07_16-26.csv', 
              '../../checkpoints/gpu/bsz/mamba2_50_bsz150_gpu_stats_15-07_16-28.csv', 
              '../../checkpoints/gpu/bsz/mamba2_50_bsz200_gpu_stats_15-07_16-30.csv']


generate_plot(csv_file)