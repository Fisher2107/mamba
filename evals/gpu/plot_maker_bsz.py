import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def generate_plot(csv_file):
    x_values = [10, 20, 30, 40, 50, 75, 100, 125, 150, 200]
    y_values = []
    
    for csv_file in csv_file:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
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
    plt.savefig('../figs/scale/bsz_memory_usage_ng.pdf')
    #plt.show()

# No point
csv_file = ['../../checkpoints/gpu/bsz/mamba2_50_bsz10_gpu_stats_15-07_16-13.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz20_gpu_stats_15-07_16-22.csv', 
            '../../checkpoints/gpu/bsz/mamba2_50_bsz30_gpu_stats_16-07_20-27.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz40_gpu_stats_16-07_20-31.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz50_gpu_stats_16-07_20-23.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz75_gpu_stats_16-07_20-35.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz100_gpu_stats_15-07_16-26.csv', 
            '../../checkpoints/gpu/bsz/mamba2_50_bsz125_gpu_stats_16-07_20-38.csv',
            '../../checkpoints/gpu/bsz/mamba2_50_bsz150_gpu_stats_15-07_16-28.csv', 
            '../../checkpoints/gpu/bsz/mamba2_50_bsz200_gpu_stats_15-07_16-30.csv']

#Point
csv_file_point = ['../../checkpoints/gpu/bsz_point/mamba2_50_bsz10_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz20_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz30_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz40_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz50_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz75_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz100_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz125_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz150_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point/mamba2_50_bsz200_gpu_stats.csv']

#Ng
csv_file_ng = ['../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz10_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz20_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz30_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz40_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz50_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz75_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz100_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz125_gpu_stats.csv',
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz150_gpu_stats.csv', 
            '../../checkpoints/gpu/bsz_point_ng/mamba2_50_bsz200_gpu_stats.csv']
generate_plot(csv_file_ng)