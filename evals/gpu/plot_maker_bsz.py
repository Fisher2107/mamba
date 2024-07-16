import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def generate_plot(csv_files64,csv_files16):
    x_values = [5, 10, 20, 50, 100]
    y_values_64 = []
    y_values_16 = []
    
    for csv_file in csv_files64:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values_64.append(memory_usage)
    
    for csv_file in csv_files16:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values_16.append(memory_usage)

    #least squares fit
    z64 = np.polyfit(x_values, y_values_64, 2)
    x_range = np.arange(0, 200, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='quad fit dim = 64')

    z16 = np.polyfit(x_values, y_values_16, 2)
    plt.plot(x_range, np.polyval(z16, x_range), label='quad fit dim = 16')

    print('quad fit dim = 64:', z64)
    print('quad fit dim = 16:', z16)

    #plot
    plt.plot(x_values, y_values_64, marker='o', linestyle='None', label='dim = 64')
    plt.plot(x_values, y_values_16, marker='o', linestyle='None', label='dim = 16')
    plt.xlabel('City Count')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by City Count')
    plt.legend()
    plt.savefig('memory_usage.pdf')
    #plt.show()

# Example usage
csv_files_64 = ['../../checkpoints/gpu/city_count/mamba2_5_gpu_stats_15-07_15-45.csv',
              '../../checkpoints/gpu/city_count/mamba2_10_gpu_stats_15-07_15-46.csv', 
              '../../checkpoints/gpu/city_count/mamba2_20_gpu_stats_15-07_15-48.csv', 
              '../../checkpoints/gpu/city_count/mamba2_50_gpu_stats_15-07_15-49.csv', 
              '../../checkpoints/gpu/city_count/mamba2_100_gpu_stats_15-07_15-53.csv']

csv_files_16 = ['../../checkpoints/gpu/city_count/mamba2_5_d16_gpu_stats_15-07_15-58.csv',
              '../../checkpoints/gpu/city_count/mamba2_10_d16_gpu_stats_15-07_16-01.csv', 
              '../../checkpoints/gpu/city_count/mamba2_20_d16_gpu_stats_15-07_16-02.csv', 
              '../../checkpoints/gpu/city_count/mamba2_50_d16_gpu_stats_15-07_16-04.csv', 
              '../../checkpoints/gpu/city_count/mamba2_100_d16_gpu_stats_15-07_16-07.csv']

generate_plot(csv_files_64,csv_files_16)