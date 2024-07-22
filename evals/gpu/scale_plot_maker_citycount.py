import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
def generate_plot(csv_files64,csv_files16,csv_files_lay1,saveto):
    x_values = [5, 10, 20, 50, 75, 100, 120]
    y_values_64 = []
    y_values_16 = []
    y_values_lay1 = []
    
    for csv_file in csv_files64:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_64.append(memory_usage)
        
    for csv_file in csv_files16:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_16.append(memory_usage)

    for csv_file in csv_files_lay1:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_lay1.append(memory_usage)

    #least squares fit
    z64 = np.polyfit(x_values, y_values_64, 2)
    x_range = np.arange(0, 200, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='quad fit dim = 64')

    z16 = np.polyfit(x_values, y_values_16, 2)
    plt.plot(x_range, np.polyval(z16, x_range), label='quad fit dim = 16')

    zlay1 = np.polyfit(x_values, y_values_lay1, 2)
    plt.plot(x_range, np.polyval(zlay1, x_range), label='quad fit lay1')

    print('quad fit dim = 64:', z64)
    print('quad fit dim = 16:', z16)
    print('quad fit lay1:', zlay1)

    #plot
    plt.plot(x_values, y_values_64, marker='o', linestyle='None', label='dim = 64')
    plt.plot(x_values, y_values_16, marker='o', linestyle='None', label='dim = 16')
    plt.plot(x_values, y_values_lay1, marker='o', linestyle='None', label='lay1')
    plt.xlabel('City Count')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by City Count')
    plt.legend()
    plt.savefig(saveto)
    #plt.show()

csv_files= ['../../checkpoints/gpu/city_count_point/mamba2_5_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_gpu_stats.csv']

csv_files_16 = ['../../checkpoints/gpu/city_count_point/mamba2_5_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_d16_gpu_stats.csv']

csv_files_lay1 = ['../../checkpoints/gpu/city_count_point/mamba2_5_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_lay1_gpu_stats.csv']

csv_files_ng = ['../../checkpoints/gpu/city_count_point/mamba2_5_ng_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_ng_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_ng_gpu_stats.csv']





generate_plot(csv_files,csv_files_16,csv_files_lay,'../figs/scale/memory_usage_ng.pdf')