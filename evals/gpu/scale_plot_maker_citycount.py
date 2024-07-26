import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
def generate_plot(csv_files,csv_files16,csv_files_lay1,csv_files_ng,saveto):
    x_values = [5, 10, 20, 50, 75, 100, 120]
    y_values = []
    y_values_16 = []
    y_values_lay1 = []
    y_values_ng = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] 
        y_values.append(memory_usage)
        
    for csv_file in csv_files16:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values_16.append(memory_usage)

    for csv_file in csv_files_lay1:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values_lay1.append(memory_usage)

    for csv_file in csv_files_ng:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1]
        y_values_ng.append(memory_usage)

    #least squares fit
    z64 = np.polyfit(x_values, y_values, 1)
    x_range = np.arange(0, 150, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='quad fit dim = 64',alpha=0.5)

    z16 = np.polyfit(x_values, y_values_16, 1)
    plt.plot(x_range, np.polyval(z16, x_range), label='quad fit dim = 16',alpha=0.5)

    zlay1 = np.polyfit(x_values, y_values_lay1, 1)
    plt.plot(x_range, np.polyval(zlay1, x_range), label='quad fit lay1',alpha=0.5)

    zng = np.polyfit(x_values, y_values_ng, 1)
    plt.plot(x_range, np.polyval(zng, x_range), label='quad fit ng',alpha=0.5)


    print('quad fit dim = 64:', z64)
    print('quad fit dim = 16:', z16)
    print('quad fit lay1:', zlay1)
    print('quad fit ng:', zng)

    #plot
    plt.plot(x_values, y_values, marker='o', linestyle='None', label='dim = 64',ms=5)
    plt.plot(x_values, y_values_16, marker='o', linestyle='None', label='dim = 16',ms=5)
    plt.plot(x_values, y_values_lay1, marker='o', linestyle='None', label='lay1',ms=5)
    plt.plot(x_values, y_values_ng, marker='o', linestyle='None', label='ng',ms=5)
    plt.xlabel('City Count')
    plt.ylabel('Memory Usage (GB)')
    plt.ylim(0, 10)
    plt.title('Memory Usage required to train model by City Count')
    plt.legend()
    plt.savefig(saveto)
    #plt.show()

csv_files= ['../../checkpoints/gpu/action2/mamba2_5_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_10_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_20_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_50_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_75_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_100_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_120_gpu_stats.csv',]

csv_files_16 = ['../../checkpoints/gpu/action2/mamba2_5_d16_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_10_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_20_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_50_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_75_d16_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_100_d16_gpu_stats.csv',
                '../../checkpoints/gpu/action2/mamba2_120_d16_gpu_stats.csv']

csv_files_lay1 = ['../../checkpoints/gpu/action2/mamba2_5_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_10_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_20_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_50_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_75_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_100_lay1_gpu_stats.csv',
                '../../checkpoints/gpu/action2/mamba2_120_lay1_gpu_stats.csv']

csv_files_ng = ['../../checkpoints/gpu/action2/mamba2_5_ng_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_10_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_20_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_50_ng_gpu_stats.csv', 
              '../../checkpoints/gpu/action2/mamba2_75_ng_gpu_stats.csv',
              '../../checkpoints/gpu/action2/mamba2_100_ng_gpu_stats.csv',
                '../../checkpoints/gpu/action2/mamba2_120_ng_gpu_stats.csv']







generate_plot(csv_files,csv_files_16,csv_files_lay1,csv_files_ng,'../figs/scale/memory_usage_action.pdf')