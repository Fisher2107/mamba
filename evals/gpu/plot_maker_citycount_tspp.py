import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
def generate_plot(csv_files64,csv_files16,csv_files_lay1):
    x_values = [10,20,50,100,150,200,250,300,400,500,600,700,800]
    y_values_64 = []
    y_values_16 = []
    y_values_lay1 = []
    
    for csv_file in csv_files64:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_64.append(memory_usage)
        
    '''for csv_file in csv_files16:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_16.append(memory_usage)

    for csv_file in csv_files_lay1:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_lay1.append(memory_usage)'''

    #least squares fit
    z64 = np.polyfit(x_values, y_values_64, 1)
    x_range = np.arange(0, 800, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='linear fit dim = 64')

    '''z16 = np.polyfit(x_values, y_values_16, 2)
    plt.plot(x_range, np.polyval(z16, x_range), label='quad fit dim = 16')

    zlay1 = np.polyfit(x_values, y_values_lay1, 2)
    plt.plot(x_range, np.polyval(zlay1, x_range), label='quad fit lay1')'''

    print('quad fit dim = 64:', z64)
    '''print('quad fit dim = 16:', z16)
    print('quad fit lay1:', zlay1)'''

    #plot
    plt.plot(x_values, y_values_64, marker='o', linestyle='None', label='dim = 64')
    '''plt.plot(x_values, y_values_16, marker='o', linestyle='None', label='dim = 16')
    plt.plot(x_values, y_values_lay1, marker='o', linestyle='None', label='lay1')'''
    plt.xlabel('City Count')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by City Count')
    plt.legend()
    plt.savefig('../figs/scale/memory_usage_tspp.pdf')
    #plt.show()

# Nopoint
csv_files_64 = ['../../checkpoints/gpu/city_count/mamba2_5_gpu_stats_15-07_15-45.csv',
              '../../checkpoints/gpu/city_count/mamba2_10_gpu_stats_15-07_15-46.csv', 
              '../../checkpoints/gpu/city_count/mamba2_20_gpu_stats_15-07_15-48.csv', 
              '../../checkpoints/gpu/city_count/mamba2_50_gpu_stats_15-07_15-49.csv', 
              '../../checkpoints/gpu/city_count/mamba2_75_gpu_stats_16-07_20-14.csv',
              '../../checkpoints/gpu/city_count/mamba2_100_gpu_stats_15-07_15-53.csv']

csv_files_16 = ['../../checkpoints/gpu/city_count/mamba2_5_d16_gpu_stats_15-07_15-58.csv',
              '../../checkpoints/gpu/city_count/mamba2_10_d16_gpu_stats_15-07_16-01.csv', 
              '../../checkpoints/gpu/city_count/mamba2_20_d16_gpu_stats_15-07_16-02.csv', 
              '../../checkpoints/gpu/city_count/mamba2_50_d16_gpu_stats_15-07_16-04.csv', 
              '../../checkpoints/gpu/city_count/mamba2_75_d16_gpu_stats_17-07_05-54.csv',
              '../../checkpoints/gpu/city_count/mamba2_100_d16_gpu_stats_15-07_16-07.csv']

csv_files_lay1 = ['../../checkpoints/gpu/city_count/mamba2_5_lay1_gpu_stats_17-07_05-58.csv',
              '../../checkpoints/gpu/city_count/mamba2_10_lay1_gpu_stats_17-07_05-59.csv', 
              '../../checkpoints/gpu/city_count/mamba2_20_lay1_gpu_stats_17-07_06-00.csv', 
              '../../checkpoints/gpu/city_count/mamba2_50_lay1_gpu_stats_17-07_06-01.csv', 
              '../../checkpoints/gpu/city_count/mamba2_75_lay1_gpu_stats_17-07_06-06.csv',
              '../../checkpoints/gpu/city_count/mamba2_100_lay1_gpu_stats_17-07_06-03.csv']

#Point
csv_files_64_point = ['../../checkpoints/gpu/city_count_point/mamba2_5_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_gpu_stats.csv']

csv_files_16_point = ['../../checkpoints/gpu/city_count_point/mamba2_5_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_d16_gpu_stats.csv']

csv_files_lay1_point = ['../../checkpoints/gpu/city_count_point/mamba2_5_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_10_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_20_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_50_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point/mamba2_75_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point/mamba2_100_lay1_gpu_stats.csv']

#Ng
csv_files_64_ng = ['../../checkpoints/gpu/city_count_point_ng/mamba2_5_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_10_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_20_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_50_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_75_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_100_gpu_stats.csv']

csv_files_16_ng = ['../../checkpoints/gpu/city_count_point_ng/mamba2_5_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_10_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_20_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_50_d16_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_75_d16_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_100_d16_gpu_stats.csv']

csv_files_lay1_ng = ['../../checkpoints/gpu/city_count_point_ng/mamba2_5_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_10_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_20_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_50_lay1_gpu_stats.csv', 
              '../../checkpoints/gpu/city_count_point_ng/mamba2_75_lay1_gpu_stats.csv',
              '../../checkpoints/gpu/city_count_point_ng/mamba2_100_lay1_gpu_stats.csv']

#TSPP
csv_files_64_tspp = csv_files_64_tspp = ['../../checkpoints/tspp/city10_gpu_stats.csv',
              '../../checkpoints/tspp/city20_gpu_stats.csv', 
              '../../checkpoints/tspp/city50_gpu_stats.csv', 
              '../../checkpoints/tspp/city100_gpu_stats.csv',
              '../../checkpoints/tspp/city150_gpu_stats.csv',
              '../../checkpoints/tspp/city200_gpu_stats.csv',
              '../../checkpoints/tspp/city250_gpu_stats.csv',
              '../../checkpoints/tspp/city300_gpu_stats.csv',
              '../../checkpoints/tspp/city400_gpu_stats.csv',
              '../../checkpoints/tspp/city500_gpu_stats.csv',
              '../../checkpoints/tspp/city600_gpu_stats.csv',
              '../../checkpoints/tspp/city700_gpu_stats.csv',
              '../../checkpoints/tspp/city800_gpu_stats.csv']

#generate_plot(csv_files_64,csv_files_16,csv_files_lay1)
#generate_plot(csv_files_64_point,csv_files_16_point,csv_files_lay1_point)
#generate_plot(csv_files_64_ng,csv_files_16_ng,csv_files_lay1_ng)
generate_plot(csv_files_64_tspp,[],[])