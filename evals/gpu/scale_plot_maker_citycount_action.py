import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
def generate_plot(csv_files64,csv_files16):
    x_values = [10, 20, 50]
    y_values_64 = []
    y_values_16 = []
    
    for csv_file in csv_files64:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_64.append(memory_usage)
        
    for csv_file in csv_files16:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] - df['Memory Usage'].iloc[0]
        y_values_16.append(memory_usage)


    #least squares fit
    '''    z64 = np.polyfit(x_values, y_values_64, 2)
    x_range = np.arange(0, 200, 0.1)
    plt.plot(x_range, np.polyval(z64, x_range), label='quad fit dim = 64')

    z16 = np.polyfit(x_values, y_values_16, 2)
    plt.plot(x_range, np.polyval(z16, x_range), label='quad fit dim = 16')'''


    '''    print('quad fit dim = 64:', z64)
    print('quad fit dim = 16:', z16)'''

    #plot
    plt.plot( x_values, y_values_64, marker='o' ,label='action')
    plt.plot( x_values, y_values_16, marker='o', label='tour')
    plt.xlabel('City Count')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by City Count')
    plt.legend()
    plt.savefig('../figs/scale/memory_usage_action.pdf')
    #plt.show()

# Nopoint
Action = ['../../checkpoints/action/initial_experiments/city10_next_city_gpu_stats_20-07_11-33.csv',
              '../../checkpoints/action/initial_experiments/city20_next_city_gpu_stats_20-07_11-34.csv', 
              '../../checkpoints/action/initial_experiments/city50_next_city_gpu_stats_20-07_11-38.csv', ]

Tour = ['../../checkpoints/action/initial_experiments/city10_gpu_stats_20-07_11-49.csv',
              '../../checkpoints/action/initial_experiments/city20_gpu_stats_20-07_11-50.csv', 
              '../../checkpoints/action/initial_experiments/city50_gpu_stats_20-07_11-54.csv', ]


#generate_plot(csv_files_64,csv_files_16,csv_files_lay1)
#generate_plot(csv_files_64_point,csv_files_16_point,csv_files_lay1_point)
generate_plot(Action,Tour)