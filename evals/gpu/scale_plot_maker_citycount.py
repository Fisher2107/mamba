import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def generate_plot_memory_train(csv_files,saveto):
    x_values = [5, 10, 20, 40, 60, 80, 100]
    total_memory = []
    memroy_train = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        memory_usage = df['Memory Usage'].iloc[-1] 
        total_memory.append(memory_usage)

    for csv_file in csv_files:
        df_events = pd.read_csv(csv_file.replace('gpu_stats','events'), parse_dates=[0])
        df = pd.read_csv(csv_file, parse_dates=[0])

        # Find the timestamp for 'Epoch 0, Step 19'
        timestamp = df_events[df_events['Event'].str.contains('Epoch 0, Step 15')].iloc[0, 0]

        # Find the first row in df where the timestamp is later than the extracted timestamp
        row = df[df['Timestamp'] > timestamp].iloc[0]

        # Return the value from the third column
        value = row.iloc[2]
        memroy_train.append(value)
    
    memroy_eval = [total_memory[i] - memroy_train[i] for i in range(len(total_memory))]

        


    train_mem = np.polyfit(x_values, memroy_train, 2)
    x_range = np.arange(0, 100, 0.1)
    plt.plot(x_range, np.polyval(train_mem, x_range), label='memory scaling')

    plt.plot(x_values, memroy_train, marker='o', linestyle='None',ms=5)
    #plt.plot(x_values, total_memory, marker='o', linestyle='None',ms=5, label='total')

    plt.rcParams['legend.fontsize'] = 14
    plt.xlabel('City Count',fontsize=16)
    plt.ylabel('Memory Usage (GB)',fontsize=16)
    plt.title('Memory Usage required to train model by City Count')
    plt.ylim(0, 22)
    plt.legend()
    plt.savefig(saveto)
    plt.close()

def generate_plot_time(csv_files,saveto):
    x_values = [5, 10, 20, 40, 60, 80, 100]
    total_time = []
    time_train = []
    time_eval = []

    for csv_file in csv_files:
        df_events = pd.read_csv(csv_file.replace('gpu_stats','events'), parse_dates=[0])
        df = pd.read_csv(csv_file, parse_dates=[0])

        train_time = pd.Timedelta(0)
        eval_time = pd.Timedelta(0)

        for epoch in range(4):  # Assuming 4 epochs
            start_event = f'Epoch {epoch}, Step 0 start'
            end_event = 'Testing on test data split 1'

            start_time = df_events[df_events['Event'].str.contains(start_event)]['Timestamp'].min()
            end_time = df_events[(df_events['Event'].str.contains(end_event)) & (df_events['Timestamp'] > start_time)]['Timestamp'].min()

            if start_time is not pd.NaT and end_time is not pd.NaT:
                train_time += end_time - start_time

        eval_time = df['Timestamp'].max() - df_events[df_events['Event'].str.contains(end_event)]['Timestamp'].max()

        total_time.append((train_time + eval_time).total_seconds())
        time_train.append(train_time.total_seconds())
        time_eval.append(eval_time.total_seconds())

    #least squares fit
    time_tot = np.polyfit(x_values, total_time, 2)
    x_range = np.arange(0, 100, 0.1)
    plt.plot(x_range, np.polyval(time_tot, x_range), label='total',alpha=0.5)

    time_trai = np.polyfit(x_values, time_train, 2)
    x_range = np.arange(0, 100, 0.1)
    plt.plot(x_range, np.polyval(time_trai, x_range), label='train',alpha=0.5)

    time_evl = np.polyfit(x_values, time_eval, 2)
    x_range = np.arange(0, 100, 0.1)
    plt.plot(x_range, np.polyval(time_evl, x_range), label='eval',alpha=0.5)

    #plot
    plt.plot(x_values, total_time, marker='o', linestyle='None',ms=5)
    plt.plot(x_values, time_train, marker='o', linestyle='None',ms=5)
    plt.plot(x_values, time_eval, marker='o', linestyle='None',ms=5)
    
    plt.rcParams['legend.fontsize'] = 14
    plt.xlabel('City Count',fontsize=16)
    plt.ylabel('Time Taken (s)',fontsize=16)
    plt.title('Time Taken required to train model by City Count')
    plt.legend()
    if 'importance_sampling' in saveto:
        plt.ylim(0, 240)
    else:
        plt.ylim(0, 530)        
            
    plt.savefig(saveto)
    plt.close()

csv_files_tour= ['../../checkpoints/gpu/tour_lightning/mamba2_5_gpu_stats.csv',
              '../../checkpoints/gpu/tour_lightning/mamba2_10_gpu_stats.csv', 
              '../../checkpoints/gpu/tour_lightning/mamba2_20_gpu_stats.csv', 
              '../../checkpoints/gpu/tour_lightning/mamba2_40_gpu_stats.csv', 
              '../../checkpoints/gpu/tour_lightning/mamba2_60_gpu_stats.csv',
              '../../checkpoints/gpu/tour_lightning/mamba2_80_gpu_stats.csv',
              '../../checkpoints/gpu/tour_lightning/mamba2_100_gpu_stats.csv',]

csv_file_importance_sampling5= ['../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_5_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_10_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_20_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_40_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_60_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_80_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/Reuse5_mamba2_100_gpu_stats.csv',]

csv_file_importance_sampling= ['../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_5_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_10_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_20_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_40_gpu_stats.csv', 
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_60_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_80_gpu_stats.csv',
                '../../checkpoints/gpu/importance_sampling_lightning/standard_mamba2_100_gpu_stats.csv',]

generate_plot_memory_train(csv_files_tour,'../figs/scale/lightning/memory_usage_train_tour2.pdf')
generate_plot_memory_train(csv_file_importance_sampling5,'../figs/scale/lightning/memory_usage_train_importance_sampling52.pdf')
generate_plot_memory_train(csv_file_importance_sampling,'../figs/scale/lightning/memory_usage_train_importance_sampling_standard2.pdf')

generate_plot_time(csv_files_tour,'../figs/scale/lightning/time_usage_tour2.pdf')
generate_plot_time(csv_file_importance_sampling5,'../figs/scale/lightning/time_usage_importance_sampling52.pdf')
generate_plot_time(csv_file_importance_sampling,'../figs/scale/lightning/time_usage_importance_sampling_standard2.pdf')