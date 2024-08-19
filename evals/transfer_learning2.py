import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn
import matplotlib.pyplot as plt
import numpy as np

coord_dim = 2
city_count = 5
test_size=2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_loc=f'../data/start_2/{test_size}_{city_count}_{coord_dim}.pt'
test_data = torch.load(test_data_loc).to(device)

plt.style.use('bmh')

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

checkpoint11 = torch.load('../checkpoints/transfer/50_24h_1.pt', map_location=device)
checkpoint12 = torch.load('../checkpoints/transfer/50_24h_2.pt', map_location=device)

checkpoint21 = torch.load('../checkpoints/transfer/50_12h_1.pt', map_location=device)
checkpoint22 = torch.load('../checkpoints/transfer/50_12h_2.pt', map_location=device)
checkpoint23 = torch.load('../checkpoints/transfer/20_12h_1.pt', map_location=device)
checkpoint24 = torch.load('../checkpoints/transfer/20_12h_2.pt', map_location=device)

checkpoint31 = torch.load('../checkpoints/transfer/50_12h_3.pt', map_location=device)
checkpoint32 = torch.load('../checkpoints/transfer/50_12h_4.pt', map_location=device)
checkpoint33 = torch.load('../checkpoints/transfer/10_4h_1.pt', map_location=device)
checkpoint34 = torch.load('../checkpoints/transfer/10_4h_2.pt', map_location=device)
checkpoint35 = torch.load('../checkpoints/transfer/20_8h_1.pt', map_location=device)
checkpoint36 = torch.load('../checkpoints/transfer/20_8h_2.pt', map_location=device)

mean_tour_length_list11 = [tensor.cpu().numpy() for tensor in checkpoint11['mean_tour_length_list']]
mean_tour_length_list12 = [tensor.cpu().numpy() for tensor in checkpoint12['mean_tour_length_list']]
mean_tour_length_list21 = [tensor.cpu().numpy() for tensor in checkpoint21['mean_tour_length_list']]
mean_tour_length_list22 = [tensor.cpu().numpy() for tensor in checkpoint22['mean_tour_length_list']]
mean_tour_length_list31 = [tensor.cpu().numpy() for tensor in checkpoint31['mean_tour_length_list']]
mean_tour_length_list32 = [tensor.cpu().numpy() for tensor in checkpoint32['mean_tour_length_list']]

time_list11 = checkpoint11['time_list']
time_list12 = checkpoint12['time_list']
time_list21 = checkpoint23['time_list'] + checkpoint21['time_list']
time_list22 = checkpoint23['time_list'] +checkpoint22['time_list']
time_list31 = checkpoint33['time_list'] + checkpoint35['time_list'] + checkpoint31['time_list']
time_list32 = checkpoint34['time_list'] + checkpoint36['time_list'] + checkpoint32['time_list']

#Divide by 3600 to get hours
time_list11 = [time/3600 for time in time_list11]
time_list12 = [time/3600 for time in time_list12]
time_list21 = [time/3600 for time in time_list21]
time_list22 = [time/3600 for time in time_list22]
time_list31 = [time/3600 for time in time_list31]
time_list32 = [time/3600 for time in time_list32]

# Increase font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

plt.plot(time_list11, mean_tour_length_list11, label='No Transfer 1', linewidth=1.5)
plt.plot(time_list12, mean_tour_length_list12, label='No Transfer 2', linewidth=1.5)
plt.plot(time_list21, mean_tour_length_list21, label='Transfer 1', linewidth=1.5)
plt.plot(time_list22, mean_tour_length_list22, label='Transfer 2', linewidth=1.5)
plt.plot(time_list31, mean_tour_length_list31, label='Transfer 3', linewidth=1.5)
plt.plot(time_list32, mean_tour_length_list32, label='Transfer 4', linewidth=1.5)

greedy = 6.9641
exact = 5.70

print(greedy)
print(exact)
plt.ylim(5.85,6.15)
plt.xlim(13,24)

plt.axhline(y=greedy, color='r', linestyle='--')
plt.axhline(y=exact, color='g', linestyle='--')

# Add labels to the axes
plt.xlabel('Hours Trained')
plt.ylabel('Mean Tour Length')
plt.title('Transfer Learning vs No Transfer Learning')
plt.legend()
plt.savefig('figs/scale/transfer/50_transfer_new2.pdf')
#plt.show()