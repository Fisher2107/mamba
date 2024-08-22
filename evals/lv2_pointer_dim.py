import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn
import matplotlib.pyplot as plt

coord_dim = 2
city_count = 10
test_size=2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_loc=f'../data/start_2/{test_size}_{city_count}_{coord_dim}.pt'
test_data = torch.load(test_data_loc).to(device)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
plt.style.use('bmh')
checkpoint= torch.load('../checkpoints/pointer_v_standard_dim/pointer16_07-07_22-00.pt', map_location=device)
checkpoint2 = torch.load('../checkpoints/pointer_v_standard_dim/pointer32_07-07_16-43.pt', map_location=device)
checkpoint3 = torch.load('../checkpoints/pointer_v_standard_dim/pointer64_07-07_15-55.pt', map_location=device)

checkpoint4 = torch.load('../checkpoints/pointer_v_standard_dim/standard16_07-07_23-58.pt', map_location=device)
checkpoint5 = torch.load('../checkpoints/pointer_v_standard_dim/standard32_07-07_13-16.pt', map_location=device)
checkpoint6 = torch.load('../checkpoints/pointer_v_standard_dim/standard64_07-07_12-38.pt', map_location=device)

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]

plt.plot(mean_tour_length_list, label='Pointer d=16')
plt.plot(mean_tour_length_list2, label='Pointer d=32')
plt.plot(mean_tour_length_list3, label='Pointer d=64')
plt.plot(mean_tour_length_list4, label='Sutskever d=16')
plt.plot(mean_tour_length_list5, label='Sutskever d=32')
plt.plot(mean_tour_length_list6, label='Sutskever d=64')

greedy = 3.1791656017303467
exact = 2.8630127906799316

# Increase font sizes
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11

plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')
# Add labels to the axes
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('Mean Tour Length',fontsize=16)
#plt.ylim(2.85, 3.2)

plt.title('Pointer v Sutskever on 10 City TSP')

plt.legend()
plt.savefig('figs/10_city/mamba2_pointer_dims11.pdf')
#plt.show()