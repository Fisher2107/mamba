import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_solvers import greedy_tsp,exact_solver
from model import MambaFull
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
checkpoint= torch.load('../checkpoints/pointer_v_standard_dim/pointer16_07-07_22-00.pt')
checkpoint2 = torch.load('../checkpoints/pointer_v_standard_dim/pointer32_07-07_16-43.pt')
checkpoint3 = torch.load('../checkpoints/pointer_v_standard_dim/pointer64_07-07_15-55.pt')

checkpoint4 = torch.load('../checkpoints/pointer_v_standard_dim/standard16_07-07_23-58.pt')
checkpoint5 = torch.load('../checkpoints/pointer_v_standard_dim/standard32_07-07_13-16.pt')
checkpoint6 = torch.load('../checkpoints/pointer_v_standard_dim/standard64_07-07_12-38.pt')

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]

plt.plot(mean_tour_length_list, label='Pointer d=16')
plt.plot(mean_tour_length_list2, label='Pointer d=32')
plt.plot(mean_tour_length_list3, label='Pointer d=64')
plt.plot(mean_tour_length_list4, label='Standard d=16')
plt.plot(mean_tour_length_list5, label='Standard d=32')
plt.plot(mean_tour_length_list6, label='Standard d=64')

greedy = 3.1791656017303467
exact = 2.8630127906799316


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')
# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.85, 3.2)

plt.title('All layers')

plt.legend()
plt.savefig('figs/mamba2_pointer_dims2.pdf')
plt.show()