import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_solvers import greedy_tsp,exact_solver
from model import MambaFull
import torch
import torch.nn
import matplotlib.pyplot as plt
import numpy as np

coord_dim = 2
city_count = 50
test_size=2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_loc=f'../data/start_2/{test_size}_{city_count}_{coord_dim}.pt'
test_data = torch.load(test_data_loc).to(device)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
plt.style.use('bmh')
checkpoint= torch.load('../checkpoints/action/mamba2_action1_25-07_12-30.pt')
checkpoint2 = torch.load('../checkpoints/action/mamba2_action2_23-07_10-57.pt')
checkpoint3 = torch.load('../checkpoints/action/mamba2_action3_24-07_01-20.pt')

checkpoint4 = torch.load('../checkpoints/action/mamba2_tour1_22-07_21-50.pt')
checkpoint5 = torch.load('../checkpoints/action/mamba2_tour2_23-07_17-07.pt')
checkpoint6 = torch.load('../checkpoints/action/mamba2_tour3_24-07_15-16.pt')

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]

mean_tour_length_lists = [mean_tour_length_list, mean_tour_length_list2, mean_tour_length_list3, mean_tour_length_list4, mean_tour_length_list5, mean_tour_length_list6]

#for the first 3 mean tour len lists find mean and s.d of the last elementand same for the last 3
last_elem = []
for i in range(3):
    last_elem.append(mean_tour_length_lists[i][-1])
mean = np.mean(last_elem)
sd = np.std(last_elem)
print('action', mean, sd)

last_elem = []
for i in range(3,6):
    last_elem.append(mean_tour_length_lists[i][-1])
mean = np.mean(last_elem)
sd = np.std(last_elem)

print('tour', mean, sd)


plt.plot(mean_tour_length_list, label='Action 1')
plt.plot(mean_tour_length_list2, label='Action 2')
plt.plot(mean_tour_length_list3, label='Action 3')
plt.plot(mean_tour_length_list4, label='Tour 1')
plt.plot(mean_tour_length_list5, label='Tour 2')
plt.plot(mean_tour_length_list6, label='Tour 3')

greedy = 6.9641
#exact = 2.8630127906799316


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
#plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')
# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
#plt.ylim(2.85, 3.2)

plt.title('Training Curves for Action and Tour Models')

plt.legend()
plt.savefig('figs/action_v_tour.pdf')
plt.show()