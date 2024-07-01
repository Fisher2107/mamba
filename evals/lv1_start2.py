import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_solvers import greedy_tsp,exact_solver
from model import MambaFull
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

checkpoint = torch.load('../checkpoints/start2/start1andhalf_29-06_18-26.pt')
checkpoint2 = torch.load('../checkpoints/start2/start2_29-06_16-12.pt')

checkpoint3 = torch.load('../checkpoints/start2/start2andhalf_29-06_16-54.pt')
checkpoint4 = torch.load('../checkpoints/start2/start3_29-06_19-08.pt')

checkpoint5 = torch.load('../checkpoints/start2/start5_29-06_19-50.pt')
checkpoint6 = torch.load('../checkpoints/start2/start100_29-06_20-32.pt')

checkpoint7 = torch.load('../checkpoints/start2/startneg1_29-06_15-30.pt')
checkpoint8 = torch.load('../checkpoints/start2/startnegpoint1_29-06_17-44.pt')
checkpoint9 = torch.load('../checkpoints/start/Linear_mlp_randstart_20-06_19-04.pt')

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]
mean_tour_length_list7 = [tensor.cpu().numpy() for tensor in checkpoint7['mean_tour_length_list']]
mean_tour_length_list8 = [tensor.cpu().numpy() for tensor in checkpoint8['mean_tour_length_list']]
mean_tour_length_list9 = [tensor.cpu().numpy() for tensor in checkpoint9['mean_tour_length_list']]

varience = np.var([mean_tour_length_list[300],
                mean_tour_length_list2[300],
                mean_tour_length_list3[300],
                mean_tour_length_list4[300],
                mean_tour_length_list5[300],
                mean_tour_length_list7[300],
                mean_tour_length_list8[300]])

print('varience  ', varience) # 3.008733e-05

plt.plot(mean_tour_length_list, label='Start 1.5')
plt.plot(mean_tour_length_list2, label='Start 2')
plt.plot(mean_tour_length_list3, label='Start 2.5')
plt.plot(mean_tour_length_list4, label='Start 3')
plt.plot(mean_tour_length_list5, label='Start 5')
plt.plot(mean_tour_length_list6, label='Start 100')
plt.plot(mean_tour_length_list7, label='Start -1')
plt.plot(mean_tour_length_list8, label='Start -0.1')
plt.plot(mean_tour_length_list9, label='Start random')

greedy = greedy_tsp(test_data)[0].item()
exact = exact_solver(test_data,device='cuda').item()
print(greedy)
print(exact)


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.1, 2.6)

plt.legend()
#plt.savefig('figs/mean_tour_length_start2.pdf')
plt.show()