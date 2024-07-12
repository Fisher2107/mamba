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

checkpoint= torch.load('../checkpoints/pointer_v_standard_dim/pointer64_07-07_15-55.pt')
checkpoint2 = torch.load('../checkpoints/pointer_layers/layer3_10-07_04-08.pt')
checkpoint3 = torch.load('../checkpoints/pointer_generalisation/normal_pointer_10-07_07-41.pt')

checkpoint4 = torch.load('../checkpoints/mamba2_reverse/mamba2_reverse_point_07-07_22-41.pt')
checkpoint5 = torch.load('../checkpoints/mamba2_reverse/mamba2_reverse_point_11-07_14-42.pt')
checkpoint6 = torch.load('../checkpoints/mamba2_reverse/mamba2_reverse_point_11-07_16-48.pt')

checkpoint7 = torch.load('../checkpoints/mamba2_reverse/mamba2_reversestart_point_07-07_14-41.pt')
checkpoint8 = torch.load('../checkpoints/mamba2_reverse/mamba2_reversestart_point_11-07_12-39.pt')
checkpoint9 = torch.load('../checkpoints/mamba2_reverse/mamba2_reversestart_point_11-07_18-54.pt')



mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]
mean_tour_length_list7 = [tensor.cpu().numpy() for tensor in checkpoint7['mean_tour_length_list']]
mean_tour_length_list8 = [tensor.cpu().numpy() for tensor in checkpoint8['mean_tour_length_list']]
mean_tour_length_list9 = [tensor.cpu().numpy() for tensor in checkpoint9['mean_tour_length_list']]

mean_tour_length_lists = [mean_tour_length_list, mean_tour_length_list2, mean_tour_length_list3, mean_tour_length_list4, mean_tour_length_list5, mean_tour_length_list6, mean_tour_length_list7, mean_tour_length_list8, mean_tour_length_list9]
bests1=[min(i) for i in mean_tour_length_lists[:4]]
bests2=[min(i) for i in mean_tour_length_lists[4:7]]
bests3=[min(i) for i in mean_tour_length_lists[7:]]

print(np.mean(bests1),np.mean(bests2),np.mean(bests3))

print(np.sqrt(np.var(bests1)))
print(np.sqrt(np.var(bests2)))
print(np.sqrt(np.var(bests3)))

plt.plot(mean_tour_length_list, label='Standard')
plt.plot(mean_tour_length_list2, label='Standard2')
plt.plot(mean_tour_length_list3, label='Standard3')
plt.plot(mean_tour_length_list4, label='Reverse')
plt.plot(mean_tour_length_list5, label='Reverse2')
plt.plot(mean_tour_length_list6, label='Reverse3')
plt.plot(mean_tour_length_list7, label='ReverseStart')
plt.plot(mean_tour_length_list8, label='ReverseStart2')
plt.plot(mean_tour_length_list9, label='ReverseStart3')

greedy = 3.1791656017303467
exact = 2.8630127906799316
print(greedy)
print(exact)


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.title('All layers')
plt.xlim(1000, 2000)
plt.ylim(2.86,2.96)
plt.legend()
plt.savefig('figs/10_city/reverse_3l_pointer.pdf')
plt.show()