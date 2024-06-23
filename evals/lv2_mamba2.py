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
test_data_loc=f'../data/start_2/test_rand_{test_size}_{city_count}_{coord_dim}.pt'
test_data = torch.load(test_data_loc).to(device)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

checkpoint= torch.load('../checkpoints/mamba2/mamba_3lay_22-06_18-55.pt')
checkpoint2 = torch.load('../checkpoints/mamba2/mamba_4lay_22-06_23-42.pt')

checkpoint3 = torch.load('../checkpoints/mamba2/mamba2_3lay_22-06_16-09.pt')
checkpoint4 = torch.load('../checkpoints/mamba2/mamba2_4lay_22-06_17-16.pt')

print(checkpoint['time_tot'])
print(checkpoint2['time_tot'])
print(checkpoint3['time_tot'])
print(checkpoint4['time_tot'])
'''args.mlp_cls = 'identity'
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_train.load_state_dict(checkpoint['model_state_dict'])
model_train.eval()'''


mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]



plt.plot(mean_tour_length_list, label='m 3')
plt.plot(mean_tour_length_list2, label='m 4')
plt.plot(mean_tour_length_list3, label='m2 3')
plt.plot(mean_tour_length_list4, label='m2 4')

greedy = 3.1791656017303467
exact = 2.8630127906799316


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.7, 3.2)
plt.title('All layers')

plt.legend()
plt.savefig('figs/mamba2_comparison.pdf')
plt.show()