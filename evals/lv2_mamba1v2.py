import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_solvers import greedy_tsp,exact_solver
from model import MambaFull
import torch
import torch.nn
import matplotlib.pyplot as plt

plt.style.use('bmh')

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

checkpoint= torch.load('../checkpoints/mamba1_vs_2/mamba1_07-07_10-14.pt')
checkpoint2 = torch.load('../checkpoints/mamba1_vs_2/mamba2_07-07_11-15.pt')

checkpoint3 = torch.load('../checkpoints/mamba1_vs_2/mamba1_10-07_23-05.pt')
checkpoint4 = torch.load('../checkpoints/mamba1_vs_2/mamba2_10-07_18-29.pt')

checkpoint5 = torch.load('../checkpoints/mamba1_vs_2/mamba1_10-07_23-10.pt')
checkpoint6 = torch.load('../checkpoints/mamba1_vs_2/mamba2_10-07_23-06.pt')

print(checkpoint['args']['d_model'])
print(checkpoint['time_tot'])
print(checkpoint2['time_tot'])
#8121.574079275131
#7162.994921207428
#print(checkpoint3['time_tot'])
#print(checkpoint4['time_tot'])
#print(checkpoint4.keys())
'''args.mlp_cls = 'identity'
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_train.load_state_dict(checkpoint['model_state_dict'])
model_train.eval()'''


mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]



plt.plot(mean_tour_length_list, label='mamba_run1')
plt.plot(mean_tour_length_list3, label='mamba_run2')
plt.plot(mean_tour_length_list5, label='mamba_run3')
plt.plot(mean_tour_length_list2, label='mamba2_run1')
plt.plot(mean_tour_length_list4, label='mamba2_run2')
plt.plot(mean_tour_length_list6, label='mamba2_run3')

greedy = 3.1791656017303467
exact = 2.8630127906799316


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')
# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
#plt.ylim(2.85, 3.2)
plt.title('All layers')

plt.legend()
plt.savefig('figs/mamba1v2_comparison.pdf')
plt.show()