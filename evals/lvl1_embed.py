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
city_count = 5
test_size=2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_loc=f'../data/start_2/{test_size}_{city_count}_{coord_dim}.pt'
test_data = torch.load(test_data_loc).to(device)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

checkpoint = torch.load('../checkpoints/try_same_3/Linear1_18-06_12-34.pt')
checkpoint2 = torch.load('../checkpoints/embed/fourier2_19-06_13-46.pt')

checkpoint3 = torch.load('../checkpoints/embed/Linear_mlp_4lay_20-06_13-11.pt')
checkpoint4 = torch.load('../checkpoints/embed/fourier2_mlp_19-06_12-11.pt')
args = checkpoint3['args']
print(checkpoint3['mean_tour_length_list'][-10:-1])
print(args)
'''args.mlp_cls = 'identity'
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_train.load_state_dict(checkpoint['model_state_dict'])
model_train.eval()'''


mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]

plt.plot(mean_tour_length_list3, label='Linear Embedding')
plt.plot(mean_tour_length_list4, label='Fourier Embedding')
#plt.plot(mean_tour_length_list3)
#plt.plot(mean_tour_length_list4)

greedy = greedy_tsp(test_data)[0].item()
exact = exact_solver(test_data).item()
print(greedy)
print(exact)


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.1, 2.64)

plt.legend()
plt.savefig('figs/mean_tour_length_embed2.pdf')
plt.show()