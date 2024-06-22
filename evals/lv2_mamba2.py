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


#checkpoint= torch.load('../checkpoints/reverse/Linear_reverse4_21-06_17-02.pt')

#args = checkpoint5['args']
#print(args)
'''args.mlp_cls = 'identity'
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_train.load_state_dict(checkpoint['model_state_dict'])
model_train.eval()'''


#mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]

#plt.plot(mean_tour_length_list, label='Reverse 4')


greedy = greedy_tsp(test_data,device=device)[0].item()
exact = exact_solver(test_data,device=device,split=10).item()
print(greedy)
print(exact)


'''plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.1, 2.3)
plt.title('All layers')

plt.legend()
plt.savefig('figs/reverse_all2.pdf')
plt.show()'''