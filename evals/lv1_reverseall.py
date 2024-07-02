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

checkpoint= torch.load('../checkpoints/reverse/Linear_reverse4_21-06_17-02.pt')
checkpoint2 = torch.load('../checkpoints/start/Linear_mlp_2start_again_21-06_12-51.pt')

checkpoint3 = torch.load('../checkpoints/reverse/Linear_reverse3_21-06_15-12.pt')
checkpoint4 = torch.load('../checkpoints/reverse/Linear_3_21-06_16-30.pt')

checkpoint5 = torch.load('../checkpoints/reverse/Linear_reverse5_21-06_17-44.pt')
checkpoint6 = torch.load('../checkpoints/reverse/Linear_5_21-06_18-35.pt')

checkpoint7 = torch.load('../checkpoints/reverse/mamba2_reversestart_3l_01-07_10-43.pt')
checkpoint8 = torch.load('../checkpoints/reverse/mamba2_reversestart_4l_01-07_11-20.pt')
checkpoint9 = torch.load('../checkpoints/reverse/mamba2_reversestart_5l_01-07_12-03.pt')

args = checkpoint5['args']
print(args)
'''args.mlp_cls = 'identity'
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_train.load_state_dict(checkpoint['model_state_dict'])
model_train.eval()'''


mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list'][:1000]]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]
mean_tour_length_list7 = [tensor.cpu().numpy() for tensor in checkpoint7['mean_tour_length_list']]
mean_tour_length_list8 = [tensor.cpu().numpy() for tensor in checkpoint8['mean_tour_length_list']]
mean_tour_length_list9 = [tensor.cpu().numpy() for tensor in checkpoint9['mean_tour_length_list']]


plt.plot(mean_tour_length_list3, label='Reverse 3')
plt.plot(mean_tour_length_list, label='Reverse 4')
plt.plot(mean_tour_length_list5, label='Reverse 5')
plt.plot(mean_tour_length_list4, label='Standard 3')
plt.plot(mean_tour_length_list2, label='Standard 4')
plt.plot(mean_tour_length_list6, label='Standard 5')
plt.plot(mean_tour_length_list7, label='RS 3')
plt.plot(mean_tour_length_list8, label='RS 4')
plt.plot(mean_tour_length_list9, label='RS 5')

greedy = greedy_tsp(test_data)[0].item()
exact = exact_solver(test_data,device='cuda').item()
print(greedy)
print(exact)


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Mean Tour Length')
plt.ylim(2.1, 2.3)
plt.title('All layers')

plt.legend()
plt.savefig('figs/reverse_all3.pdf')
plt.show()