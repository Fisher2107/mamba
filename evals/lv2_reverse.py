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

plt.style.use('bmh')

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

checkpoint= torch.load('../checkpoints/bugged/reverse10/mamba2_03-07_02-44.pt')
checkpoint2 = torch.load('../checkpoints/bugged/reverse10/mamba2_reverse_03-07_00-27.pt')
checkpoint3 = torch.load('../checkpoints/bugged/reverse10/mamba2_reversestart_02-07_22-09.pt')

checkpoint4 = torch.load('../checkpoints/reverse10/fixed_04-07_12-32.pt')
checkpoint5 = torch.load('../checkpoints/reverse10/fixed_reverse_04-07_11-19.pt')
checkpoint6 = torch.load('../checkpoints/reverse10/fixed_reversestart_04-07_10-13.pt')

checkpointlist = [checkpoint, checkpoint2, checkpoint3, checkpoint4, checkpoint5, checkpoint6]
print([i['args']['d_model'] for i in checkpointlist])

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



plt.plot(mean_tour_length_list, label='Standard')
plt.plot(mean_tour_length_list2, label='Bugged Reverse')
plt.plot(mean_tour_length_list3, label='RS')
#plt.plot(mean_tour_length_list4, label='Fixed Standard')
plt.plot(mean_tour_length_list5, label='Fixed Reverse')
#plt.plot(mean_tour_length_list6, label='Fixed RS')

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

plt.legend()
#plt.savefig('figs/10_city/reverse_3l.pdf')
plt.show()