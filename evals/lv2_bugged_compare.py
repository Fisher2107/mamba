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
checkpoint= torch.load('../checkpoints/bugged/pointer/mamba2_1l_ident_27-06_18-08.pt')
checkpoint2 = torch.load('../checkpoints/bugged/pointer/mamba2_1l_band_27-06_19-30.pt')

checkpoint3 = torch.load('../checkpoints/bugged/pointer/mamba2_3l_ident_27-06_15-34.pt')
checkpoint4 = torch.load('../checkpoints/bugged/pointer/mamba2_3l_band_27-06_12-46.pt')

checkpoint5 = torch.load('../checkpoints/reverse10/fixed_128_04-07_18-37.pt')
checkpoint6 = torch.load('../checkpoints/reverse10/fixed_128_point_05-07_00-50.pt')
checkpoint7 = torch.load('../checkpoints/reverse10/fixed_reversestart_128_04-07_13-37.pt')
checkpoint8 = torch.load('../checkpoints/reverse10/fixed_reversestart_128_point_04-07_19-50.pt')

checkpoint9 = torch.load('../checkpoints/reverse10/fixed_reverse_128_04-07_14-50.pt')
checkpoint10 = torch.load('../checkpoints/reverse10/fixed_reverse_128_point_04-07_23-38.pt')


checkpointlist = [checkpoint3, checkpoint4, checkpoint5, checkpoint6, checkpoint7, checkpoint8, checkpoint9, checkpoint10]
print([i['args']['d_model'] for i in checkpointlist])

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]
mean_tour_length_list6 = [tensor.cpu().numpy() for tensor in checkpoint6['mean_tour_length_list']]
mean_tour_length_list7 = [tensor.cpu().numpy() for tensor in checkpoint7['mean_tour_length_list']]
mean_tour_length_list8 = [tensor.cpu().numpy() for tensor in checkpoint8['mean_tour_length_list']]
mean_tour_length_list9 = [tensor.cpu().numpy() for tensor in checkpoint9['mean_tour_length_list']]
mean_tour_length_list10 = [tensor.cpu().numpy() for tensor in checkpoint10['mean_tour_length_list']]

#plt.plot(mean_tour_length_list, label='Reverse 1l')
#plt.plot(mean_tour_length_list2, label='Reverse Pointer 1l')
plt.plot(mean_tour_length_list3, label='Bugged Reverse 3l')
plt.plot(mean_tour_length_list4, label='Bugged Reverse Pointer 3l')

plt.plot(mean_tour_length_list5, label='Standard 3l')
plt.plot(mean_tour_length_list6, label='Standard Pointer 3l')

plt.plot(mean_tour_length_list7, label='RS 3l')
plt.plot(mean_tour_length_list8, label='RS Pointer 3l')

plt.plot(mean_tour_length_list9, label='Fixed Reverse 3l')
plt.plot(mean_tour_length_list10, label='Fixed Reverse Pointer 3l')

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
plt.savefig('figs/mamba2_pointer2.pdf')
plt.show()