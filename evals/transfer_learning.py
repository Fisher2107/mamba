import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

checkpoint = torch.load('../checkpoints/big/64_G_city20_20-07_16-04.pt', map_location=device)
checkpoint2 = torch.load('../checkpoints/big/share_20_26-07_19-52.pt', map_location=device)

checkpoint3 = torch.load('../checkpoints/big/64_G_city50_20-07_17-16.pt', map_location=device)
checkpoint4 = torch.load('../checkpoints/big/share_50_27-07_11-12.pt', map_location=device)


mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]

mean_tour_length_lists = [mean_tour_length_list, mean_tour_length_list2, mean_tour_length_list3, mean_tour_length_list4,]

#plt.plot(mean_tour_length_list, label='No Transfer')
#plt.plot(mean_tour_length_list2[3991:], label='Transferred weights from city count 10')
plt.plot(mean_tour_length_list3, label='No Transfer')
plt.plot(mean_tour_length_list4[3991:], label='Transferred weights from city count 20')


# Increase font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

#greedy = 4.4843
greedy = 6.9641
#exact = 3.85
exact = 5.70

print(greedy)
print(exact)
#plt.ylim(3.8,4.5)
plt.ylim(5.68,7)

plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Mean Tour Length', fontsize=16)
plt.title('Transfer Learning vs No Transfer Learning')
plt.legend()
plt.savefig('figs/scale/transfer/50_transfer.pdf')
#plt.show()