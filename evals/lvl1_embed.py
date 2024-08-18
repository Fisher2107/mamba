import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

#5city 3layer
'''checkpoint = torch.load('../checkpoints/embed2/fourier1_5city_3l_06-07_06-17.pt', map_location=device)
checkpoint2 = torch.load('../checkpoints/embed2/fourier2_5city_3l_05-07_22-04.pt', map_location=device)
checkpoint3 = torch.load('../checkpoints/embed2/fourier5_5city_3l_05-07_17-09.pt', map_location=device)
checkpoint4 = torch.load('../checkpoints/embed2/fourier10_5city_3l_05-07_23-04.pt', map_location=device)
checkpoint5 = torch.load('../checkpoints/embed2/linear_5city_3l_05-07_23-11.pt', map_location=device)'''

#5city 4layer
'''checkpoint = torch.load('../checkpoints/embed2/fourier1_5city_4l_05-07_21-38.pt')
checkpoint2 = torch.load('../checkpoints/embed2/fourier2_5city_4l_05-07_20-37.pt')
checkpoint3 = torch.load('../checkpoints/embed2/fourier5_5city_4l_06-07_07-24.pt')
checkpoint4 = torch.load('../checkpoints/embed2/fourier10_5city_4l_05-07_20-11.pt')
checkpoint5 = torch.load('../checkpoints/embed2/linear_5city_4l_06-07_00-11.pt')'''

#10city 3layer
checkpoint = torch.load('../checkpoints/embed2/fourier1_10city_3l_06-07_09-10.pt', map_location=device)
checkpoint2 = torch.load('../checkpoints/embed2/fourier2_10city_3l_06-07_01-37.pt', map_location=device)
checkpoint3 = torch.load('../checkpoints/embed2/fourier5_10city_3l_06-07_08-51.pt', map_location=device)
checkpoint4 = torch.load('../checkpoints/embed2/fourier10_10city_3l_06-07_03-54.pt', map_location=device)
checkpoint5 = torch.load('../checkpoints/embed2/linear_10city_3l_05-07_18-18.pt', map_location=device)

#10city 4layer
'''checkpoint = torch.load('../checkpoints/embed2/fourier1_10city_4l_06-07_00-17.pt')
checkpoint2 = torch.load('../checkpoints/embed2/fourier2_10city_4l_06-07_03-17.pt')
checkpoint3 = torch.load('../checkpoints/embed2/fourier5_10city_4l_06-07_22-45.pt')
checkpoint4 = torch.load('../checkpoints/embed2/fourier10_10city_4l_06-07_06-11.pt')
checkpoint5 = torch.load('../checkpoints/embed2/linear_10city_4l_05-07_17-11.pt')'''

mean_tour_length_list = [tensor.cpu().numpy() for tensor in checkpoint['mean_tour_length_list']]
mean_tour_length_list2 = [tensor.cpu().numpy() for tensor in checkpoint2['mean_tour_length_list']]
mean_tour_length_list3 = [tensor.cpu().numpy() for tensor in checkpoint3['mean_tour_length_list']]
mean_tour_length_list4 = [tensor.cpu().numpy() for tensor in checkpoint4['mean_tour_length_list']]
mean_tour_length_list5 = [tensor.cpu().numpy() for tensor in checkpoint5['mean_tour_length_list']]

#10 city
greedy = 3.1791656017303467
exact = 2.8630127906799316

'''#5 city
greedy = 2.207540988922119
exact = 2.121398448944092'''


print(min(mean_tour_length_list))
print(min(mean_tour_length_list2))
print(min(mean_tour_length_list3))
print(min(mean_tour_length_list4))
print(min(mean_tour_length_list5))

print('optimality gap', (min(mean_tour_length_list)-exact)*100/exact)
print('optimality gap', (min(mean_tour_length_list2)-exact)*100/exact)
print('optimality gap', (min(mean_tour_length_list3)-exact)*100/exact)
print('optimality gap', (min(mean_tour_length_list4)-exact)*100/exact)
print('optimality gap', (min(mean_tour_length_list5)-exact)*100/exact)

plt.plot(mean_tour_length_list, label='Fourier1 ')
plt.plot(mean_tour_length_list2, label='Fourier2 ')
plt.plot(mean_tour_length_list3, label='Fourier5 ')
plt.plot(mean_tour_length_list4, label='Fourier10 ')
plt.plot(mean_tour_length_list5, label='Linear ')

print(greedy)
print(exact)


plt.axhline(y=greedy, color='r', linestyle='--', label='Greedy Solver')
plt.axhline(y=exact, color='g', linestyle='--', label='Exact Solver')

# Add labels to the axes
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Mean Tour Length', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=11)
#plt.ylim(2.1, 2.64)

plt.legend(fontsize=12)
plt.savefig('figs/10_city/embed_3l.pdf')
plt.show()