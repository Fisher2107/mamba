import torch
from model import generate_data
test_data = generate_data('cpu', 2000, 20, 2,start=2)
torch.save(test_data, 'data/start_2/2000_20_2.pt')
'''test_data = generate_data('cpu', 2000, 10, 2,start=2)
torch.save(test_data, 'data/start_2/2000_10_2.pt')
test_data = generate_data('cpu', 2000, 50, 2,start=2)
torch.save(test_data, 'data/start_2/2000_50_2.pt')
test_data = generate_data('cpu', 2000, 100, 2,start=2)
torch.save(test_data, 'data/start_2/2000_100_2.pt')
test_data = generate_data('cpu', 2000, 500, 2,start=2)
torch.save(test_data, 'data/start_2/2000_500_2.pt')
test_data = generate_data('cpu', 2000, 1000, 2,start=2)
torch.save(test_data, 'data/start_2/2000_1000_2.pt')

#Testing Start Tokens
test_data_loc1 = 'data/start_2/2000_5_2.pt'
test_data1 = torch.load(test_data_loc1)
print(test_data1.shape)

#Remove start token
test_data1 = test_data1[:,:-1,:]
print(test_data1.shape)

start_data = torch.full((2000, 1, 2), 5)
test_data1_5 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_5, 'data/start_5/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), -1)
test_data1_min1 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_min1, 'data/start_min1/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), -0.1)
test_data1_min01 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_min01, 'data/start_min01/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), 1.5)
test_data1_1p5 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_1p5, 'data/start_1p5/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), 2.5)
test_data1_2p5 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_2p5, 'data/start_2p5/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), 3)
test_data1_3 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_3, 'data/start_3/2000_5_2.pt')

start_data = torch.full((2000, 1, 2), 100)
test_data1_100 = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_100, 'data/start_100/2000_5_2.pt')

print(test_data1_5.shape)
print(test_data1_5[0,-1])

start_data = torch.rand(2000, 1, 2)
test_data1_rand = torch.cat((test_data1,start_data), dim=1)
torch.save(test_data1_rand, 'data/start_rand/2000_5_2.pt')'''

