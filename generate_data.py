import torch
from model import generate_data
'''test_data = generate_data('cpu', 2000, 5, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_5_2.pt')
test_data = generate_data('cpu', 2000, 10, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_10_2.pt')
test_data = generate_data('cpu', 2000, 50, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_50_2.pt')
test_data = generate_data('cpu', 2000, 100, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_100_2.pt')
test_data = generate_data('cpu', 2000, 500, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_500_2.pt')
test_data = generate_data('cpu', 2000, 1000, 2,start=2)
torch.save(test_data, 'mamba/data/start_2/2000_1000_2.pt')'''

test_data_loc1 = 'data/start_2/2000_5_2.pt'
'''test_data_loc2 = 'data/start_2/2000_10_2.pt'
test_data_loc3 = 'data/start_2/2000_50_2.pt'
test_data_loc4 = 'data/start_2/2000_100_2.pt'
test_data_loc5 = 'data/start_2/2000_500_2.pt'
test_data_loc6 = 'data/start_2/2000_1000_2.pt'''

test_data1 = torch.load(test_data_loc1)
'''test_data2 = torch.load(test_data_loc2)
test_data3 = torch.load(test_data_loc3)
test_data4 = torch.load(test_data_loc4)
test_data5 = torch.load(test_data_loc5)
test_data6 = torch.load(test_data_loc6)'''
print(test_data1.shape)

#Remove start token
test_data1 = test_data1[:,:-1,:]
'''test_data2 = test_data2[:,:-1,:]
test_data3 = test_data3[:,:-1,:]
test_data4 = test_data4[:,:-1,:]
test_data5 = test_data5[:,:-1,:]
test_data6 = test_data6[:,:-1,:]'''
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

'''start_data = torch.rand(2000, 1, 2)

test_data1_rand = torch.cat((test_data1,start_data), dim=1)
test_data2_rand = torch.cat((test_data2,start_data), dim=1)
test_data3_rand = torch.cat((test_data3,start_data), dim=1)
test_data4_rand = torch.cat((test_data4,start_data), dim=1)
test_data5_rand = torch.cat((test_data5,start_data), dim=1)
test_data6_rand = torch.cat((test_data6,start_data), dim=1)

torch.save(test_data1_rand, 'data/start_rand/2000_5_2.pt')
torch.save(test_data2_rand, 'data/start_rand/2000_10_2.pt')
torch.save(test_data3_rand, 'data/start_rand/2000_50_2.pt')
torch.save(test_data4_rand, 'data/start_rand/2000_100_2.pt')
torch.save(test_data5_rand, 'data/start_rand/2000_500_2.pt')
torch.save(test_data6_rand, 'data/start_rand/2000_1000_2.pt')'''
