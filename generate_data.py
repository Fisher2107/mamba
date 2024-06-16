import torch
from model import generate_data
test_data = generate_data('cpu', 2000, 5, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_5_2.pt')
test_data = generate_data('cpu', 2000, 10, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_10_2.pt')
test_data = generate_data('cpu', 2000, 50, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_50_2.pt')
test_data = generate_data('cpu', 2000, 100, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_100_2.pt')
test_data = generate_data('cpu', 2000, 500, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_500_2.pt')
test_data = generate_data('cpu', 2000, 1000, 2,start=0)
torch.save(test_data, 'mamba/data/start_0/test_rand_2000_1000_2.pt')