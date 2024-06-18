import torch
import itertools

def greedy_tsp(cities, device='cpu'):
    """
    Solve TSP using a greedy approach.

    Args:
    cities: Tensor of shape (batch_size, sequence_length+1, 2) representing the cities.

    Returns:
    A tensor of shape (batch_size, sequence_length) representing the tour.
    """
    cities = cities[:, :-1, :]
    bsz, city_count, _ = cities.shape
    arange_vec = torch.arange(bsz, device=device).unsqueeze(-1)
    visited = torch.zeros(bsz, city_count, dtype=torch.bool, device=device)
    tour = torch.zeros(bsz, city_count, dtype=torch.long, device=device)
    tour_length = torch.zeros(bsz, device=device)
    current_city = torch.zeros(bsz, 2, device=device)
    for i in range(city_count):
        if i == 0:
            current_city = cities[arange_vec, i, :]
            visited[arange_vec, i] = True
            
        else:
            current_city = cities[arange_vec, tour[arange_vec, i-1], :]
            visited[arange_vec, tour[arange_vec, i-1]] = True
        if i == city_count - 1:
            visited[arange_vec, 0] = False

        distances = torch.zeros(bsz, city_count, device=device)
        for j in range(bsz):
            for k in range(city_count):
                if visited[j, k]:
                    distances[j, k] = float('inf')
                else:
                    distances[j, k] = torch.norm(current_city[j] - cities[j, k], dim=-1)
        #print('distances are ',distances)
        next_city = torch.argmin(distances, dim=-1)
        tour[arange_vec, i] = next_city.unsqueeze(-1)
        '''print('tour is ',tour)
        print(next_city.shape)
        print(distances[arange_vec.squeeze(-1), next_city].shape)'''
        tour_length += distances[arange_vec.squeeze(-1), next_city]
    return tour_length.mean(),tour

def compute_tour_length(x, tour,remove_start_token=True): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, city_count+1, 2) batch of tsp tour instances
             tour of size (bsz, city_count) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    if remove_start_token:
        x = x[:,:-1,:]
    bsz = x.shape[0]
    arange_vec = torch.arange(bsz, device=x.device).unsqueeze(-1)
    tour = tour.to(x.device)

    # Get the cities in the order of the tour
    ordered_cities = x[arange_vec, tour, :] # size(ordered_cities)=(bsz, city_count, 2)

    # Compute the differences between each pair of consecutive cities
    diffs = ordered_cities[:, 1:, :] - ordered_cities[:, :-1, :] # size(diffs)=(bsz, city_count-1, 2)

    # Compute the distance between each pair of consecutive cities
    distances = torch.sqrt(torch.sum(diffs**2, dim=2)) # size(distances)=(bsz, city_count-1)

    # Add the distance from the last city to the first
    distances = torch.cat([distances, torch.norm(ordered_cities[:, 0, :] - ordered_cities[:, -1, :], dim=1).unsqueeze(-1)], dim=1)

    # Sum the distances to get the total length of each tour
    L = torch.sum(distances, dim=1)

    return L

def exact_solver(cities,device='cpu'):
    cities = cities[:, :-1, :]
    bsz, city_count, _ = cities.shape
    arange_vec = torch.arange(bsz, device=device).unsqueeze(-1)
    permutations = torch.tensor(list(itertools.permutations(range(city_count))), device=device).unsqueeze(1)
    permutations = permutations.repeat(1, 2000, 1)  # Repeat it 2000 times along the second dimension to make it (120, 2000, 5)
    #print(permutations.shape)
    all_tour_lengths = torch.zeros(bsz, len(permutations), device=device)
    #print(all_tour_lengths.shape)
    for i in range(permutations.shape[0]):
        tour_length = compute_tour_length(cities, permutations[i], remove_start_token=False)
        #print(tour_length)
        all_tour_lengths[:, i] = tour_length
    #print(all_tour_lengths)
    #print(all_tour_lengths.shape)
    tour_length,_ = torch.min(all_tour_lengths, dim=1)
    #print(tour_length)
    return tour_length.mean()
    

test_input = torch.load('data/start_2/test_rand_2000_5_2.pt')
print(greedy_tsp(test_input))
print(exact_solver(test_input))