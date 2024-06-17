import torch

def greedy_tsp(cities):
    """
    Solve TSP using a greedy approach.

    Args:
    cities: Tensor of shape (batch_size, sequence_length, 2) representing the cities.

    Returns:
    A tensor of shape (batch_size, sequence_length) representing the tour.
    """
    batch_size, sequence_length, _ = cities.size()

    # Initialize tour starting from the first city
    tour = torch.zeros(batch_size, sequence_length, dtype=torch.long)

    # Distance matrix
    dist = torch.cdist(cities, cities, p=2)

    # Set large value for distance to self
    batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, sequence_length)
    sequence_indices = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1)
    dist[batch_indices, sequence_indices, sequence_indices] = float('inf')

    for i in range(1, sequence_length):
        last_city = tour[:, i-1]

        # Get the distance to the last city in the tour
        dist_to_last = dist[torch.arange(batch_size), last_city]

        # Find the closest city
        closest_city = dist_to_last.min(dim=-1).indices

        # Add the closest city to the tour
        tour[:, i] = closest_city

        # Set distance to the chosen city to infinity to avoid choosing it again
        dist[torch.arange(batch_size), closest_city] = float('inf')

    return tour
