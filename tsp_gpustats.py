import pynvml
import argparse


#Take in one argument
parser = argparse.ArgumentParser(description='Get GPU information')
parser.add_argument('--gpu_id', type=int, help='The GPU ID to get information for')
gpu_id = parser.parse_args().gpu_id
pynvml.nvmlInit()

def get_gpu_count():
    return pynvml.nvmlDeviceGetCount()

def get_gpu_info(gpu_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    name = pynvml.nvmlDeviceGetName(handle)
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)  # Convert to GB
    return f"GPU {gpu_id}: {name}, Total Memory: {total_memory:.2f} GB"

# Get and print the number of GPUs
gpu_count = get_gpu_count()

# Open a file in write mode
with open('gpu_info.txt', 'w') as f:
    f.write(f"Number of GPUs detected: {gpu_count}\n")

    # Write information for each GPU to the file
    for i in range(gpu_count):
        f.write(get_gpu_info(i) + '\n')
    f.write(str(gpu_id))

pynvml.nvmlShutdown()