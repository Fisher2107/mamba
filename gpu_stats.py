import time
import argparse
import csv
from datetime import datetime
import pynvml
from threading import Thread , Event
from queue import Queue
import os

def get_gpu_count():
    return pynvml.nvmlDeviceGetCount()

def get_gpu_info(gpu_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    name = pynvml.nvmlDeviceGetName(handle)
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)  # Convert to GB
    return f"GPU {gpu_id}: {name}, Total Memory: {total_memory:.2f} GB"

class GPULogger:
    def __init__(self, gpu_id):
        pynvml.nvmlInit()
        self.event_queue = Queue()
        self.gpu_stats_queue = Queue()
        self.logging_thread = None
        self.writing_thread = None
        self.gpu_id = gpu_id
        self.stop_flag = Event()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    def get_gpu_stats(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert to Watts
        return util.gpu, mem_info.used / mem_info.total * 100, power

    def log_gpu_stats(self, interval_ms):
        while not self.stop_flag.is_set():
            timestamp = datetime.now().isoformat()
            utilization, memory_usage, power = self.get_gpu_stats()
            self.gpu_stats_queue.put((timestamp, utilization, memory_usage, power))
            self.stop_flag.wait(timeout=interval_ms / 1000)

    def write_gpu_stats(self, output_file):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'GPU Utilization', 'Memory Usage', 'Power Usage (W)'])
            
            while not (self.stop_flag.is_set() and self.gpu_stats_queue.empty()):
                stats = []
                while not self.gpu_stats_queue.empty():
                    stats.append(self.gpu_stats_queue.get())
                
                if stats:
                    writer.writerows(stats)
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
                
                self.stop_flag.wait(timeout=1)  # Wait for 1 second before next write

    def start_gpu_logging(self, output_file, interval_ms):
        self.stop_flag.clear()
        self.logging_thread = Thread(target=self.log_gpu_stats, args=(interval_ms,))
        self.writing_thread = Thread(target=self.write_gpu_stats, args=(output_file,))
        self.logging_thread.start()
        self.writing_thread.start()

    def log_event(self, event_name):
        self.event_queue.put((datetime.now().isoformat(), event_name))

    def stop_logging(self):
        self.stop_flag.set()
        if self.logging_thread:
            self.logging_thread.join()
        if self.writing_thread:
            self.writing_thread.join()
        pynvml.nvmlShutdown()

    def export_events(self, output_file):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Event'])
            writer.writerows(list(self.event_queue.queue))

if __name__ == '__main__':
    #Take in one argument
    parser = argparse.ArgumentParser(description='Get GPU information')
    parser.add_argument('--gpu_id', type=int, help='The GPU ID to get information for')
    gpu_id = parser.parse_args().gpu_id
    pynvml.nvmlInit()
    # Get and print the number of GPUs
    gpu_count = get_gpu_count()

    # Open a file in write mode
    with open('logs/gpu_info.txt', 'w') as f:
        f.write(f"Number of GPUs detected: {gpu_count}\n")

        # Write information for each GPU to the file
        for i in range(gpu_count):
            f.write(get_gpu_info(i) + '\n')
        f.write(str(gpu_id))

    pynvml.nvmlShutdown()
