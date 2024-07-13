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

    def get_gpu_stats(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, mem_info.used / mem_info.total * 100

    def log_gpu_stats(self, duration_seconds, interval_ms):
        end_time = time.time() + duration_seconds
        while time.time() < end_time and not self.stop_flag.is_set():
            timestamp = datetime.now().isoformat()
            utilization, memory_usage = self.get_gpu_stats()
            self.gpu_stats_queue.put((timestamp, utilization, memory_usage))
            time.sleep(interval_ms / 1000)

    def write_gpu_stats(self, output_file):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'GPU Utilization', 'Memory Usage'])
            
            while not self.stop_flag.is_set():
                try:
                    stats = self.gpu_stats_queue.get(timeout=1)  # Wait for up to 1 second for an item
                except queue.Empty:
                    continue
                except Exception as e:
                    self.log_event(f"Error writing GPU stats: {e}")
                    break
                
                writer.writerow(stats)
                csvfile.flush()
                os.fsync(csvfile.fileno())  # Ensure data is written to disk

    def start_gpu_logging(self, output_file, duration_seconds, interval_ms):
        self.stop_flag.clear()
        self.logging_thread = Thread(target=self.log_gpu_stats, args=(duration_seconds, interval_ms))
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

    def get_events(self):
        return list(self.event_queue.queue)
    
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
