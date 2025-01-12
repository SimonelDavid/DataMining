import os
import time
import psutil

def log_time(times, step_name):
  end = time.time()
  start = times[-1]
  times.append(end)
  print(f"[{step_name}] Time Taken: {end - start:.2f} seconds")
  return times

def log_resources(step_name):
  cpu = psutil.cpu_percent(interval=0.1)
  process = psutil.Process(os.getpid())
  memory = process.memory_info().rss / (1024 ** 2)
  print(f"[{step_name}] CPU Usage: {cpu}% | Memory Usage: {memory:.2f} MB")
