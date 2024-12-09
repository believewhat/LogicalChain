import torch
import subprocess
import time

def allocate_memory_if_needed(target_memory=51300, poll_interval=10, memory_buffer=500):
    allocated_tensors = {}
    
    while True:
        # Check free memory using PyTorch directly
        free_memory_list = get_free_memory_per_gpu()

        for i, free_memory in enumerate(free_memory_list):
            print(f"GPU {i}: Current free memory: {free_memory} MB")
            
            if free_memory > target_memory:
                memory_to_allocate = min(free_memory - memory_buffer, free_memory - target_memory)
                if memory_to_allocate > 0:
                    memory_to_allocate_bytes = memory_to_allocate * 1024 * 1024
                    print(f"GPU {i}: Allocating {memory_to_allocate} MB.")
                    
                    try:
                        device = torch.device(f'cuda:{i}')
                        allocated_tensors[i] = torch.empty(memory_to_allocate_bytes, dtype=torch.uint8, device=device)
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"GPU {i}: Out of memory error: {e}")
        
        time.sleep(poll_interval)

def get_free_memory_per_gpu():
    free_memory_list = []
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory_mb = free_memory // (1024 * 1024)
        free_memory_list.append(free_memory_mb)
    return free_memory_list

# Run the function
allocate_memory_if_needed()
