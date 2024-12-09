import torch
from safetensors.torch import load_file


pt_state_dict = load_file("/data/experiment_data/junda/chatdoctor/llama-7b-32k-mimic4-2/checkpoint-2/adapter_model/adapter_model.safetensors", device="cpu")
torch.save(pt_state_dict, "/data/experiment_data/junda/chatdoctor/llama-7b-32k-mimic4-2/checkpoint-2/adapter_model/adapter_model.bin")