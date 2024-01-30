import torch

if torch.cuda.is_available():
    print("CUDA GPU is available")
    print(torch.cuda.get_device_name(0))  # Print GPU name
    print(torch.cuda.get_device_name(1))  # Print GPU name
    print(torch.cuda.get_device_name(2))  # Print GPU name
    print(torch.cuda.get_device_name(3))  # Print GPU name
else:
    print("CUDA GPU is not available")

