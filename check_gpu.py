import torch
print(torch.cuda.is_available())  # Prints True if GPU is available
torch.zeros(1).cuda()
