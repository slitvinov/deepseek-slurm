import torch
if torch.cuda.is_available():
   for i in range(torch.cuda.device_count()):
      p = torch.cuda.get_device_properties(i)
      print(f"{p.name}; {p.total_memory >> 30}Gb; {p.uuid}")
else:
   print("CUDA is not available")
