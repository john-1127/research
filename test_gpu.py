import torch

if torch.cuda.is_available():
    print(f"CUDA ✅, Numbers of GPU: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("❌ PyTorch can't use GPU")

device = torch.device('cuda')
torch.rand(10).to(device)
