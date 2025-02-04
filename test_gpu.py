import torch

if torch.cuda.is_available():
    print(f"CUDA 可用 ✅, GPU 數量: {torch.cuda.device_count()}")
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("❌ PyTorch 無法使用 GPU，請檢查 CUDA 安裝及 docker 參數")
