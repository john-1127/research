import torch
from chemprop.utils import load_checkpoint

checkpoint = torch.load("./output/model_classical/model.pt", map_location=torch.device('cpu'))  # 如果沒 GPU 就用 CPU

print(checkpoint.keys())
# 取出 state_dict
state_dict = checkpoint["state_dict"]

# 顯示所有 layer 及其 shape
for key, value in state_dict.items():
    print(f"Layer: {key} | Shape: {value.shape}")

args =checkpoint["args"]
model = load_checkpoint(
            path="./output/model_classical/model.pt", cuda=args.cuda
        )

model.load_state_dict(state_dict, strict=False)
