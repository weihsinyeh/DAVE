import torch

# 載入 checkpoint
checkpoint_path = "/tmp2/r13922043/DAVE/material/DAVE_3_shot.pth"
checkpoint = torch.load(checkpoint_path)

# 假設 checkpoint 包含 'epoch' 的 key
if "epoch" in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"best_val_ae: {checkpoint['best_val_ae']}")
else:
    print("Checkpoint does not contain 'epoch' information.")
