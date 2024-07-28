import torch

from model import VRWKV_Adapter

t = torch.randn(1, 3, 224, 224).cuda()

model = VRWKV_Adapter()
model = model.cuda()

output = model(t)

print(output.shape)
