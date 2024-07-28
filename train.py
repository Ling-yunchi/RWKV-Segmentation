import torch

from model import VRWKV_Adapter

t = torch.randn(1, 3, 224, 224).cuda()

model = VRWKV_Adapter(deform_num_heads=8, interaction_indexes=[[0,2],[3,5],[6,8],[9,11]])
model = model.cuda()

output = model(t)

print(output)
