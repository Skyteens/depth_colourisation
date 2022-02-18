import torch

image = torch.ones(16, 3, 256, 256, dtype=torch.float, requires_grad=False)
test = image *0.5
img2 = torch.ones(16, 3, 256, 256, dtype=torch.float, requires_grad=False)
test2 = image *0.3


fun = torch.add(test,test2 )
print(fun.shape)
print(fun[0][0][5,5])