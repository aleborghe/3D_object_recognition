from cl_orion import NTXentLoss
import torch

t = 1

z_q = torch.tensor([[4, 2, 3], [4, 0, 6]], dtype = torch.float32, requires_grad=True)
z_p = torch.tensor([[7, 2, 9], [10, 11, 12]], dtype = torch.float32, requires_grad=True)

loss = NTXentLoss(temperature=t)
l = loss(z_q, z_p)
print(l)
l.backward()
print(z_q.grad)
i = torch.arange(2)
a = torch.tensor([[1, 2, 3, 4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
b = a[i,   i + 2]   # sim(q_i,  q⁺_i)
print(b)