import numpy as np
import torch

a=np.array([1,2,3,4])
a=torch.from_numpy(a)

b=np.array([2,2,2,2])
b=torch.from_numpy(b)

print(np.dot(a,b))

m=[[1,2,3],[4,5,6]]
m=torch.tensor(m)
x,y=m.shape
print(x,y)