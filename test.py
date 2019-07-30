import torch
import numpy as np


# array_list=[]
# for i in range(0,12):
#     array = np.zeros((1024, 3))
#     array=np.expand_dims(array.transpose(),axis=2)
#     array=torch.from_numpy(array)
#     array_list.append(array)
#
# result=torch.stack(array_list)
# print(result.size())
# print(result.view(result.size(0)*result.size(1),-1).shape)

# print(torch.from_numpy(np.empty((20,0))).shape)


x=torch.arange(15).view(5,3)
x_mean=torch.mean(x,dim=0,keepdim=True)
x_mean0=torch.mean(x,dim=1,keepdim=True)
# print(torch.mean(x,dim=1))
# print(torch.stack(array_list).unsqueeze(3).size())
# print(np.expand_dims(array.transpose(),axis=2).shape)
