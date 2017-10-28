import numpy as np 
import torch

'''
Tensors are similar to numpy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
'''
arr = [[1,2],[3,4]]
np_arr = np.array(arr)
tch_arr = torch.Tensor(arr)
np.ones((2,2))
torch.ones((2,2))
'''
If we give seed then random function genreates same random number always. Seeds are used for reproducibility.
'''
np.random.seed(0)
np.random.rand(2,2)
torch.rand(2,2)
torch.manual_seed(0)

'''
if we are using GPU we can set seed in the following way
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(0)
'''
np_arr = np.array(arr, dtype = np.int32)
torch.from_numpy(np_arr)
torch_to_numpy = torch.numpy()

'''
Note in above code np_array elements must be from dtypes that torch support that are
#np types : torch types#
float32 : float tensor
float64 : double tensor
double : double tensor
int32 : integer tensor
int64 : long tensor
uint8 : byte tensor
'''

'''
Torch tensors at CPU vs GPU
Let we have a torch tensor named tch_arr
'''

if torch.cuda.is_available():
	tch_arr.cuda()

#To bring bak from GPU to CPU
tch_arr.cpu()


'''
TENSOR OPERATIONS
'''
tch_arr.size()
#Convet to a specific size
tch_arr.view(mention_any_size)
tch_arr.view(mention_any_size).size()
#Note _ after add denotes that change c also i.e. add in place
tch_arr.add_(another_tensor/th_arrr)
'''
Functions
>>Element Wise
#add,sub,mul,div
>>Statitical
#mean,std
#tch_arr.mean(dim=0)
What does dim tell 
Let size(shape of tensor) is [2,10]
dim = 0 is index 0 of above tuple, so it will pick 2 elements each and return vector of size 10
dim = 1 is index 1 of above tuple, so it will pick 10 elements each and return vector of size 2 
'''