import torch
import numpy as np

# Form a tensor from a list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from list: \n {x_data} \n")

# Form a tensor from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from numpy: \n {x_np} \n")

# Form a tensor from other tensors
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor from x_data: \n {x_rand} \n")

# Form tensor with given shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor with shape {shape}: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Tensor attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")
print(f"Tensor: \n {tensor} \n")

# ------------------
# Tensor operations

# Move tensor to GPU if available
if torch.cuda.is_available():
    print("GPU is available")
    tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device} \n")

# Standard numpy-like indexing and slicing
# tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])