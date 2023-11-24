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
tensor = torch.rand(10, 4)
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

# Joining tensors
# torch.cat takes a list of tensors to concatenate together
t1 = torch.cat([tensor, tensor], dim=1)
print(f"Concatenate by columns: \n {t1} \n")

# Arithmetic operations
tensor2 = torch.from_numpy(np.array([[10.0, 15.0, 20.0], [30.0, 35.0, 40.0]]))
print(f"Tensor 2: \n {tensor2} \n")
if torch.cuda.is_available():
    print("Moving tensor2 to GPU")
    tensor2 = tensor2.to('cuda')
# Matrix multiplication
y1 = tensor2 @ tensor2.T
y2 = tensor2.matmul(tensor2.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor2, tensor2.T, out=y3)
print("Matrix multiplication: \n")
print(f"y1: \n {y1} \n")
print(f"y2: \n {y2} \n")
print(f"y3: \n {y3} \n")

# Element-wise product
z1 = tensor2 * tensor2
z2 = tensor2.mul(tensor2)
z3 = torch.rand_like(tensor2)
torch.mul(tensor2, tensor2, out=z3)
print("Element-wise product: \n")
print(f"z1: \n {z1} \n")
print(f"z2: \n {z2} \n")
print(f"z3: \n {z3} \n")

# Single-element tensors
agg = tensor2.sum()
agg_item = agg.item()
print(f"agg: \n {agg} \n")
print(f"agg_item: \n {agg_item} \n")

# In-place operations (operations that have a _ suffix)
print("In-place operation of adding 5 to tensor2: \n")
tensor2.add_(5)
print(f"tensor2: \n {tensor2} \n")

# Bridge with NumPy
# Tensor to NumPy array
tensor3 = torch.ones(5) # tensor3 is on CPU
print(f"tensor3: \n {tensor3} \n")
numpy_array2 = tensor3.numpy() # can only be done on CPU
print(f"numpy_array2: \n {numpy_array2} \n")
# A change in tensor3 reflects in numpy_array2
print("Adding 1 to tensor3: \n")
tensor3.add_(1)
print(f"tensor3: \n {tensor3} \n")
print(f"numpy_array2: \n {numpy_array2} \n")
# A change in numpy_array2 reflects in tensor3
print("Setting numpy_array2 to all zeros: \n")
numpy_array2 = np.zeros(5)
tensor3 = torch.from_numpy(numpy_array2)
print(f"tensor3: \n {tensor3} \n")
print(f"numpy_array2: \n {numpy_array2} \n")

