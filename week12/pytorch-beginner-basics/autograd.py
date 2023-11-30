import torch

# Create a computational graph for a one-layer neural network
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Show reference to backward propagation function, i.e. gradient function
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Compute gradients
loss.backward()
print("Gradient of w:")
print(w.grad)
print("Gradient of b:")
print(b.grad)

# Disable gradient tracking --
# There are reasons you might want to disable gradient tracking:
#   - To mark some parameters in your neural network at frozen parameters.
#   - To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
z = torch.matmul(x, w)+b
print("z requires grad:", z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print("z requires grad:", z.requires_grad)

# Use detach() to achieve the same effect
z = torch.matmul(x, w)+b
z_det = z.detach()
print("z_det requires grad:", z_det.requires_grad)

optional_reading = False
# Optional reading: Tensor gradients and Jacobian products
if optional_reading:
    print("\nOptional reading: Tensor gradients and Jacobian products")
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp+1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")