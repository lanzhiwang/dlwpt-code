# PyTorch Hook

## Module

* https://pytorch.org/docs/2.5/generated/torch.nn.Module.html

```python
"""
class torch.nn.Module(*args, **kwargs)

Module.register_forward_pre_hook()  # The hook will be called every time before forward() is invoked.
Module.register_forward_hook()  # The hook will be called every time after forward() has computed an output.

Module.register_full_backward_pre_hook()  # The hook will be called every time the gradients for the module are computed.
Module.register_full_backward_hook()  # The hook will be called every time the gradients with respect to a module are computed.
Module.register_backward_hook()  # This function is deprecated

Module.register_load_state_dict_pre_hook()
Module.register_load_state_dict_post_hook()

Module.register_state_dict_pre_hook()
Module.register_state_dict_post_hook()
"""

cat << EOF > temp.py

import torch
import torch.nn as nn

# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 33, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create an instance of the CNN model
model = CNN()

feats = {}

# hook(module, input, output) -> None or modified output
def hook_func(module, input, output):
    # print("module:", module)
    # print("input:", input.detach().shape)
    # print("output:", output.detach().size())
    feats['feat'] = output.detach()

model.pool.register_forward_hook(hook_func)

x= torch.randn(20, 16, 50, 100)
output = model(x)
# print(feats['feat'])
print(feats['feat'].shape)

EOF

# (.env) @lanzhiwang ➜ /workspaces/dlwpt-code (master) $ python temp.py
# torch.Size([20, 33, 25, 50])
# (.env) @lanzhiwang ➜ /workspaces/dlwpt-code (master) $

```

## torch.Tensor

* https://pytorch.org/docs/2.5/tensors.html

```python
"""
class torch.Tensor

Tensor.register_hook(hook)  # Registers a backward hook.
    hook(grad) -> Tensor or None

Tensor.register_post_accumulate_grad_hook(hook)  # Registers a backward hook that runs after grad accumulation.
    hook(param: Tensor) -> None
"""

>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> v.grad

>>> v.backward()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/autograd/__init__.py", line 340, in backward
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/autograd/__init__.py", line 198, in _make_grads
    raise RuntimeError(
RuntimeError: grad can be implicitly created only for scalar outputs

>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v.grad
tensor([1., 2., 3.])
>>>
>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v.grad
tensor([2., 4., 6.])
>>> h.remove()  # removes the hook
>>>

>>> tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> output = tensor.sum()
>>> output
tensor(6., grad_fn=<SumBackward0>)
>>> output.backward()
>>> tensor.grad
tensor([1., 1., 1.])
>>> output.grad
<stdin>:1: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
>>>
>>> tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> h = tensor.register_hook(lambda grad: grad * 2)
>>> output = tensor.sum()
>>> output
tensor(6., grad_fn=<SumBackward0>)
>>> output.backward()
>>> tensor.grad
tensor([2., 2., 2.])
>>> output.grad
>>>

#################################################

>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v
tensor([0., 0., 0.], requires_grad=True)
>>> v.grad
tensor([1., 2., 3.])
>>>
>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad))
>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v
tensor([1., 2., 3.], requires_grad=True)
>>> v.grad
tensor([1., 2., 3.])
>>>

```

