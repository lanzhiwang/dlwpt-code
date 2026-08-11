# torch.nn

* https://pytorch.org/docs/2.5/nn.html

These are the basic building blocks for graphs:
以下是图形的基本构建块:

* Containers 	器皿

* Convolution Layers 	卷积层

* Pooling layers 	池化层

* Padding Layers 	填充层

* Non-linear Activations (weighted sum, nonlinearity) 	非线性激活(加权和、非线性)

* Non-linear Activations (other) 	非线性激活(其他)

* Normalization Layers 	归一化图层

* Recurrent Layers 	循环层

* Transformer Layers 	变压器层

* Linear Layers 	线性层

* Dropout Layers 	Dropout 图层

* Sparse Layers 	稀疏层

* Distance Functions 	距离函数

* Loss Functions 	损失函数

* Vision Layers

* Shuffle Layers 	随机排列图层

* DataParallel Layers (multi-GPU, distributed) 	DataParallel Layers(多GPU, 分布式)

* Utilities 	公用

* Quantized Functions 	量化函数

* Lazy Modules Initialization 	惰性模块初始化
  * Aliases

---

* Buffer 	A kind of Tensor that should not be considered a model parameter.
			一种不应被视为模型参数的 Tensor.

* Parameter 	A kind of Tensor that is to be considered a module parameter.
				一种 Tensor, 将被视为 module 参数.

* UninitializedParameter 	A parameter that is not initialized.
							未初始化的参数.

* UninitializedBuffer 	A buffer that is not initialized.
						未初始化的缓冲区.

---

## Containers
器皿

* Module 	Base class for all neural network modules.
			所有神经网络模块的基类.

* Sequential 	A sequential container.
				顺序容器.

* ModuleList 	Holds submodules in a list.
				将子模块保存在列表中.

* ModuleDict 	Holds submodules in a dictionary.
				在字典中保存子模块.

* ParameterList 	Holds parameters in a list.
					在列表中保存参数.

* ParameterDict 	Holds parameters in a dictionary.
					在字典中保存参数.

---

### Global Hooks For Module
Module 的全局钩子

* register_module_forward_pre_hook 	Register a forward pre-hook common to all modules.
									注册一个所有模块通用的正向预钩子.

* register_module_forward_hook 	Register a global forward hook for all the modules.
								为所有模块注册一个全局 forward hook.

* register_module_backward_hook 	Register a backward hook common to all the modules.
									注册一个所有模块通用的反向钩子.

* register_module_full_backward_pre_hook 	Register a backward pre-hook common to all the modules.
											注册一个所有模块通用的向后预钩子.

* register_module_full_backward_hook 	Register a backward hook common to all the modules.
										注册一个所有模块通用的反向钩子.

* register_module_buffer_registration_hook 	Register a buffer registration hook common to all modules.
											注册一个所有模块通用的缓冲区注册钩子.

* register_module_module_registration_hook 	Register a module registration hook common to all modules.
											注册一个所有模块通用的模块注册钩子.

* register_module_parameter_registration_hook 	Register a parameter registration hook common to all modules.
												注册一个所有模块通用的参数注册钩子.

---

## nn.Linear

```python
"""
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
"""

>>> import torch
>>> import torch.nn as nn
>>> m = nn.Linear(20, 30)
>>> m
Linear(in_features=20, out_features=30, bias=True)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> output.size()
torch.Size([128, 30])
>>>
>>> input = torch.randn(128, 21)
>>> output = m(input)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dlwpt-code/.env/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x21 and 20x30)
>>>
```

## nn.MSELoss

```python
>>> loss = nn.MSELoss()
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randn(3, 5)
>>> output = loss(input, target)
>>> output.backward()
>>> output
tensor(1.1493, grad_fn=<MseLossBackward0>)
>>>
```

## nn.Tanh

```python
>>> m = nn.Tanh()
>>> input = torch.randn(3, 5)
>>> output = m(input)
>>> output
tensor([[-0.8764, -0.6256, -0.7935, -0.0895, -0.4195],
        [ 0.0686, -0.7900,  0.2948,  0.3193, -0.9256],
        [-0.3045,  0.9412, -0.0325,  0.8426,  0.0034]])
>>> output.size()
torch.Size([3, 5])
>>>

```

## nn.Hardtanh

## nn.Sigmoid

## nn.Softplus

## nn.ReLU

## nn.LeakyReLU

## nn.Tanhshrink

## nn.Softshrink

## nn.Hardshrink

## nn.Sequential

```python
# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

```


## nn.Conv1d、nn.Conv2d、nn.Conv3d

```python
"""
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
Input: (N, C, L) or (C, L)
(batch, channel, Length)

class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
Input: (N, C, H, W) or (C, H, W)
(batch, channel, Height, Width)

class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
Input: (N, C, D, H, W) or (C, D, H, W)
(batch, channel, Depth, Height, Width)

"""
# in_channels = 16
>>> m = nn.Conv1d(16, 33, 3, stride=2)
>>> input = torch.randn(20, 16, 50)
>>> output = m(input)

##############################################

# in_channels = 16
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)  # in_channels = 16
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)  # input 的形状是 (N, C, H, W), C = in_channels = 16
>>> output = m(input)

##############################################

# in_channels = 16
>>> # With square kernels and equal stride
>>> m = nn.Conv3d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
>>> input = torch.randn(20, 16, 10, 50, 100)
>>> output = m(input)

```

## nn.Embedding

* https://zhuanlan.zhihu.com/p/647536930

```python
"""
class torch.nn.Embedding(
    num_embeddings,  # (int) – size of the dictionary of embeddings
    embedding_dim,  # (int) – the size of each embedding vector
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None,
    _freeze=False,
    device=None,
    dtype=None
)

classmethod from_pretrained(
    embeddings,
    freeze=True,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False
)

"""





>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])


>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.3489,  0.4710, -0.8160],
        [-0.0586,  0.2668, -0.1658],
        [-0.3928, -0.1596,  0.5922],
        [-0.2360, -0.3142, -1.3988],
        [-1.9123, -1.1718, -1.4204],
        [ 0.3745, -0.0303, -1.5870],
        [-0.7438, -0.8898,  0.1122],
        [-0.3444, -0.4155,  0.7549],
        [-1.1764, -0.5080, -0.8645]], requires_grad=True)
>>> input = torch.LongTensor([[0, 2, 0, 5]])
>>> embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [-0.0586,  0.2668, -0.1658],
         [ 0.0000,  0.0000,  0.0000],
         [-1.9123, -1.1718, -1.4204]]], grad_fn=<EmbeddingBackward0>)
>>>

>>> padding_idx = 0
>>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.5989, -0.5338, -1.6830],
        [-0.3169,  0.9848,  0.9241]], requires_grad=True)
>>> with torch.no_grad():
...     embedding.weight[padding_idx] = torch.ones(3)
...
>>> embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.5989, -0.5338, -1.6830],
        [-0.3169,  0.9848,  0.9241]], requires_grad=True)
>>> input = torch.LongTensor([[1, 2, 0]])
>>> embedding(input)
tensor([[[-0.5989, -0.5338, -1.6830],
         [-0.3169,  0.9848,  0.9241],
         [ 1.0000,  1.0000,  1.0000]]], grad_fn=<EmbeddingBackward0>)
>>>


>>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
>>> weight
tensor([[1.0000, 2.3000, 3.0000],
        [4.0000, 5.1000, 6.3000]])
>>> weight.shape
torch.Size([2, 3])
>>> embedding = nn.Embedding.from_pretrained(weight)
>>> embedding.weight
Parameter containing:
tensor([[1.0000, 2.3000, 3.0000],
        [4.0000, 5.1000, 6.3000]])
>>> input = torch.LongTensor([1, 0])
>>> embedding(input)
tensor([[4.0000, 5.1000, 6.3000],
        [1.0000, 2.3000, 3.0000]])
>>>

```

