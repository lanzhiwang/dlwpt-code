## mean

* https://pytorch.org/docs/2.5/generated/torch.mean.html

```python
"""
torch.mean(input, *, dtype=None) -> Tensor

Returns the mean value of all elements in the input tensor. Input must be floating point or complex.
返回输入张量中所有元素的平均值. 输入必须是浮点数或复数.

Parameters
input (Tensor) – the input tensor, either of floating point or complex dtype

Keyword Arguments
dtype (torch.dtype, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None.
返回张量的所需数据类型. 如果指定, 则在执行之前将输入张量强制转换为 dtype. 这对于防止数据类型溢出非常有用. 默认值: None.
"""

>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.2294, -0.5481,  1.3288]])
>>> torch.mean(a)
tensor(0.3367)

"""
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None) -> Tensor

Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
返回给定维度 dim 中输入张量的每一行的平均值. 如果 dim 是维度列表, 则 reduce 所有维度.

If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
如果 keepdim 为 True, 则输出张量的大小与输入相同, 但在维度 dim 中的大小为 1. 否则, dim 会被挤压(参见 torch.squeeze()), 导致输出张量的维度减少 1 个(或 len(dim)).

Parameters
input (Tensor) – the input tensor.
dim (int or tuple of ints) – the dimension or dimensions to reduce.
keepdim (bool) – whether the output tensor has dim retained or not.

Keyword Arguments
dtype (torch.dtype, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None.
返回张量的所需数据类型. 如果指定, 则在执行作之前将输入张量强制转换为 dtype. 这对于防止数据类型溢出非常有用. 默认值: None.
out (Tensor, optional) – the output tensor.

See also
torch.nanmean() computes the mean value of non-NaN elements.
计算非 NaN 元素的平均值.
"""

>>> a = torch.randn(3, 4)
>>> a
tensor([[-1.0658,  0.7092, -0.1809, -0.2242],
        [ 0.2784,  0.0197, -0.2646, -0.9813],
        [ 0.6937,  0.3176, -0.6098, -0.2039]])

>>> torch.mean(a, 0)
tensor([-0.0312,  0.3489, -0.3518, -0.4698])

>>> torch.mean(a, 1)
tensor([-0.1904, -0.2369,  0.0494])
>>>

# torch.mean(a, 0)
>>> torch.mean(a[:, 0])  # dim 参数指定哪个维度, 那个维度就全选
tensor(-0.0312)
>>> torch.mean(a[:, 1])
tensor(0.3489)
>>> torch.mean(a[:, 2])
tensor(-0.3518)
>>> torch.mean(a[:, 3])
tensor(-0.4698)

# torch.mean(a, 1)
>>> torch.mean(a[0, :])
tensor(-0.1904)
>>> torch.mean(a[1, :])
tensor(-0.2369)
>>> torch.mean(a[2, :])
tensor(0.0494)
>>>

>>> a = torch.randn(3, 4, 5)
>>> a
tensor([[[ 0.1734, -0.9008,  0.6201, -1.4654,  0.6282],
         [ 1.8499, -0.0666,  0.5799, -0.3610, -0.0646],
         [ 0.9795,  0.0151,  0.1907, -0.9491,  1.3667],
         [ 1.0962, -0.7383,  1.1137,  0.0894,  0.6311]],

        [[-1.6416,  0.3586,  1.5026, -0.8274,  0.0360],
         [ 1.0356, -0.7951, -1.0857,  0.4790, -1.1847],
         [ 0.7071, -0.7747, -0.1511,  0.0646,  1.2306],
         [ 0.5951,  0.1961, -0.5167,  0.3533, -0.6123]],

        [[-0.9465, -0.4132, -1.5208, -2.1440, -0.4017],
         [-1.7500,  0.5246, -0.3066, -1.6945, -1.4194],
         [-0.2433,  0.8501, -0.5046,  1.9856,  0.7863],
         [ 0.6825, -0.2934,  0.0348,  0.2896, -0.2885]]])
>>> torch.mean(a, 0)
tensor([[-0.8049, -0.3185,  0.2006, -1.4789,  0.0875],
        [ 0.3785, -0.1124, -0.2708, -0.5255, -0.8896],
        [ 0.4811,  0.0302, -0.1550,  0.3670,  1.1279],
        [ 0.7913, -0.2786,  0.2106,  0.2441, -0.0899]])
>>> torch.mean(a, 1)
tensor([[ 1.0248, -0.4226,  0.6261, -0.6715,  0.6403],
        [ 0.1741, -0.2538, -0.0627,  0.0173, -0.1326],
        [-0.5643,  0.1670, -0.5743, -0.3908, -0.3308]])
>>> torch.mean(a, 2)
tensor([[-0.1889,  0.3875,  0.3206,  0.4384],
        [-0.1144, -0.3102,  0.2153,  0.0031],
        [-1.0852, -0.9292,  0.5748,  0.0850]])

# torch.mean(a, 0)
>>> torch.mean(a[:, 0, 0])
tensor(-0.8049)
>>> torch.mean(a[:, 0, 1])
tensor(-0.3185)
>>>

>>> img_t = torch.randn(3, 5, 5)
>>> torch.mean(img_t, -3)
tensor([[ 1.0150,  0.3355,  0.9295,  0.0713, -0.2461],
        [-0.2707,  0.7579,  0.4891,  0.0578, -0.6774],
        [-0.6965, -0.0115, -0.5458,  1.0249, -0.3107],
        [-0.7189, -0.6133, -0.7907, -0.6978,  0.3195],
        [-1.3997, -0.2835, -0.2878, -0.0780, -0.0569]])
>>>
>>> img_t = torch.randn(3, 5, 5)
>>> img_t
tensor([[[-0.2513,  0.0881, -1.9254,  1.2597,  0.1620],
         [-0.5259, -1.6570,  0.0340, -1.7161, -0.5263],
         [ 0.4108,  0.4584, -0.2993, -0.0337,  1.1124],
         [-1.8439,  0.2942, -1.8753,  1.1862,  0.2709],
         [-0.1729,  1.2340, -0.4306, -0.6829, -0.1983]],

        [[ 0.6381,  0.4470, -0.1816,  0.9032, -1.3376],
         [-0.0734, -0.1686,  0.4084,  1.9673, -0.3155],
         [ 0.6503, -0.4344, -1.6969, -1.2072,  0.6676],
         [ 2.3876,  0.1810,  1.1811, -0.0494,  2.0169],
         [ 1.6879,  0.8552, -0.5129,  0.0370, -1.5014]],

        [[-0.5907, -0.5521,  0.1566, -0.2139, -0.2300],
         [ 2.3604, -0.4079,  0.3606,  0.1222, -0.5677],
         [ 0.2689, -1.4144,  0.6207,  0.0631,  2.2525],
         [ 0.6691,  0.5957,  1.1622,  0.5568, -2.6317],
         [-0.2384,  1.1975,  0.2165, -1.7551,  0.6989]]])
>>> torch.mean(img_t, -3)
tensor([[-0.0680, -0.0057, -0.6501,  0.6497, -0.4686],
        [ 0.5870, -0.7445,  0.2677,  0.1245, -0.4699],
        [ 0.4433, -0.4635, -0.4585, -0.3926,  1.3442],
        [ 0.4043,  0.3570,  0.1560,  0.5645, -0.1146],
        [ 0.4255,  1.0955, -0.2423, -0.8003, -0.3336]])
>>> torch.mean(img_t[:, 0, 0])
tensor(-0.0680)
>>> img_t[:, 0, 0]
tensor([-0.2513,  0.6381, -0.5907])
>>>

```

## einsum

* https://pytorch.org/docs/2.5/generated/torch.einsum.html
* [Einsum is All you Need - Einstein Summation in Deep Learning](https://rockt.ai/2018/04/30/einsum)

```python
"""
torch.einsum(equation, *operands) -> Tensor

Parameters
equation (str) – The subscripts for the Einstein summation.
operands (List[Tensor]) – The tensors to compute the Einstein summation of.

Return type
Tensor

"""

```

## refine_names

## rename

## align_as

## sum

```python
"""
torch.sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the input tensor.

Parameters
input (Tensor) – the input tensor.

Keyword Arguments
dtype (torch.dtype, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None.

"""

>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.1133, -0.9567,  0.2958]])
>>> torch.sum(a)
tensor(-0.5475)

"""
torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor

Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.

If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

Parameters
input (Tensor) – the input tensor.
dim (int or tuple of ints, optional) – the dimension or dimensions to reduce. If None, all dimensions are reduced.
keepdim (bool) – whether the output tensor has dim retained or not.

Keyword Arguments
dtype (torch.dtype, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. Default: None.
"""

>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> s1 = torch.sum(x, 0)
>>> s1
tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])
>>> s1.size()
torch.Size([3, 4])
>>> x[:, 0, 0].sum()
tensor(12)
>>> x[:, 0, 1].sum()
tensor(14)
>>>
>>> s1 = torch.sum(x, (0, 2))
>>> s1
tensor([ 60,  92, 124])
>>> s1.size()
torch.Size([3])
>>> x[:, 0, :].sum()
tensor(60)
>>> x[:, 1, :].sum()
tensor(92)
>>> x[:, 2, :].sum()
tensor(124)
>>>

>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>> s1 = torch.sum(x, (0, 1))
>>> s1
tensor([60, 66, 72, 78])
>>> s1.size()
torch.Size([4])
>>> x[..., 0].sum()
tensor(60)
>>> x[..., 1].sum()
tensor(66)
>>> x[..., 2].sum()
tensor(72)
>>> x[..., 3].sum()
tensor(78)
>>>

```


## Softmax

```python

>>> import torch
>>> import torch.nn as nn

>>> x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
>>> x
tensor([[0., 1., 2.],
        [3., 4., 5.]])
>>> m = nn.Softmax(dim=0)
>>> m(x)
tensor([[0.0474, 0.0474, 0.0474],
        [0.9526, 0.9526, 0.9526]])
>>>
#############
# m = nn.Softmax(dim=0) 计算过程
>>> x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
>>> x
tensor([[0., 1., 2.],
        [3., 4., 5.]])
>>> exp_x = torch.exp(x)
>>> exp_x
tensor([[  1.0000,   2.7183,   7.3891],
        [ 20.0855,  54.5981, 148.4132]])
>>> s = torch.sum(exp_x, 0)
>>> s
tensor([ 21.0855,  57.3164, 155.8022])
>>> exp_x / s
tensor([[0.0474, 0.0474, 0.0474],
        [0.9526, 0.9526, 0.9526]])
>>>
>>> exp_x[:, 0]
tensor([ 1.0000, 20.0855])
>>> exp_x[:, 1]
tensor([ 2.7183, 54.5981])
>>> exp_x[:, 2]
tensor([  7.3891, 148.4132])

#################################################################

>>> x = torch.tensor([[0, 4, 1], [2, 1, 4]], dtype=torch.float32)
>>> x
tensor([[0., 4., 1.],
        [2., 1., 4.]])
>>> m = nn.Softmax(dim=0)
>>> m(x)
tensor([[0.1192, 0.9526, 0.0474],
        [0.8808, 0.0474, 0.9526]])
>>>
# m = nn.Softmax(dim=0) 计算过程
>>> x = torch.tensor([[0, 4, 1], [2, 1, 4]], dtype=torch.float32)
>>> x
tensor([[0., 4., 1.],
        [2., 1., 4.]])
>>> exp_x = torch.exp(x)
>>> exp_x
tensor([[ 1.0000, 54.5981,  2.7183],
        [ 7.3891,  2.7183, 54.5981]])
>>> s = torch.sum(exp_x, 0)
>>> s
tensor([ 8.3891, 57.3164, 57.3164])
>>> exp_x / s
tensor([[0.1192, 0.9526, 0.0474],
        [0.8808, 0.0474, 0.9526]])
>>>

#################################################################

>>> x = torch.randn(2, 3, 4)
>>> x
tensor([[[ 1.6960, -0.4364,  0.3784,  0.6687],
         [ 1.0310, -0.5816, -2.8830, -1.8634],
         [ 0.3257,  0.6262, -0.7754, -0.2647]],

        [[ 0.5610,  0.4467, -0.7751, -0.0647],
         [ 0.3111, -1.2507, -1.5323, -0.7719],
         [ 0.2721, -0.0276,  1.7256, -1.3012]]])
>>> m = nn.Softmax(dim=1)
>>> m(x)
tensor([[[0.5655, 0.2101, 0.7387, 0.6790],
         [0.2908, 0.1817, 0.0283, 0.0540],
         [0.1437, 0.6081, 0.2330, 0.2670]],

        [[0.3956, 0.5539, 0.0732, 0.5607],
         [0.3081, 0.1014, 0.0343, 0.2764],
         [0.2963, 0.3447, 0.8925, 0.1628]]])
# m = nn.Softmax(dim=1) 计算过程
>>> exp_x = torch.exp(x)
>>> exp_x
tensor([[[5.4519, 0.6464, 1.4600, 1.9517],
         [2.8038, 0.5590, 0.0560, 0.1551],
         [1.3851, 1.8705, 0.4605, 0.7675]],

        [[1.7524, 1.5632, 0.4606, 0.9374],
         [1.3650, 0.2863, 0.2160, 0.4621],
         [1.3127, 0.9728, 5.6156, 0.2722]]])
>>> s = torch.sum(exp_x, 1)
>>> s
tensor([[9.6408, 3.0758, 1.9764, 2.8743],
        [4.4300, 2.8223, 6.2923, 1.6717]])

>>> s.size()
torch.Size([2, 4])

>>> torch.stack((s,), dim=1)
tensor([[[9.6408, 3.0758, 1.9764, 2.8743]],

        [[4.4300, 2.8223, 6.2923, 1.6717]]])

>>> torch.stack((s,), dim=1).size()
torch.Size([2, 1, 4])

>>> exp_x / (torch.stack((s,), dim=1))
tensor([[[0.5655, 0.2101, 0.7387, 0.6790],
         [0.2908, 0.1817, 0.0283, 0.0540],
         [0.1437, 0.6081, 0.2330, 0.2670]],

        [[0.3956, 0.5539, 0.0732, 0.5607],
         [0.3081, 0.1014, 0.0343, 0.2764],
         [0.2963, 0.3447, 0.8925, 0.1628]]])
>>>

```

## select

```python
"""
torch.select(input, dim, index) -> Tensor

Parameters
input (Tensor) – the input tensor.
dim (int) – the dimension to slice
index (int) – the index to select with

select() is equivalent to slicing.
For example,
tensor.select(0, index) is equivalent to tensor[index]
tensor.select(2, index) is equivalent to tensor[:, :, index]

"""

```

## index_select

```python
"""
torch.index_select(input, dim, index, *, out=None) -> Tensor

Parameters
input (Tensor) – the input tensor.
dim (int) – the dimension in which we index
index (IntTensor or LongTensor) – the "1-D tensor" containing the indices to index

"""

>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices)
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> torch.index_select(x, 1, indices)
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])

```

