# Indexing, Slicing, Joining, Mutating Ops

## Indexing, Slicing

```python

# 张量在内存中是如何存储的
>>> points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 6.0]])
>>> points
tensor([[4., 1.],
        [5., 3.],
        [2., 6.]])
>>>
>>> points.storage()
<stdin>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
 4.0
 1.0
 5.0
 3.0
 2.0
 6.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]
>>>
>>> points.untyped_storage()
 0
 0
 128
 64
 0
 0
 128
 63
 0
 0
 160
 64
 0
 0
 64
 64
 0
 0
 0
 64
 0
 0
 192
 64
[torch.storage.UntypedStorage(device=cpu) of size 24]
>>>
>>> points_storage = points.storage()
>>> points_storage[0]
4.0
>>> points_storage[1]
1.0
>>> points_storage[2]
5.0
>>> points_storage[3]
3.0
>>> points_storage[4]
2.0
>>> points_storage[5]
6.0
>>>
>>> points_storage[0] = 2.0
>>> points
tensor([[2., 1.],
        [5., 3.],
        [2., 6.]])
>>>

# 张量的大小(形状), 步长
>>> points = torch.tensor([[5, 7, 4], [1, 3, 2], [7, 3, 8]])
>>> points.size()
torch.Size([3, 3])
>>> points.stride()
(3, 1)
>>> points[0, 2]
tensor(4)
>>> points[0][2]
tensor(4)
>>>
>>> points.storage()
 5
 7
 4
 1
 3
 2
 7
 3
 8
[torch.storage.TypedStorage(dtype=torch.int64, device=cpu) of size 9]
>>>

"""
points[0, 2]
"""
>>> 0 * 3 + 2 * 1
2
>>>
>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>>
>>> x.storage()
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
[torch.storage.TypedStorage(dtype=torch.int32, device=cpu) of size 24]
>>>
>>> x.size()
torch.Size([2, 3, 4])
>>> x.stride()
(12, 4, 1)
>>> x[1, 2, 3]
tensor(23, dtype=torch.int32)
>>> 1 * 12 + 2 * 4 + 3 * 1
23
>>>

# 一维索引
>>> some_list = list(range(6))
>>> some_list
[0, 1, 2, 3, 4, 5]
>>> some_list[:]
[0, 1, 2, 3, 4, 5]
>>> some_list[1:4]
[1, 2, 3]
>>> some_list[1:]
[1, 2, 3, 4, 5]
>>> some_list[:4]
[0, 1, 2, 3]
>>> some_list[:-1]  # 不包括最后一个元素
[0, 1, 2, 3, 4]
>>> some_list[1:4:2]
[1, 3]
>>>

# 二维索引
>>> points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
>>> points
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])
>>> points[0, 1]
tensor(1.)
>>> points[0]
tensor([4., 1.])
>>> points[1:]
tensor([[5., 3.],
        [2., 1.]])
>>> points[1:, :]
tensor([[5., 3.],
        [2., 1.]])
>>> points[1:, 0]
tensor([5., 2.])
>>> points[None]
tensor([[[4., 1.],
         [5., 3.],
         [2., 1.]]])
>>> points[None].size()
torch.Size([1, 3, 2])
>>>

# 三维索引
>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>> x[1, :, :]
tensor([[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]], dtype=torch.int32)
>>> x[1, 2, :]
tensor([20, 21, 22, 23], dtype=torch.int32)
>>>
>>> x[1, 2, 3]
tensor(23, dtype=torch.int32)
>>>
>>> x[1, 0:2, :]
tensor([[12, 13, 14, 15],
        [16, 17, 18, 19]], dtype=torch.int32)
>>>
>>> x[1, 0:2, 3]
tensor([15, 19], dtype=torch.int32)
>>>

```

## transpose

> swapaxes 	Alias for torch.transpose().
> swapdims 	Alias for torch.transpose().

```python
"""
torch.transpose(input, dim0, dim1) -> Tensor

Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
返回一个张量, 该张量是 input 的转置版本. 给定的维度 dim0 和 dim1 被交换.

If input is a strided tensor then the resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.
如果 input 是一个跨步张量, 则生成的 out 张量与 input 张量共享其底层存储, 因此更改一个张量的内容会更改另一个张量的内容.

If input is a sparse tensor then the resulting out tensor does not share the underlying storage with the input tensor.
如果 input 是稀疏张量, 则生成的 out 张量不会与 input 张量共享底层存储.

If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC) the arguments dim0 and dim1 must be both batch dimensions, or must both be sparse dimensions. The batch dimensions of a sparse tensor are the dimensions preceding the sparse dimensions.
如果 input 是具有压缩布局的稀疏张量(SparseCSR、SparseBSR、SparseCSC 或 SparseBSC), 则参数 dim0 和 dim1 必须都是批处理维度, 或者都必须是稀疏维度. 稀疏张量的批量维度是稀疏维度之前的维度.


在 PyTorch中, transpose() 是一种操作, 它交换张量中两个指定维度的位置. 实现这一点的关键在于不实际移动数据, 而是通过改变张量的元数据( 包括步长(stride)和尺寸(size) )来达到效果.

举例来说, 假设我们有一个形状为 (3, 4) 的二维张量, 其内存布局为行优先 (row-major) 即 C 风格的. 当我们对这个张量执行 transpose(0, 1) 操作时, 我们期望该张量行变成列, 列变成行, 即得到一个形状为 (4, 3) 的新视图.

这是通过以下步骤完成的:

改变尺寸: 改变 size 元数据, 使得原本第一个维度(行)的大小与第二个维度(列)的大小交换.

改变步长: 步长(stride)是一个数组, 指示了在每个维度上移动一个元素需要跳过的内存位置数. 执行 transpose() 时, 交换了两个维度的步长. 在行优先存储的张量中, 行的步长通常比列的步长大.

不移动数据: 实际上数据并没有在内存中移动, 只是改变了在这块内存空间上的解释方式.

"""

>>> import torch

>>> x = torch.arange(6, dtype=torch.int32).reshape(2, 3)
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>> t1 = torch.transpose(x, 0, 1)
>>> t1.size()
torch.Size([3, 2])
>>> t1.stride()
(1, 3)
>>>
"""
    size()              stride()
x   torch.Size([2, 3])  (3, 1)
t1  torch.Size([3, 2])  (1, 3)  t1 = torch.transpose(x, 0, 1)
"""
>>> x = torch.arange(6, dtype=torch.int32).reshape(2, 3)
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]], dtype=torch.int32)
>>> x.size()
torch.Size([2, 3])
>>> torch.transpose(x, 0, 1)
tensor([[0, 3],
        [1, 4],
        [2, 5]], dtype=torch.int32)
>>> torch.transpose(x, 0, 1).size()
torch.Size([3, 2])
>>>


>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>> x.size()
torch.Size([2, 3, 4])
>>> x.stride()
(12, 4, 1)
>>>
>>> t1 = torch.transpose(x, 0, 1)
>>> t1.size()
torch.Size([3, 2, 4])
>>> t1.stride()
(4, 12, 1)
>>>
>>> t2 = torch.transpose(x, 0, 2)
>>> t2.size()
torch.Size([4, 3, 2])
>>> t2.stride()
(1, 4, 12)
>>>
>>> t3 = torch.transpose(x, 1, 2)
>>> t3.size()
torch.Size([2, 4, 3])
>>> t3.stride()
(12, 1, 4)
>>>
"""
    size()                 stride()
x   torch.Size([2, 3, 4])  (12, 4, 1)
t1  torch.Size([3, 2, 4])  (4, 12, 1)  t1 = torch.transpose(x, 0, 1)
t2  torch.Size([4, 3, 2])  (1, 4, 12)  t2 = torch.transpose(x, 0, 2)
t3  torch.Size([2, 4, 3])  (12, 1, 4)  t3 = torch.transpose(x, 1, 2)
"""

>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x.size()
torch.Size([2, 3, 4])
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)

>>> t1 = torch.transpose(x, 0, 1)
>>> t1
tensor([[[ 0,  1,  2,  3],
         [12, 13, 14, 15]],

        [[ 4,  5,  6,  7],
         [16, 17, 18, 19]],

        [[ 8,  9, 10, 11],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>> t1.size()
torch.Size([3, 2, 4])
>>>
>>> t2 = torch.transpose(x, 0, 2)
>>> t2
tensor([[[ 0, 12],
         [ 4, 16],
         [ 8, 20]],

        [[ 1, 13],
         [ 5, 17],
         [ 9, 21]],

        [[ 2, 14],
         [ 6, 18],
         [10, 22]],

        [[ 3, 15],
         [ 7, 19],
         [11, 23]]], dtype=torch.int32)
>>> t2.size()
torch.Size([4, 3, 2])

```

## t

```python
"""
torch.t(input) -> Tensor

Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
预期 input 为 <= 2-D 张量并转置维度 0 和 1.

0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to transpose(input, 0, 1).
0-D 和 1-D 张量按原样返回. 当 input 是 2-D 张量时, 这相当于 transpose(input, 0, 1).
"""

# 0-D
>>> x = torch.randn(())
>>> x
tensor(0.6791)
>>> x.size()
torch.Size([])
>>> x.shape
torch.Size([])
>>> torch.t(x)
tensor(0.6791)
>>>

# 1-D
>>> x = torch.randn(3)
>>> x
tensor([ 0.2887, -1.8138, -0.3809])
>>> x.size()
torch.Size([3])
>>> x.shape
torch.Size([3])
>>> torch.t(x)
tensor([ 0.2887, -1.8138, -0.3809])
>>>

# 2-D
>>> x = torch.randn(2, 3)
>>> x
tensor([[-1.1509, -1.2261,  1.7048],
        [ 1.0926,  0.3198, -0.7155]])
>>> x.size()
torch.Size([2, 3])
>>> x.shape
torch.Size([2, 3])
>>> torch.t(x)
tensor([[-1.1509,  1.0926],
        [-1.2261,  0.3198],
        [ 1.7048, -0.7155]])
>>>

# 3-D
>>> x = torch.randn(2, 3, 4)
>>> x
tensor([[[-0.4031, -0.6331, -0.1904,  0.8779],
         [-0.1151, -1.2163,  1.9360,  0.6746],
         [ 1.1644,  0.4030, -1.7059, -0.5242]],

        [[-0.4914, -0.0453,  1.3255,  0.7283],
         [ 0.3376,  0.1478, -0.8284,  0.4289],
         [-0.3383, -1.0533, -0.3200, -0.5382]]])
>>> x.size()
torch.Size([2, 3, 4])
>>> torch.t(x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
>>>

```

## permute

```python
"""
torch.permute(input, dims) -> Tensor
Returns a view of the original tensor input with its dimensions permuted.

Parameters
input (Tensor) – the input tensor.
dims (tuple of int) – The desired ordering of dimensions

"""

>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> x.stride()
(15, 5, 1)
>>>
>>> p1 = torch.permute(x, (2, 0, 1))
>>> p1.size()
torch.Size([5, 2, 3])
>>> p1.stride()
(1, 15, 5)
>>>

"""
    size()                 stride()
x   torch.Size([2, 3, 5])  (15, 5, 1)
p1  torch.Size([5, 2, 3])  (1, 15, 5)  p1 = torch.permute(x, (2, 0, 1))
"""

```

## reshape

```python
"""
torch.reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.
返回一个张量, 其数据和元素数与输入相同, 但具有指定的形状. 如果可能, 返回的张量将是 input 的视图. 否则, 它将是一个副本. 连续的 inputs 和具有兼容步幅的 inputs 可以在不复制的情况下重塑, 但您不应依赖于复制与查看行为.

See torch.Tensor.view() on when it is possible to return a view.
参见 torch.Tensor.view() 何时可以返回视图.

A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input.
单个维度可能是 -1, 在这种情况下, 它是从剩余维度和输入中的元素数推断出来的.

Parameters  参数
input (Tensor) – the tensor to be reshaped
shape (tuple of int) – the new shape

"""

>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x.size()
torch.Size([2, 3, 4])
>>> x.stride()
(12, 4, 1)
>>> r1 = torch.reshape(x, (3, 2, 4))
>>> r1.size()
torch.Size([3, 2, 4])
>>> r1.stride()
(8, 4, 1)
>>>
"""
    size()                 stride()
x   torch.Size([2, 3, 4])  (12, 4, 1)
r1  torch.Size([3, 2, 4])  (8, 4, 1)  r1 = torch.reshape(x, (3, 2, 4))
"""


>>> a = torch.arange(4.)
>>> a
tensor([0., 1., 2., 3.])
>>> a.size()
torch.Size([4])
>>> torch.reshape(a, (2, 2))
tensor([[0., 1.],
        [2., 3.]])
>>>

>>> b = torch.tensor([[0, 1], [2, 3]])
>>> b
tensor([[0, 1],
        [2, 3]])
>>> b.size()
torch.Size([2, 2])
>>> torch.reshape(b, (-1,))  # 变成一维
tensor([0, 1, 2, 3])
>>>
>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>> x.size()
torch.Size([2, 3, 4])
>>>
>>> torch.reshape(x, (-1,))
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23], dtype=torch.int32)
>>>
>>> torch.reshape(x, (3, -1,))
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23]], dtype=torch.int32)
>>>
>>> torch.reshape(x, (3, 2, -1))
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7]],

        [[ 8,  9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23]]], dtype=torch.int32)
>>>
>>> torch.reshape(x, (-1, 2))
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21],
        [22, 23]], dtype=torch.int32)
>>>

```

## stack

```python
"""
torch.stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.
沿新维度连接一系列张量.

All tensors need to be of the same size.
所有张量的大小都需要相同.

See also  另请参阅
torch.cat() concatenates the given sequence along an existing dimension.
torch.cat() 沿现有维度连接给定序列.

Parameters
tensors (sequence of Tensors) – sequence of tensors to concatenate
dim (int, optional) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive). Default: 0

Keyword Arguments
out (Tensor, optional) – the output tensor.

"""

>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y
tensor([[ 6,  7,  8],
        [ 9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>>

"""
[2, 3]
[i, 2, 3]
[2, i, 3]
[2, 3, i]
"""
>>> torch.stack((x, y, x, y), dim=-4).size()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: Dimension out of range (expected to be in range of [-3, 2], but got -4)
>>> torch.stack((x, y, x, y), dim=-3).size()
torch.Size([4, 2, 3])
>>> torch.stack((x, y, x, y), dim=-2).size()
torch.Size([2, 4, 3])
>>> torch.stack((x, y, x, y), dim=-1).size()
torch.Size([2, 3, 4])
>>> torch.stack((x, y, x, y), dim=0).size()
torch.Size([4, 2, 3])
>>> torch.stack((x, y, x, y), dim=1).size()
torch.Size([2, 4, 3])
>>> torch.stack((x, y, x, y), dim=2).size()
torch.Size([2, 3, 4])
>>> torch.stack((x, y, x, y), dim=3).size()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
>>>



>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)
>>>
>>> s0 = torch.stack((x, y), dim=0)
>>> s0
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
>>> s0.size()
torch.Size([2, 2, 3])
>>> s0.stride()
(6, 3, 1)
>>>
"""
    size()                 stride()
x   torch.Size([2, 3])     (3, 1)
y   torch.Size([2, 3])     (3, 1)
s0  torch.Size([2, 2, 3])  (6, 3, 1)  s0 = torch.stack((x, y), dim=0)
"""

```

## cat

> concat 	Alias of torch.cat().
> concatenate 	Alias of torch.cat().

```python
"""
torch.cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).
连接给定维度中给定的 seq 张量序列. 所有张量必须具有相同的形状(连接维度除外)或大小为 (0, ) 的一维空张量.

torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk().
torch.cat() 可以看作是 torch.split() 和 torch.chunk() 的反向作.

torch.cat() can be best understood via examples.
torch.cat() 可以通过示例最好地理解.

See also  另请参阅
torch.stack() concatenates the given sequence along a new dimension.
torch.stack() 沿新维度连接给定的序列.

Parameters
tensors (sequence of Tensors) – any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.
dim (int, optional) – the dimension over which the tensors are concatenated

Keyword Arguments
out (Tensor, optional) – the output tensor.

"""

>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y
tensor([[ 6,  7,  8],
        [ 9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>>

>>> torch.cat((x, y, x, y), dim=-3).size()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got -3)
>>> torch.cat((x, y, x, y), dim=-2).size()
torch.Size([8, 3])
>>> torch.cat((x, y, x, y), dim=-1).size()
torch.Size([2, 12])
>>> torch.cat((x, y, x, y), dim=0).size()
torch.Size([8, 3])
>>> torch.cat((x, y, x, y), dim=1).size()
torch.Size([2, 12])
>>> torch.cat((x, y, x, y), dim=2).size()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
>>>

```

## vstack 	Stack tensors in sequence vertically (row wise).

> row_stack 	Alias of torch.vstack().

```python
"""
torch.vstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence vertically (row wise).
按顺序垂直(逐行)堆叠张量. 

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by torch.atleast_2d().
这相当于在所有一维张量都被 torch.atleast_2d() 重塑后沿第一轴的串联. 

Parameters
tensors (sequence of Tensors) – sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional) – the output tensor.
"""

>>> a = torch.tensor([1, 2, 3])
>>> a
tensor([1, 2, 3])
>>> a.size()
torch.Size([3])
>>> b = torch.tensor([4, 5, 6])
>>> b
tensor([4, 5, 6])
>>> b.size()
torch.Size([3])
>>> v1 = torch.vstack((a, b))
>>> v1
tensor([[1, 2, 3],
        [4, 5, 6]])
>>> v1.size()
torch.Size([2, 3])
>>>
>>> a = torch.tensor([[1],[2],[3]])
>>> a
tensor([[1],
        [2],
        [3]])
>>> a.size()
torch.Size([3, 1])
>>> b = torch.tensor([[4],[5],[6]])
>>> b
tensor([[4],
        [5],
        [6]])
>>> b.size()
torch.Size([3, 1])
>>> v2 = torch.vstack((a, b))
>>> v2
tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6]])
>>> v2.size()
torch.Size([6, 1])
>>>
>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y
tensor([[ 6,  7,  8],
        [ 9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> v3 = torch.vstack((x, y))
>>> v3
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
>>> v3.size()
torch.Size([4, 3])
>>>

```

## hstack

```python
"""
torch.hstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence horizontally (column wise).
水平顺序(逐列)堆叠张量. 

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.
这相当于一维张量沿第一轴连接, 所有其他张量沿第二轴连接. 

Parameters
tensors (sequence of Tensors) – sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional) – the output tensor.

"""

>>> a = torch.tensor([1, 2, 3])
>>> a
tensor([1, 2, 3])
>>> a.size()
torch.Size([3])
>>> b = torch.tensor([4, 5, 6])
>>> b
tensor([4, 5, 6])
>>> b.size()
torch.Size([3])
>>> h1 = torch.hstack((a, b))
>>> h1
tensor([1, 2, 3, 4, 5, 6])
>>> h1.size()
torch.Size([6])
>>>
>>> a = torch.tensor([[1],[2],[3]])
>>> a
tensor([[1],
        [2],
        [3]])
>>> a.size()
torch.Size([3, 1])
>>> b = torch.tensor([[4],[5],[6]])
>>> b
tensor([[4],
        [5],
        [6]])
>>> b.size()
torch.Size([3, 1])
>>> h2 = torch.hstack((a, b))
>>> h2
tensor([[1, 4],
        [2, 5],
        [3, 6]])
>>> h2.size()
torch.Size([3, 2])
>>>
>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y
tensor([[ 6,  7,  8],
        [ 9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>>
>>> h3 = torch.hstack((x, y))
>>> h3
tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
>>> h3.size()
torch.Size([2, 6])
>>>

```

## dstack

```python
"""
torch.dstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence depthwise (along third axis).
按顺序纵向堆叠张量(沿第三个轴). 

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().
这相当于在 1-D 和 2-D 张量被 torch.atleast_3d() 重塑后沿第三轴的串联. 

Parameters
tensors (sequence of Tensors) – sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional) – the output tensor.

"""

```

* column_stack 	Creates a new tensor by horizontally stacking the tensors in tensors.

atleast_3d()
atleast_2d()




* chunk 	Attempts to split a tensor into the specified number of chunks.

* dsplit 	Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.
* hsplit 	Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.
* split 	Splits the tensor into chunks.
* tensor_split 	Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.
* vsplit 	Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.





* adjoint 	Returns a view of the tensor conjugated and with the last two dimensions transposed.

* argwhere 	Returns a tensor containing the indices of all `non-zero` elements of input.

* conj 	Returns a view of input with a flipped conjugate bit.

* gather 	Gathers values along an axis specified by dim.

* index_add 	See index_add_() for function description.

* index_copy 	See index_add_() for function description.

* index_reduce 	See index_reduce_() for function description.

* index_select 	Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.

* masked_select 	Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.

* movedim 	Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

* moveaxis 	Alias for torch.movedim().

* narrow 	Returns a new tensor that is a narrowed version of input tensor.

* narrow_copy 	Same as Tensor.narrow() except this returns a copy rather than shared storage.

* nonzero

* select 	Slices the input tensor along the selected dimension at the given index.

* scatter 	Out-of-place version of torch.Tensor.scatter_()

* diagonal_scatter 	Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.

* select_scatter 	Embeds the values of the src tensor into input at the given index.

* slice_scatter 	Embeds the values of the src tensor into input at the given dimension.

* scatter_add 	Out-of-place version of torch.Tensor.scatter_add_()

* scatter_reduce 	Out-of-place version of torch.Tensor.scatter_reduce_()

* squeeze 	Returns a tensor with all specified dimensions of input of size 1 removed.

* take 	Returns a new tensor with the elements of input at the given indices.

* take_along_dim 	Selects values from input at the 1-dimensional indices from indices along the given dim.

* tile 	Constructs a tensor by repeating the elements of input.

* unbind 	Removes a tensor dimension.

* unravel_index 	Converts a tensor of flat indices into a tuple of coordinate tensors that index into an arbitrary tensor of the specified shape.

* unsqueeze 	Returns a new tensor with a dimension of size one inserted at the specified position.

* where 	Return a tensor of elements selected from either input or other, depending on condition.

