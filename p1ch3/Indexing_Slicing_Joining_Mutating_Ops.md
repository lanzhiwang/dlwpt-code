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
<stdin>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly. To access UntypedStorage directly, use tensor.untyped_storage()instead of tensor.storage()
 4.0
 1.0
 5.0
 3.0
 2.0
 6.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu)of size 6]
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
[torch.storage.UntypedStorage(device=cpu)of size 24]
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
[torch.storage.TypedStorage(dtype=torch.int64, device=cpu)of size 9]
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
[torch.storage.TypedStorage(dtype=torch.int32, device=cpu)of size 24]
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
torch.transpose(input, dim0, dim1)-> Tensor

Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
返回一个张量, 该张量是 input 的转置版本. 给定的维度 dim0 和 dim1 被交换.

If input is a strided tensor then the resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.
如果 input 是一个跨步张量, 则生成的 out 张量与 input 张量共享其底层存储, 因此更改一个张量的内容会更改另一个张量的内容.

If input is a sparse tensor then the resulting out tensor does not share the underlying storage with the input tensor.
如果 input 是稀疏张量, 则生成的 out 张量不会与 input 张量共享底层存储.

If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC)the arguments dim0 and dim1 must be both batch dimensions, or must both be sparse dimensions. The batch dimensions of a sparse tensor are the dimensions preceding the sparse dimensions.
如果 input 是具有压缩布局的稀疏张量(SparseCSR、SparseBSR、SparseCSC 或 SparseBSC), 则参数 dim0 和 dim1 必须都是批处理维度, 或者都必须是稀疏维度. 稀疏张量的批量维度是稀疏维度之前的维度.


在 PyTorch中, transpose()是一种操作, 它交换张量中两个指定维度的位置. 实现这一点的关键在于不实际移动数据, 而是通过改变张量的元数据( 包括步长(stride)和尺寸(size))来达到效果.

举例来说, 假设我们有一个形状为 (3, 4)的二维张量, 其内存布局为行优先 (row-major)即 C 风格的. 当我们对这个张量执行 transpose(0, 1)操作时, 我们期望该张量行变成列, 列变成行, 即得到一个形状为 (4, 3)的新视图.

这是通过以下步骤完成的:

改变尺寸: 改变 size 元数据, 使得原本第一个维度(行)的大小与第二个维度(列)的大小交换.

改变步长: 步长(stride)是一个数组, 指示了在每个维度上移动一个元素需要跳过的内存位置数. 执行 transpose()时, 交换了两个维度的步长. 在行优先存储的张量中, 行的步长通常比列的步长大.

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
    size()             stride()
x   torch.Size([2, 3]) (3, 1)
t1  torch.Size([3, 2]) (1, 3) t1 = torch.transpose(x, 0, 1)
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
    size()                stride()
x   torch.Size([2, 3, 4]) (12, 4, 1)
t1  torch.Size([3, 2, 4]) (4, 12, 1) t1 = torch.transpose(x, 0, 1)
t2  torch.Size([4, 3, 2]) (1, 4, 12) t2 = torch.transpose(x, 0, 2)
t3  torch.Size([2, 4, 3]) (12, 1, 4) t3 = torch.transpose(x, 1, 2)
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
torch.t(input)-> Tensor

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
RuntimeError: t()expects a tensor with <= 2 dimensions, but self is 3D
>>>

```

## permute

```python
"""
torch.permute(input, dims)-> Tensor
Returns a view of the original tensor input with its dimensions permuted.

Parameters
input (Tensor)– the input tensor.
dims (tuple of int)– The desired ordering of dimensions

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
    size()                stride()
x   torch.Size([2, 3, 5]) (15, 5, 1)
p1  torch.Size([5, 2, 3]) (1, 15, 5) p1 = torch.permute(x, (2, 0, 1))
"""

```

## reshape

```python
"""
torch.reshape(input, shape)-> Tensor

Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.
返回一个张量, 其数据和元素数与输入相同, 但具有指定的形状. 如果可能, 返回的张量将是 input 的视图. 否则, 它将是一个副本. 连续的 inputs 和具有兼容步幅的 inputs 可以在不复制的情况下重塑, 但您不应依赖于复制与查看行为.

See torch.Tensor.view()on when it is possible to return a view.
参见 torch.Tensor.view()何时可以返回视图.

A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input.
单个维度可能是 -1, 在这种情况下, 它是从剩余维度和输入中的元素数推断出来的.

Parameters  参数
input (Tensor)– the tensor to be reshaped
shape (tuple of int)– the new shape

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
    size()                stride()
x   torch.Size([2, 3, 4]) (12, 4, 1)
r1  torch.Size([3, 2, 4]) (8, 4, 1) r1 = torch.reshape(x, (3, 2, 4))
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
>>> torch.reshape(b, (-1,)) # 变成一维
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
torch.stack(tensors, dim=0, *, out=None)-> Tensor

Concatenates a sequence of tensors along a new dimension.
沿新维度连接一系列张量.

All tensors need to be of the same size.
所有张量的大小都需要相同.

See also  另请参阅
torch.cat()concatenates the given sequence along an existing dimension.
torch.cat()沿现有维度连接给定序列.

Parameters
tensors (sequence of Tensors)– sequence of tensors to concatenate
dim (int, optional)– dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive). Default: 0

Keyword Arguments
out (Tensor, optional)– the output tensor.

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
    size()                stride()
x   torch.Size([2, 3])    (3, 1)
y   torch.Size([2, 3])    (3, 1)
s0  torch.Size([2, 2, 3]) (6, 3, 1) s0 = torch.stack((x, y), dim=0)
"""

```

## cat

> concat 	Alias of torch.cat().
> concatenate 	Alias of torch.cat().

```python
"""
torch.cat(tensors, dim=0, *, out=None)-> Tensor

Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension)or be a 1-D empty tensor with size (0,).
连接给定维度中给定的 seq 张量序列. 所有张量必须具有相同的形状(连接维度除外)或大小为 (0, )的一维空张量.

torch.cat()can be seen as an inverse operation for torch.split()and torch.chunk().
torch.cat()可以看作是 torch.split()和 torch.chunk()的反向作.

torch.cat()can be best understood via examples.
torch.cat()可以通过示例最好地理解.

See also  另请参阅
torch.stack()concatenates the given sequence along a new dimension.
torch.stack()沿新维度连接给定的序列.

Parameters
tensors (sequence of Tensors)– any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.
dim (int, optional)– the dimension over which the tensors are concatenated

Keyword Arguments
out (Tensor, optional)– the output tensor.

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


>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)
>>>
>>> c0 = torch.cat((x, y, x, y), dim=0)
>>> c0
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
>>> c0.size()
torch.Size([8, 3])
>>> c0.stride()
(3, 1)
>>>

>>> x = torch.randn(())
>>> x
tensor(-0.9239)
>>> c0 = torch.cat((x, x, x, x), dim=0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: zero-dimensional tensor (at position 0)cannot be concatenated
>>>
>>> x = torch.randn(1)
>>> x
tensor([-1.1279])
>>> c0 = torch.cat((x, x, x, x), dim=0)
>>> c0
tensor([-1.1279, -1.1279, -1.1279, -1.1279])
>>> x = torch.randn(2)
>>> x
tensor([-0.6517,  0.1509])
>>> c0 = torch.cat((x, x, x, x), dim=0)
>>> c0
tensor([-0.6517,  0.1509, -0.6517,  0.1509, -0.6517,  0.1509, -0.6517,  0.1509])
>>>

```

## vstack

> row_stack 	Alias of torch.vstack().

```python
"""
torch.vstack(tensors, *, out=None)-> Tensor

Stack tensors in sequence vertically (row wise).
按顺序垂直(逐行)堆叠张量.

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by torch.atleast_2d().
这相当于在所有一维张量都被 torch.atleast_2d()重塑后沿第一轴的串联.

Parameters
tensors (sequence of Tensors)– sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional)– the output tensor.
"""


>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)
>>>
>>> v1 = torch.vstack((x, y)) # 相当于 torch.cat((x, y), dim=0)
>>> v1.size()
torch.Size([4, 3])
>>> v1.stride()
(3, 1)
>>>
>>> v1
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
>>>
>>> torch.cat((x, y), dim=0)
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
>>>

```

## hstack

```python
"""
torch.hstack(tensors, *, out=None)-> Tensor

Stack tensors in sequence horizontally (column wise).
水平顺序(逐列)堆叠张量.

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.
这相当于一维张量沿第一轴连接, 所有其他张量沿第二轴连接.

Parameters
tensors (sequence of Tensors)– sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional)– the output tensor.

"""


>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)

>>> h1 = torch.hstack((x, y)) # 相当于 torch.cat((x, y), dim=1)
>>> h1.size()
torch.Size([2, 6])
>>> h1.stride()
(6, 1)
>>> h1
tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
>>> torch.cat((x, y), dim=1)
tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
>>>

>>> x = torch.tensor([0, 1, 2])
>>> x.size()
torch.Size([3])
>>> x.stride()
(1,)
>>> y = torch.tensor([3, 4, 5])
>>> y.size()
torch.Size([3])
>>> y.stride()
(1,)
>>> h2 = torch.hstack((x, y))
>>> h2.size()
torch.Size([6])
>>> h2.stride()
(1,)
>>> h2
tensor([0, 1, 2, 3, 4, 5])
>>>

```

## dstack

```python
"""
torch.dstack(tensors, *, out=None)-> Tensor

Stack tensors in sequence depthwise (along third axis).
按顺序纵向堆叠张量(沿第三个轴).

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().
这相当于在 1-D 和 2-D 张量被 torch.atleast_3d()重塑后沿第三轴的串联.

Parameters
tensors (sequence of Tensors)– sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional)– the output tensor.

"""

>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)

>>> d1 = torch.dstack((x, y))
>>> d1.size()
torch.Size([2, 3, 2])
>>> d1.stride()
(6, 2, 1)
>>> d1
tensor([[[ 0,  6],
         [ 1,  7],
         [ 2,  8]],

        [[ 3,  9],
         [ 4, 10],
         [ 5, 11]]])
>>> torch.cat((x, y), dim=2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
>>>

>>> x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
>>> x.size()
torch.Size([2, 3, 4])
>>> x.stride()
(12, 4, 1)
>>>
>>> y = torch.arange(25, 49, dtype=torch.int32).reshape(2, 3, 4)
>>> y
tensor([[[25, 26, 27, 28],
         [29, 30, 31, 32],
         [33, 34, 35, 36]],

        [[37, 38, 39, 40],
         [41, 42, 43, 44],
         [45, 46, 47, 48]]], dtype=torch.int32)
>>> y.size()
torch.Size([2, 3, 4])
>>> y.stride()
(12, 4, 1)
>>>
>>> d2 = torch.dstack((x, y)) # 相当于 torch.cat((x, y), dim=2)
>>> d2.size()
torch.Size([2, 3, 8])
>>> d2.stride()
(24, 8, 1)
>>> d2
tensor([[[ 0,  1,  2,  3, 25, 26, 27, 28],
         [ 4,  5,  6,  7, 29, 30, 31, 32],
         [ 8,  9, 10, 11, 33, 34, 35, 36]],

        [[12, 13, 14, 15, 37, 38, 39, 40],
         [16, 17, 18, 19, 41, 42, 43, 44],
         [20, 21, 22, 23, 45, 46, 47, 48]]], dtype=torch.int32)
>>>
>>> torch.cat((x, y), dim=2)
tensor([[[ 0,  1,  2,  3, 25, 26, 27, 28],
         [ 4,  5,  6,  7, 29, 30, 31, 32],
         [ 8,  9, 10, 11, 33, 34, 35, 36]],

        [[12, 13, 14, 15, 37, 38, 39, 40],
         [16, 17, 18, 19, 41, 42, 43, 44],
         [20, 21, 22, 23, 45, 46, 47, 48]]], dtype=torch.int32)
>>>

```

## atleast_2d and atleast_3d

## column_stack

```python
"""
torch.column_stack(tensors, *, out=None)-> Tensor

Creates a new tensor by horizontally stacking the tensors in tensors.
通过将张量水平堆叠在张量中来创建新的张量.

Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1)column before being stacked horizontally.
等效于 torch.hstack(tensors), 不同之处在于张量中的每个零维或一维张量 t 在水平堆叠之前首先重塑为 (t.numel(), 1)列.

Parameters
tensors (sequence of Tensors)– sequence of tensors to concatenate

Keyword Arguments
out (Tensor, optional)– the output tensor.
"""

>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> x.size()
torch.Size([2, 3])
>>> x.stride()
(3, 1)
>>>
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> y.size()
torch.Size([2, 3])
>>> y.stride()
(3, 1)

>>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])
>>> y = torch.tensor([[6, 7, 8], [9, 10, 11]])
>>> ca = torch.column_stack((x, y))
>>> ca.size()
torch.Size([2, 6])
>>> ca.stride()
(6, 1)
>>> ca
tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
>>>

```

## split

```python
"""
torch.split(tensor, split_size_or_sections, dim=0)

Splits the tensor into chunks. Each chunk is a view of the original tensor.
将张量拆分为块. 每个块都是原始张量的视图.

If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
如果 split_size_or_sections 是整数类型, 则 tensor 将被拆分为大小相等的块(如果可能). 如果沿给定维度 dim 的张量大小不能被 split_size 整除, 则最后一个块将更小.

If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections)chunks with sizes in dim according to split_size_or_sections.
如果 split_size_or_sections 是一个列表, 那么 tensor 将根据 split_size_or_sections 被分成 len(split_size_or_sections)个大小为 dim 的块.

Parameters
tensor (Tensor)– tensor to split.
split_size_or_sections (int)or (list(int))– size of a single chunk or list of sizes for each chunk
dim (int)– dimension along which to split the tensor.

Return type
Tuple[Tensor, …]

"""

```

## chunk

```python
"""
torch.chunk(input, chunks, dim=0)-> List of Tensors

Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
尝试将张量拆分为指定数量的块. 每个块都是输入张量的视图.

Note
This function may return fewer than the specified number of chunks!
此函数返回的块数可能少于指定数量的块!

See also
torch.tensor_split()a function that always returns exactly the specified number of chunks
torch.tensor_split()一个始终返回指定数量的 chunk 的函数

If the tensor size along the given dimension dim is divisible by chunks, all returned chunks will be the same size. If the tensor size along the given dimension dim is not divisible by chunks, all returned chunks will be the same size, except the last one. If such division is not possible, this function may return fewer than the specified number of chunks.
如果沿给定维度 dim 的张量大小可被块整除, 则所有返回的块将具有相同的大小. 如果沿给定维度 dim 的张量大小不能被块整除, 则所有返回的块都将具有相同的大小, 但最后一个块除外. 如果无法进行此类划分, 则此函数返回的块数可能会少于指定数量的块.

Parameters
input (Tensor)– the tensor to split
chunks (int)– number of chunks to return
dim (int)– dimension along which to split the tensor
"""

```

## tensor_split

```python
"""
torch.tensor_split(input, indices_or_sections, dim=0)-> List of Tensors

Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections. This function is based on NumPy’s numpy.array_split().
根据 indices_or_sections 指定的索引或截面数, 将一个张量拆分为多个子张量, 所有这些子张量都是沿维度 dim 输入视图. 此函数基于 NumPy 的 numpy.array_split().

Parameters
input (Tensor)– the tensor to split
indices_or_sections (Tensor, int or list or tuple of ints)–
    If indices_or_sections is an integer n or a zero dimensional long tensor with value n, input is split into n sections along dimension dim. If input is divisible by n along dimension dim, each section will be of equal size, input.size(dim)/ n. If input is not divisible by n, the sizes of the first int(input.size(dim)% n)sections will have size int(input.size(dim)/ n)+ 1, and the rest will have size int(input.size(dim)/ n).
    如果 indices_or_sections 是整数 n 或值为 n 的零维长张量, 则输入沿维度 dim 分成 n 个部分. 如果 input 沿维度 dim 可被 n 整除, 则每个部分的大小相等, 即 input.size(dim)/ n. 如果 input 不能被 n 整除, 则第一个 int(input.size(dim)% n)部分的大小将具有 int(input.size(dim)/ n)+ 1 的大小, 其余部分的大小将为 int(input.size(dim)/ n).

    If indices_or_sections is a list or tuple of ints, or a one-dimensional long tensor, then input is split along dimension dim at each of the indices in the list, tuple or tensor. For instance, indices_or_sections=[2, 3] and dim=0 would result in the tensors input[:2], input[2:3], and input[3:].
    如果 indices_or_sections 是 int 的列表或 Tuples, 或者是一维 long tensor, 则 input 将沿 dim 维度在列表、元组或 tensor 中的每个索引处进行拆分. 例如, indices_or_sections=[2, 3] 和 dim=0 将导致张量 input[: 2]、input[2: 3] 和 input[3: ].

    If indices_or_sections is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.
    如果 indices_or_sections 是张量, 则它必须是 CPU 上的零维或一维长张量.

dim (int, optional)– dimension along which to split the tensor. Default: 0

"""

```

## dsplit

```python
"""
torch.dsplit(input, indices_or_sections)-> List of Tensors

Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.
根据 indices_or_sections 将输入(具有三个或更多维度的张量)深度拆分为多个张量. 每个 split 都是一个 input 视图.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=2)(the split dimension is 2), except that if indices_or_sections is an integer it must evenly divide the split dimension or a runtime error will be thrown.
这相当于调用 torch.tensor_split(input, indices_or_sections, dim=2)(split dimension 为 2), 不同之处在于, 如果 indices_or_sections 是整数, 则必须均匀划分 split dimension, 否则将引发运行时错误.

This function is based on NumPy's numpy.dsplit().

Parameters
input (Tensor)– tensor to split.
indices_or_sections (int or list or tuple of ints)– See argument in torch.tensor_split().
"""

```

## hsplit

```python
"""
torch.hsplit(input, indices_or_sections)-> List of Tensors

Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections. Each split is a view of input.
根据 indices_or_sections 将输入(具有一个或多个维度的张量)水平拆分为多个张量. 每个 split 都是一个 input 视图.

If input is one dimensional this is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0)(the split dimension is zero), and if input has two or more dimensions it’s equivalent to calling torch.tensor_split(input, indices_or_sections, dim=1)(the split dimension is 1), except that if indices_or_sections is an integer it must evenly divide the split dimension or a runtime error will be thrown.
如果 input 是一维的, 这相当于调用 torch.tensor_split(input, indices_or_sections, dim=0)(split dimension is zero), 如果 input 有两个或多个维度, 则相当于调用 torch.tensor_split(input, indices_or_sections, dim=1)(split dimension 为 1), 不同之处在于, 如果 indices_or_sections 是一个整数, 它必须均匀地划分 split dimension, 否则将引发运行时错误.

This function is based on NumPy's numpy.hsplit().

Parameters
input (Tensor)– tensor to split.
indices_or_sections (int or list or tuple of ints)– See argument in torch.tensor_split().
"""

```

## vsplit

```python
"""
torch.vsplit(input, indices_or_sections)-> List of Tensors

Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections. Each split is a view of input.
根据 indices_or_sections 将输入(具有两个或多个维度的张量)垂直拆分为多个张量. 每个 split 都是一个 input 视图.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0)(the split dimension is 0), except that if indices_or_sections is an integer it must evenly divide the split dimension or a runtime error will be thrown.
这相当于调用 torch.tensor_split(input, indices_or_sections, dim=0)(split dimension 为 0), 不同之处在于, 如果 indices_or_sections 是整数, 则必须均匀划分 split 维度, 否则将引发运行时错误.

This function is based on NumPy's numpy.vsplit().

Parameters  参数
input (Tensor)– tensor to split.
indices_or_sections (int or list or tuple of ints)– See argument in torch.tensor_split().

"""

```

## select

```python
"""
torch.select(input, dim, index)-> Tensor

Slices the input tensor along the selected dimension at the given index. This function returns a view of the original tensor with the given dimension removed.
在给定索引处沿所选维度对输入张量进行切片. 此函数返回删除了给定维度的原始张量的视图.

Note
If input is a sparse tensor and returning a view of the tensor is not possible, a RuntimeError exception is raised. In this is the case, consider using torch.select_copy()function.
如果 input 是稀疏张量并且无法返回张量的视图, 则会引发 RuntimeError 异常. 在这种情况下, 请考虑使用 torch.select_copy()函数.

Parameters
input (Tensor)– the input tensor.
dim (int)– the dimension to slice
index (int)– the index to select with

Note
select()is equivalent to slicing. For example, tensor.select(0, index)is equivalent to tensor[index] and tensor.select(2, index)is equivalent to tensor[:,:,index].
select()等效于切片. 例如, tensor.select(0, index)等同于 tensor[index], tensor.select(2, index)等同于 tensor[:, :, index].
"""

```

## index_select

```python
"""
torch.index_select(input, dim, index, *, out=None)-> Tensor

Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
返回一个新的张量, 该张量使用 index 中的条目(即 LongTensor)沿维度 dim 为输入张量编制索引.

The returned tensor has the same number of dimensions as the original tensor (input). The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.
返回的张量与原始张量(输入)具有相同的维数. dimth 维度的大小与 length of index 相同; 其他维度的大小与原始 Tensor 中的大小相同.

Note
The returned tensor does not use the same storage as the original tensor. If out has a different shape than expected, we silently change it to the correct shape, reallocating the underlying storage if necessary.
返回的 Tensor 使用与原始 Tensor 相同的存储. 如果 out 的形状与预期不同, 我们会静默地将其更改为正确的形状, 并在必要时重新分配底层存储.

Parameters
input (Tensor)– the input tensor.
dim (int)– the dimension in which we index
index (IntTensor or LongTensor)– the 1-D tensor containing the indices to index

Keyword Arguments
out (Tensor, optional)– the output tensor.
"""

```

## masked_select

```python
"""
torch.masked_select(input, mask, *, out=None)-> Tensor

Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.
返回一个新的 1-D 张量, 该张量根据布尔掩码掩码(即 BoolTensor)为输入张量编制索引.

The shapes of the mask tensor and the input tensor don’t need to match, but they must be broadcastable.
掩码张量和输入张量的形状不需要匹配, 但它们必须是可广播的.

Note
The returned tensor does not use the same storage as the original tensor

Parameters
input (Tensor)– the input tensor.
mask (BoolTensor)– the tensor containing the binary mask to index with

Keyword Arguments
out (Tensor, optional)– the output tensor.

"""

>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
        [-1.2035,  1.2252,  0.5002,  0.6248],
        [ 0.1307, -2.0608,  0.1244,  2.0139]])
>>> mask = x.ge(0.5)
>>> mask
tensor([[False, False, False, False],
        [False, True, True, True],
        [False, False, False, True]])
>>> torch.masked_select(x, mask)
tensor([ 1.2252,  0.5002,  0.6248,  2.0139])

```

## where

```python
"""
torch.where(condition, input, other, *, out=None)-> Tensor

Return a tensor of elements selected from either input or other, depending on condition.

The operation is defined as:

if condition
    return input
else
    return other

Note
The tensors condition, input, other must be broadcastable.

Parameters
condition (BoolTensor)– When True (nonzero), yield input, otherwise yield other
input (Tensor or Scalar)– value (if input is a scalar)or values selected at indices where condition is True
other (Tensor or Scalar)– value (if other is a scalar)or values selected at indices where condition is False

Keyword Arguments
out (Tensor, optional)– the output tensor.

Returns
A tensor of shape equal to the broadcasted shape of condition, input, other

Return type
Tensor

"""

>>> x = torch.randn(3, 2)
>>> y = torch.ones(3, 2)
>>> x
tensor([[-0.4620,  0.3139],
        [ 0.3898, -0.7197],
        [ 0.0478, -0.1657]])
>>> torch.where(x > 0, 1.0, 0.0)
tensor([[0., 1.],
        [1., 0.],
        [1., 0.]])
>>> torch.where(x > 0, x, y)
tensor([[ 1.0000,  0.3139],
        [ 0.3898,  1.0000],
        [ 0.0478,  1.0000]])
>>> x = torch.randn(2, 2, dtype=torch.double)
>>> x
tensor([[ 1.0779,  0.0383],
        [-0.8785, -1.1089]], dtype=torch.float64)
>>> torch.where(x > 0, x, 0.)
tensor([[1.0779, 0.0383],
        [0.0000, 0.0000]], dtype=torch.float64)

```

## take_along_dim

```python
"""
torch.take_along_dim(input, indices, dim=None, *, out=None)-> Tensor

Selects values from input at the 1-dimensional indices from indices along the given dim.
从沿给定维度的索引的 1 维索引处的输入中选择值.

If dim is None, the input array is treated as if it has been flattened to 1d.
如果 dim 为 None, 则 input 数组被视为已展平为 1d.

Functions that return indices along a dimension, like torch.argmax()and torch.argsort(), are designed to work with this function. See the examples below.
返回沿维度返回索引的函数, 如 torch.argmax()和 torch.argsort(), 旨在与此函数一起使用. 请参阅下面的示例.

Note
This function is similar to NumPy's take_along_axis. See also torch.gather().
这个函数类似于 NumPy 的 take_along_axis. 另请参见 torch.gather().

Parameters
input (Tensor)– the input tensor.
indices (tensor)– the indices into input. Must have long dtype.
dim (int, optional)– dimension to select along.

Keyword Arguments
out (Tensor, optional)– the output tensor.

"""

>>> t = torch.tensor([[10, 30, 20], [60, 40, 50]])
>>> max_idx = torch.argmax(t)
>>> torch.take_along_dim(t, max_idx)
tensor([60])
>>> sorted_idx = torch.argsort(t, dim=1)
>>> torch.take_along_dim(t, sorted_idx, dim=1)
tensor([[10, 20, 30],
        [40, 50, 60]])

```

## gather

```python
"""
torch.gather(input, dim, index, *, sparse_grad=False, out=None)-> Tensor

Gathers values along an axis specified by dim.

For a 3-D tensor the output is specified by:

out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

input and index must have the same number of dimensions. It is also required that index.size(d)<= input.size(d)for all dimensions d != dim. out will have the same shape as index. Note that input and index do not broadcast against each other.

Parameters
input (Tensor)– the source tensor
dim (int)– the axis along which to index
index (LongTensor)– the indices of elements to gather

Keyword Arguments
sparse_grad (bool, optional)– If True, gradient w.r.t. input will be a sparse tensor.
out (Tensor, optional)– the destination tensor

"""

>>> t = torch.tensor([[1, 2], [3, 4]])
>>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([[ 1,  1],
        [ 4,  3]])

```

## unsqueeze

```python
"""
torch.unsqueeze(input, dim)-> Tensor

Returns a new tensor with a dimension of size one inserted at the specified position.
返回在指定位置插入维度大小为 1 的新张量.

The returned tensor shares the same underlying data with this tensor.
返回的张量与此张量共享相同的底层数据.

A dim value within the range [-input.dim()- 1, input.dim()+ 1)can be used. Negative dim will correspond to unsqueeze()applied at dim = dim + input.dim()+ 1.
可以使用 [-input.dim()- 1, input.dim()+ 1)范围内的 dim 值. 负 dim 将对应于在 dim = dim + input.dim()+ 1 处应用的 unsqueeze().

Parameters
input (Tensor)– the input tensor.
dim (int)– the index at which to insert the singleton dimension

"""

>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])

```

## tile

```python
"""
torch.tile(input, dims)-> Tensor

Constructs a tensor by repeating the elements of input. The dims argument specifies the number of repetitions in each dimension.
通过重复 input 的元素来构造张量. dims 参数指定每个维度中的重复次数.

If dims specifies fewer dimensions than input has, then ones are prepended to dims until all dimensions are specified. For example, if input has shape (8, 6, 4, 2)and dims is (2, 2), then dims is treated as (1, 1, 2, 2).
如果 dims 指定的维度少于输入的维度, 则 1 将附加到 dims 之前, 直到指定所有维度为止. 例如, 如果输入的形状为 (8,  6,  4,  2) 且 dims 为 (2,  2), 则 dims 将被视为 (1, 1, 2, 2).

Analogously, if input has fewer dimensions than dims specifies, then input is treated as if it were unsqueezed at dimension zero until it has as many dimensions as dims specifies. For example, if input has shape (4, 2)and dims is (3, 3, 2, 2), then input is treated as if it had the shape (1, 1, 4, 2).
类似地, 如果 input 的维度数少于 dims 指定的维度数, 则 input 将被视为在维度 0 处未压缩, 直到它具有 dims 指定的维度数. 例如, 如果输入的形状为 (4, 2), 而 dims 为 (3, 3, 2, 2), 则输入将被视为其形状为 (1, 1, 4, 2).

Note
This function is similar to NumPy's tile function.

Parameters
input (Tensor)– the tensor whose elements to repeat.
dims (tuple)– the number of repetitions per dimension.

"""

>>> x = torch.tensor([1, 2, 3])
>>> x.tile((2,))
tensor([1, 2, 3, 1, 2, 3])
>>> y = torch.tensor([[1, 2], [3, 4]])
>>> torch.tile(y, (2, 2))
tensor([[1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]])



```



* select_scatter 	Embeds the values of the src tensor into input at the given index.

* unravel_index 	Converts a tensor of flat indices into a tuple of coordinate tensors that index into an arbitrary tensor of the specified shape.

* adjoint 	Returns a view of the tensor conjugated and with the last two dimensions transposed.

* argwhere 	Returns a tensor containing the indices of all `non-zero` elements of input.

* conj 	Returns a view of input with a flipped conjugate bit.

* index_add 	See index_add_()for function description.

* index_copy 	See index_add_()for function description.

* index_reduce 	See index_reduce_()for function description.

* movedim 	Moves the dimension(s)of input at the position(s)in source to the position(s)in destination.

* moveaxis 	Alias for torch.movedim().

* narrow 	Returns a new tensor that is a narrowed version of input tensor.

* narrow_copy 	Same as Tensor.narrow()except this returns a copy rather than shared storage.

* nonzero

* scatter 	Out-of-place version of torch.Tensor.scatter_()

* diagonal_scatter 	Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.

* slice_scatter 	Embeds the values of the src tensor into input at the given dimension.

* scatter_add 	Out-of-place version of torch.Tensor.scatter_add_()

* scatter_reduce 	Out-of-place version of torch.Tensor.scatter_reduce_()

* squeeze 	Returns a tensor with all specified dimensions of input of size 1 removed.

* take 	Returns a new tensor with the elements of input at the given indices.

* unbind 	Removes a tensor dimension.


