# torch

* https://pytorch.org/docs/2.5/torch.html

## Tensors

* is_tensor 	Returns True if obj is a `PyTorch tensor`.

* is_storage 	Returns True if obj is a PyTorch `storage object`.

* is_complex 	Returns True if the `data type` of input is a complex data type i.e., one of `torch.complex64`, and `torch.complex128`.

* is_conj 	Returns True if the input is a `conjugated tensor`, i.e. its `conjugate bit` is set to True.
			如果 input 是共轭张量, 即其共轭位设置为 True, 则返回 True.

* is_floating_point 	Returns True if the `data type` of input is a floating point data type i.e., one of `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`.

* is_nonzero 	Returns True if the input is a single element tensor which is not equal to zero after type conversions.
				如果 input 类型转换后是不等于零的单元素张量, 则返回 True.

* set_default_dtype(d) 	Sets the default floating point dtype to d.

* get_default_dtype 	Get the current default floating point `torch.dtype`.

* set_default_device	Sets the default torch.Tensor to be allocated on device.

* get_default_device 	Gets the default torch.Tensor to be allocated on device

* set_default_tensor_type

* numel 	Returns the total number of elements in the input tensor.
			返回 input 张量中的元素总数.

* set_printoptions 	Set options for printing.
					设置打印选项.

* set_flush_denormal 	Disables `denormal floating numbers` on CPU.
						禁用 CPU 上的非正规浮点数.

---

## Creation Ops

> Note
> Random sampling creation ops are listed under Random sampling and include: `torch.rand()` `torch.rand_like()` `torch.randn()` `torch.randn_like()` `torch.randint()` `torch.randint_like()` `torch.randperm()`
> You may also use torch.empty() with the `In-place random sampling` methods to create torch.Tensor s with values sampled from a broader range of distributions.

* tensor 	Constructs a tensor with no `autograd history` (also known as a `"leaf tensor"`, see Autograd mechanics) by copying data.

* sparse_coo_tensor 	Constructs a `sparse tensor` in `COO(rdinate)` format with specified values at the given indices.

* sparse_csr_tensor 	Constructs a `sparse tensor` in `CSR(Compressed Sparse Row)` with specified values at the given crow_indices and col_indices.

* sparse_csc_tensor 	Constructs a `sparse tensor` in `CSC(Compressed Sparse Column)` with specified values at the given ccol_indices and row_indices.

* sparse_bsr_tensor 	Constructs a `sparse tensor` in `BSR(Block Compressed Sparse Row)` with specified 2-dimensional blocks at the given crow_indices and col_indices.

* sparse_bsc_tensor 	Constructs a `sparse tensor` in `BSC(Block Compressed Sparse Column)` with specified 2-dimensional blocks at the given ccol_indices and row_indices.

* asarray 	Converts obj to a tensor.

* as_tensor 	Converts data into a tensor, sharing data and preserving autograd history if possible.

* as_strided 	Create a `view` of an existing torch.Tensor input with specified `size`, `stride` and `storage_offset`.

* from_file 	Creates a CPU tensor with a storage backed by a `memory-mapped file`.

* from_numpy 	Creates a Tensor from a `numpy.ndarray`.

* from_dlpack 	Converts a tensor from an external library into a torch.Tensor.

* frombuffer 	Creates a 1-dimensional Tensor from an object that implements the Python buffer protocol.

* zeros 	Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.

* zeros_like 	Returns a tensor filled with the scalar value 0, with the same size as input.

* ones 	Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.

* ones_like 	Returns a tensor filled with the scalar value 1, with the same size as input.

* arange 	Returns a 1-D tensor of size [(end − start) / step] with values from the interval [start, end] taken with common difference step beginning from start.

* range 	Returns a 1-D tensor of size [(end − start) / step] + 1 with values from start to end with step.

* linspace 	Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
			创建一个 size steps 的一维张量, 其值从 start to end 均匀分布.

* logspace 	Creates a one-dimensional tensor of size steps whose values are evenly spaced from `base^start` to `base^end`, inclusive, on a logarithmic scale with base base.

* eye 	Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

* empty 	Returns a tensor filled with uninitialized data.

* empty_like 	Returns an uninitialized tensor with the same size as input.

* empty_strided 	Creates a tensor with the specified size and stride and filled with undefined data.

* full 	Creates a tensor of size size filled with fill_value.

* full_like 	Returns a tensor with the same size as input filled with fill_value.

* quantize_per_tensor 	Converts a float tensor to a `quantized tensor` with given scale and zero point.

* quantize_per_channel 	Converts a float tensor to a `per-channel quantized tensor` with given scales and zero points.

* dequantize 	Returns an fp32 Tensor by `dequantizing` a quantized Tensor

* complex 	Constructs a `complex tensor` with its real part equal to real and its imaginary part equal to imag.

* polar 	Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value abs and angle angle.

* heaviside 	Computes the Heaviside step function for each element in input.

---

## Indexing, Slicing, Joining, Mutating Ops

* adjoint 	Returns a view of the tensor conjugated and with the last two dimensions transposed.

* argwhere 	Returns a tensor containing the indices of all `non-zero` elements of input.

* cat 	Concatenates the given sequence of seq tensors in the given dimension.

* concat 	Alias of torch.cat().

* concatenate 	Alias of torch.cat().

* conj 	Returns a view of input with a flipped conjugate bit.

* chunk 	Attempts to split a tensor into the specified number of chunks.

* dsplit 	Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.

* column_stack 	Creates a new tensor by horizontally stacking the tensors in tensors.

* dstack 	Stack tensors in sequence depthwise (along third axis).

* gather 	Gathers values along an axis specified by dim.

* hsplit 	Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.

* hstack 	Stack tensors in sequence horizontally (column wise).

* index_add 	See `index_add_()` for function description.

* index_copy 	See `index_add_()` for function description.

* index_reduce 	See `index_reduce_()` for function description.

* index_select 	Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.

* masked_select 	Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.

* movedim 	Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

* moveaxis 	Alias for torch.movedim().

* narrow 	Returns a new tensor that is a narrowed version of input tensor.

* narrow_copy 	Same as Tensor.narrow() except this returns a copy rather than shared storage.

* nonzero

* permute 	Returns a view of the original tensor input with its dimensions permuted.

* reshape 	Returns a tensor with the same data and number of elements as input, but with the specified shape.

* row_stack 	Alias of torch.vstack().

* select 	Slices the input tensor along the selected dimension at the given index.

* scatter 	Out-of-place version of torch.Tensor.scatter_()

* diagonal_scatter 	Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.

* select_scatter 	Embeds the values of the src tensor into input at the given index.

* slice_scatter 	Embeds the values of the src tensor into input at the given dimension.

* scatter_add 	Out-of-place version of torch.Tensor.scatter_add_()

* scatter_reduce 	Out-of-place version of torch.Tensor.scatter_reduce_()

* split 	Splits the tensor into chunks.

* squeeze 	Returns a tensor with all specified dimensions of input of size 1 removed.

* stack 	Concatenates a sequence of tensors along a new dimension.

* swapaxes 	Alias for torch.transpose().

* swapdims 	Alias for torch.transpose().

* t 	Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

* take 	Returns a new tensor with the elements of input at the given indices.

* take_along_dim 	Selects values from input at the 1-dimensional indices from indices along the given dim.

* tensor_split 	Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.

* tile 	Constructs a tensor by repeating the elements of input.

* transpose 	Returns a tensor that is a transposed version of input.

* unbind 	Removes a tensor dimension.

* unravel_index 	Converts a tensor of flat indices into a tuple of coordinate tensors that index into an arbitrary tensor of the specified shape.

* unsqueeze 	Returns a new tensor with a dimension of size one inserted at the specified position.

* vsplit 	Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.

* vstack 	Stack tensors in sequence vertically (row wise).

* where 	Return a tensor of elements selected from either input or other, depending on condition.

---

## Accelerators
加速器

Within the PyTorch repo, we define an "Accelerator" as a `torch.device` that is being used alongside a CPU to speed up computation. These device use an asynchronous execution scheme, using `torch.Stream` and `torch.Event` as their main way to perform synchronization. We also assume that only one such accelerator can be available at once on a given host. This allows us to use the current accelerator as the default device for relevant concepts such as `pinned memory`, `Stream device_type,` `FSDP`, etc.
在 PyTorch 存储库中, 我们将"加速器"定义为 torch.device 与 CPU 一起使用以加快计算速度的加速器. 这些设备使用异步执行方案, 使用 torch.Stream 和 torch.Event 作为执行同步的主要方式. 我们还假设给定主机上一次只能有一个这样的加速器可用. 这允许我们使用当前的加速器作为相关概念的默认设备, 例如固定内存、流device_type、FSDP 等.

As of today, accelerator devices are (in no particular order) "CUDA", "MTIA", "XPU", and PrivateUse1 (many device not in the PyTorch repo itself).
截至今天, 加速器设备是(排名不分先后)"CUDA"、"MTIA"、"XPU"和 PrivateUse1(许多设备不在 PyTorch 存储库本身中).

* Stream 	An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order.
			按先进先出 (FIFO) 顺序异步执行相应任务的顺序队列.

* Event 	Query and record Stream status to identify or control dependencies across Stream and measure timing.
			查询和记录 Stream 状态, 以识别或控制 Stream 中的依赖关系并测量计时.

---

## Generators

* Generator 	Creates and returns a `generator object` that manages the state of the algorithm which produces pseudo random numbers.
				创建并返回一个生成器对象, 该对象管理生成伪随机数的算法的状态.

---

## Random sampling
随机抽样

* seed 	Sets the seed for generating random numbers to a non-deterministic random number on all devices.
		在所有设备上将用于生成随机数的种子设置为非确定性随机数.

* manual_seed 	Sets the seed for generating random numbers on all devices.
				设置用于在所有设备上生成随机数的种子.

* initial_seed 	Returns the initial seed for generating random numbers as a Python long.
				返回用于生成 Python long 形式的随机数的初始种子.

* get_rng_state 	Returns the random number generator state as a `torch.ByteTensor`.

* set_rng_state 	Sets the random number generator state.
					设置随机数生成器状态.

* default_generator 	Returns the default CPU `torch.Generator`

* bernoulli 	Draws binary random numbers (0 or 1) from a Bernoulli distribution.
				从伯努利分布中绘制二进制随机数(0 或 1).

* multinomial 	Returns a tensor where each row contains num_samples indices sampled from the multinomial (a stricter definition would be multivariate, refer to `torch.distributions.multinomial.Multinomial` for more details) probability distribution located in the corresponding row of tensor input.
				返回一个张量, 其中每行都包含 num_samples 从位于相应张量行中的多项式(更严格的定义是多元, 请参阅更多详细信息 torch.distributions.multinomial.Multinomial )概率分布采样的索引 input.

* normal 	Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
			返回从给定平均值和标准差的单独正态分布中提取的随机数张量.

* poisson 	Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input i.e.,
			返回一个张量, 其大小与 input 从泊松分布中采样的每个元素相同, 其中 rate 参数由相应的元素给出

* rand 	Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
		返回一个张量, 其中填充了区间 [0, 1) 上均匀分布的随机数

* rand_like 	Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0, 1)
				返回一个大小 input 相同的张量, 该张量填充了区间 [0, 1) 上均匀分布的随机数.

* randint 	Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
			返回一个张量, 其中填充了在 low(含) 和 high(不含) 之间均匀生成的随机整数.

* randint_like 	Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).
				返回一个与 Tensor 形状相同的张量, input 其中填充了在 low(含) 和 high(不包括) 之间均匀生成的随机整数.

* randn 	Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
			返回一个张量, 该张量填充了均值为 0 且方差为 1 的正态分布中的随机数(也称为标准正态分布).

* randn_like 	Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
				返回一个大小 input 相同的张量, 该张量填充了来自均值为 0 且方差为 1 的正态分布中的随机数.

* randperm 	Returns a random permutation of integers from 0 to n - 1.
			返回整数的随机排列 from 0 to n - 1.

---

### In-place random sampling
就地随机采样

There are a few more in-place random sampling functions defined on Tensors as well. Click through to refer to their documentation:
在 Tensor 上还定义了一些更多的就地随机采样函数. 单击以参考其文档:

* `torch.Tensor.bernoulli_()` - in-place version of torch.bernoulli()
                                 torch.bernoulli() 的就地版本 torch.bernoulli()

* `torch.Tensor.cauchy_()` - numbers drawn from the Cauchy distribution
                             从 Cauchy 分布中抽取的数字

* `torch.Tensor.exponential_()` - numbers drawn from the exponential distribution
                                  从指数分布中抽取的数字

* `torch.Tensor.geometric_()` - elements drawn from the geometric distribution
                                从几何分布中提取的元素

* `torch.Tensor.log_normal_()` - samples from the log-normal distribution
                                 来自对数正态分布的样本

* `torch.Tensor.normal_()` - in-place version of torch.normal()
                             torch.normal() 的就地版本 torch.normal()

* `torch.Tensor.random_()` - numbers sampled from the discrete uniform distribution
                             从离散均匀分布中采样的数字

* `torch.Tensor.uniform_()` - numbers sampled from the continuous uniform distribution
                              从连续均匀分布中采样的数字

---

### Quasi-random sampling
准随机采样

* quasirandom.SobolEngine 	The torch.quasirandom.SobolEngine is an engine for generating (scrambled) Sobol sequences.
							这是一个 `torch.quasirandom.SobolEngine` 用于生成 (打乱) Sobol 序列的引擎.

---

## Serialization

* save 	Saves an object to a disk file.

* load 	Loads an object saved with torch.save() from a file.

---

## Parallelism

* get_num_threads 	Returns the number of threads used for parallelizing CPU operations

* set_num_threads 	Sets the number of threads used for `intraop parallelism` on CPU.

* get_num_interop_threads 	Returns the number of threads used for interop parallelism on CPU

* set_num_interop_threads 	Sets the number of threads used for interop parallelism

---

## Locally disabling gradient computation
在本地禁用梯度计算

The context managers `torch.no_grad()`, `torch.enable_grad()`, and `torch.set_grad_enabled()` are helpful for locally disabling and enabling gradient computation. See Locally disabling gradient computation for more details on their usage. These context managers are thread local, so they won’t work if you send work to another thread using the threading module, etc.
上下文管理器 torch.no_grad()、 torch.enable_grad() 和 torch.set_grad_enabled() 有助于在本地禁用和启用梯度计算. 有关其用法的更多详细信息, 请参阅本地禁用梯度计算. 这些上下文管理器是线程本地的, 因此如果您使用 threading 模块等将工作发送到另一个线程, 它们将不起作用.

```python

>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

* no_grad 	Context-manager that disables gradient calculation.

* enable_grad 	Context-manager that enables gradient calculation.

* autograd.grad_mode.set_grad_enabled 	Context-manager that sets gradient calculation on or off.

* is_grad_enabled 	Returns True if grad mode is currently enabled.

* autograd.grad_mode.inference_mode 	Context-manager that enables or disables inference mode.

* is_inference_mode_enabled 	Returns True if inference mode is currently enabled.

---

## Math operations

### Pointwise Ops

### Reduction Ops

### Comparison Ops

### Spectral Ops

### Other Operations

### BLAS and LAPACK Operations

### Foreach Operations

---

## Utilities

* compiled_with_cxx11_abi 	Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1
							返回 PyTorch 是否是使用 _GLIBCXX_USE_CXX11_ABI=1 构建的

* result_type 	Returns the torch.dtype that would result from performing an arithmetic operation on the provided input tensors.
				返回 torch.dtype 对提供的输入张量执行算术运算的结果.

* can_cast 	Determines if a type conversion is allowed under PyTorch casting rules described in the type promotion documentation.
			确定类型提升文档中描述的 PyTorch 强制转换规则是否允许类型转换.

* promote_types 	Returns the torch.dtype with the smallest size and scalar kind that is not smaller nor of lower kind than either type1 or type2.
					返回具有最小大小和标量种类的,  torch.dtype 该标量种类不小于 type1 或 type2.

* use_deterministic_algorithms 	Sets whether PyTorch operations must use "deterministic" algorithms.
								设置 PyTorch 操作是否必须使用 "确定性" 算法.

* are_deterministic_algorithms_enabled 	Returns True if the global deterministic flag is turned on.
										如果全局确定性标志处于打开状态, 则返回 True.

* is_deterministic_algorithms_warn_only_enabled 	Returns True if the global deterministic flag is set to warn only.
													如果全局确定性标志设置为 warn only, 则返回 True.

* set_deterministic_debug_mode 	Sets the debug mode for deterministic operations.
								设置确定性操作的调试模式.

* get_deterministic_debug_mode 	Returns the current value of the debug mode for deterministic operations.
								返回确定性作的调试模式的当前值.

* set_float32_matmul_precision 	Sets the internal precision of float32 matrix multiplications.
								设置 float32 矩阵乘法的内部精度.

* get_float32_matmul_precision 	Returns the current value of float32 matrix multiplication precision.
								返回 float32 矩阵乘法精度的当前值.

* set_warn_always 	When this flag is False (default) then some PyTorch warnings may only appear once per process.
					当此标志为 False (默认) 时, 某些 PyTorch 警告可能每个进程只出现一次.

* get_device_module 	Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
						返回与给定设备关联的模块(例如, torch.device('cuda'),  "mtia：0",  "xpu",  ...).

* is_warn_always_enabled 	Returns True if the global warn_always flag is turned on.
							如果打开了全局 warn_always 标志, 则返回 True.

* vmap 	vmap is the vectorizing map; vmap(func) returns a new function that maps func over some dimension of the inputs.
		vmap 是矢量化映射; vmap(func) 返回一个映射 func 输入的某个维度的新函数.

* _assert 	A wrapper around Python's assert which is symbolically traceable.
			Python 的 assert 的包装器, 可符号地跟踪.

---

## Symbolic Numbers
符号数字

```python

class torch.SymInt(node)

class torch.SymFloat(node)

class torch.SymBool(node)

```

* sym_float 	SymInt-aware utility for float casting.

* sym_int 	SymInt-aware utility for int casting.

* sym_max 	SymInt-aware utility for max which avoids branching on a < b.

* sym_min 	SymInt-aware utility for min().

* sym_not 	SymInt-aware utility for logical negation.

* sym_ite

---

## Export Path

---

## Control Flow

* cond 	Conditionally applies true_fn or false_fn.

---

## Optimizations

* compile 	Optimizes given model/function using TorchDynamo and specified backend.
			使用 TorchDynamo 和指定的后端优化给定的模型/函数.

---

## Operator Tags

```python

class torch.Tag

```

## todo

* size, stride and storage_offset
* storage object
* set_default_tensor_type 和 set_default_dtype 区别
* sparse tensor
* COO(rdinate)
* CSR(Compressed Sparse Row)
* CSC(Compressed Sparse Column)
* BSR(Block Compressed Sparse Row)
* quantized tensor
* per-channel quantized tensor
* dequantizing

