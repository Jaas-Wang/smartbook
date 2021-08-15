# A Neural Net from the Foundations

# 基础的神经网络

This chapter begins a journey where we will dig deep into the internals of the models we used in the previous chapters. We will be covering many of the same things we've seen before, but this time around we'll be looking much more closely at the implementation details, and much less closely at the practical issues of how and why things are as they are.

本章节会开始一段路程，在这里我们将使用之前章节提供的内容深入挖掘模型的内部。我们会解释之前我们已经学习过的很多相同内容，但这次的范围我们会更仔细的关注于细节的实现，同时更少贴近于为什么事情是那样的实际问题。

We will build everything from scratch, only using basic indexing into a tensor. We'll write a neural net from the ground up, then implement backpropagation manually, so we know exactly what's happening in PyTorch when we call `loss.backward`. We'll also see how to extend PyTorch with custom *autograd* functions that allow us to specify our own forward and backward computations.

我们将从零开始创建所有内容，只使用基础索引到张量中。我们将从基础开始编写一个神经网络，然后手动实现反向传播，所以当我们调用`loss.backward`时就能够准确的知道PyTorch里发生了什么。我们也会学习扩展PyTorch，用自定义*autograd*函数允许我们详细说明我们自己的正向和反向计算。

## Building a Neural Net Layer from Scratch

## 从零创建一个神经网络层

Let's start by refreshing our understanding of how matrix multiplication is used in a basic neural network. Since we're building everything up from scratch, we'll use nothing but plain Python initially (except for indexing into PyTorch tensors), and then replace the plain Python with PyTorch functionality once we've seen how to create it.

让我们通过刷新在基础神经网络如何使用矩阵乘法的理解开始。既然我们从零开始构建所有内容，最初我们只会使用的纯Python语言而不会使用其它任何功能（除了索引到PyTorch张量），其后一旦我们学习了如何创建它，就用PyTorch功能替换纯Python代码。

### Modeling a Neuron

### 建模神经元

A neuron receives a given number of inputs and has an internal weight for each of them. It sums those weighted inputs to produce an output and adds an inner bias. In math, this can be written as:

$$ out = \sum_{i=1}^{n} x_{i} w_{i} + b$$

if we name our inputs$(x_{1},\dots,x_{n})$, our weights $(w_{1},\dots,w_{n})$, and our bias $b$. In code this translates into:

一个神经元接收给一些定的输入，且对每个输入都有一个内部的权重。神经元合计那些加权输入生成一个输出，并加上一个内部的偏差。这个过程能够用数学公式表示为：

$$ out = \sum_{i=1}^{n} x_{i} w_{i} + b$$

如果我们命名我们的输入$(x_{1},\dots,x_{n})$，权重s $(w_{1},\dots,w_{n})$，和偏差$b$。这转化为代码：

```python
output = sum([x*w for x,w in zip(inputs,weights)]) + bias
```

This output is then fed into a nonlinear function called an *activation function* before being sent to another neuron. In deep learning the most common of these is the *rectified Linear unit*, or *ReLU*, which, as we've seen, is a fancy way of saying:

然后这个输出在被发送到其它神经元之前会传递给一个称为*激活函数*的非线性函数。在深度学习中，最常用的是*线性整流函数*或*ReLU*，正如我们学习过的，它是一个奇特的方法：

```python
def relu(x): return x if x >= 0 else 0
```

A deep learning model is then built by stacking a lot of those neurons in successive layers. We create a first layer with a certain number of neurons (known as *hidden size*) and link all the inputs to each of those neurons. Such a layer is often called a *fully connected layer* or a *dense layer* (for densely connected), or a *linear layer*.

It requires to compute, for each `input` in our batch and each neuron with a give `weight`, the dot product:

在连续的层中深度学习模型然后通过堆积很多那种神经元被创建。我们用确定数量的神经元创建了第一层（称为*隐含大小*）和连接所有的输入到那些每个神经元上。如此一个层通常被称为一个*全连接层*或一个*稠密层*（稠密连接）或一个*线性层*。

对我们批次中的每个`输入`和用给定`权重`的每个神经元，需要来计算点积：

```python
sum([x*w for x,w in zip(input,weight)])
```

If you have done a little bit of linear algebra, you may remember that having a lot of those dot products happens when you do a *matrix multiplication*. More precisely, if our inputs are in a matrix `x` with a size of `batch_size` by `n_inputs`, and if we have grouped the weights of our neurons in a matrix `w` of size `n_neurons` by `n_inputs` (each neuron must have the same number of weights as it has inputs) and all the biases in a vector `b` of size `n_neurons`, then the output of this fully connected layer is:

如果你学过一些线性代数，你可能记得当你做一个*矩阵乘法*时发生很多那种点积计算。更恰当的说，如果我们的输入是在`n_inputs`乘`batch_size`大小的矩阵`x`中，及如果我们在`n_inputs`乘`n_neurons`大小的矩阵`w`中已经分好我们神经元的权重组（每个神经元权重数量必须与它的输入相同）和`n_neurons`大小的向量`b`中是所有的偏差。那么这个全连接的输出是：

```python
y = x @ w.t() + b
```

where `@` represents the matrix product and `w.t()` is the transpose matrix of `w`. The output `y` is then of size `batch_size` by `n_neurons`, and in position `(i,j)` we have (for the mathy folks out there):

$$y_{i,j} = \sum_{k=1}^{n} x_{i,k} w_{k,j} + b_{j}$$

Or in code:

`@`代表矩阵乘法，`w.t()`是变换矩阵`w`。输出`y`是`n_neurons`乘`batch_size`大小，在`(i,j)`这个位置上，我们用公式表示（对于数学家来说）：

$$y_{i,j} = \sum_{k=1}^{n} x_{i,k} w_{k,j} + b_{j}$$

或代码：

```python
y[i,j] = sum([a * b for a,b in zip(x[i,:],w[j,:])]) + b[j]
```

The transpose is necessary because in the mathematical definition of the matrix product `m @ n`, the coefficient `(i,j)` is:

因为在矩阵乘积`m @ n`的数学定义中转置是必须的，相应的系数`(i,j)`是：

```python
sum([a * b for a,b in zip(m[i,:],n[:,j])])
```

So the very basic operation we need is a matrix multiplication, as it's what is hidden in the core of a neural net.

所以我们需要矩阵乘法这个非常基本的运算，因为它隐藏在神经网络的核心。

### Matrix Multiplication from Scratch

### 从零开始实现矩阵乘法

Let's write a function that computes the matrix product of two tensors, before we allow ourselves to use the PyTorch version of it. We will only use the indexing in PyTorch tensors:

在我们允许自己使用PyTorch版本功能之前，让我们编写计算两个张量矩阵乘积的函数。我们只会使用PyTorch张量中的索引：

实验代码：

```
import torch
from torch import tensor
```

We'll need three nested `for` loops: one for the row indices, one for the column indices, and one for the inner sum. `ac` and `ar` stand for number of columns of `a` and number of rows of `a`, respectively (the same convention is followed for `b`), and we make sure calculating the matrix product is possible by checking that `a` has as many columns as `b` has rows:

我们将使用三个嵌套`for`循环：一个是指代行，一个指代列，一个为内部的和。`ac`和`ar`分别代表`a`列数和`a`的行数（同对`b`遵循同样的规则），通过检查`a`的列数与`b`的列数是相同的，我们来确保矩阵乘积的计算是可行的：

实验代码：

```
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): c[i,j] += a[i,k] * b[k,j]
    return c
```

To test this out, we'll pretend (using random matrices) that we're working with a small batch of 5 MNIST images, flattened into 28×28 vectors, with linear model to turn them into 10 activations:

来测试这个输出，我们模拟（使用随机矩阵）处理 5 张MNIST图像的小批次数据，摊平到 28×28 的向量中，用线性模型变换它们为 10 个激活：

实验代码：

```
m1 = torch.randn(5,28*28)
m2 = torch.randn(784,10)
```

Let's time our function, using the Jupyter "magic" command `%time`:

让使用Jupyter“magic”命令`%time`，我们对功能进行记时：

实验代码：

```
%time t1=matmul(m1, m2)
```

实验输出：

​			CPU times: user 1.15 s, sys: 4.09 ms, total: 1.15 s
​			Wall time: 1.15 s

And see how that compares to PyTorch's built-in `@`:

并看一下这个功能与PyTorch的内置`@`对比如何的：

实验代码：

```
%timeit -n 20 t2=m1@m2
```

实验输出：

​			14 µs ± 8.95 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)	

As we can see, in Python three nested loops is a very bad idea! Python is a slow language, and this isn't going to be very efficient. We see here that PyTorch is around 100,000 times faster than Python—and that's before we even start using the GPU!

你可以看出，在Python中三个嵌套循环是个非常糟糕的注意！Python是一个慢语言，且不是非常有效率。我们看到在这里PyTorch差不多比Python快大约 100,000 倍，且那是我们在开始使用GPU之前！

Where does this difference come from? PyTorch didn't write its matrix multiplication in Python, but rather in C++ to make it fast. In general, whenever we do computations on tensors we will need to *vectorize* them so that we can take advantage of the speed of PyTorch, usually by using two techniques: elementwise arithmetic and broadcasting.

这个区别来自什么地方？PyTorch不是用Python编写的矩阵乘法，而是C++以使它很快。无论什么时候我们在张量上做计算，我们将需要*向量化*它们，所以我们可以采纳PyTorch的速度优势，通常使用两个技术：按元素计算和广播。

### Elementwise Arithmetic

### 按元素计算

All the basic operators (`+`, `-`, `*`, `/`, `>`, `<`, `==`) can be applied elementwise. That means if we write `a+b` for two tensors `a` and `b` that have the same shape, we will get a tensor composed of the sums the elements of `a` and `b`:

所有基础运算符 (`+`, `-`, `*`, `/`, `>`, `<`, `==`) 能够被按元素应用。这表示如果我们编写两个具有相同形状的张量`a`和`b`，我们会得到一个由`a`和`b`的元素合计组成的张量：

实验代码：

```
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a + b
```

实验输出：

```
tensor([12., 14.,  3.])
```

The Booleans operators will return an array of Booleans:

布尔操作符会返回布尔数组：

实验代码：

```
a < b
```

实验输出：

```
tensor([False,  True,  True])
```

If we want to know if every element of `a` is less than the corresponding element in `b`, or if two tensors are equal, we need to combine those elementwise operations with `torch.all`:

如果我们想知道`a`的每个元素是否比`b`中的相应元素小，或两个张量是否是相等的，我们需要用`torch.all`组合那些按元素计算：

实验代码：

```
(a < b).all(), (a==b).all()
```

实验输出：

```
(tensor(False), tensor(False))
```

Reduction operations like `all()`, `sum()` and `mean()` return tensors with only one element, called rank-0 tensors. If you want to convert this to a plain Python Boolean or number, you need to call `.item()`:

像`all()`，`sum()`和`mean()`这样的简化运算会返回只有一个元素的张量，被称为 0 阶张量。如果你想把它转换为一个普通的Python的布尔值或数值，你需要调用`.item()`：

实验代码：

```
(a + b).mean().item()
```

实验输出：

```
9.666666984558105
```

The elementwise operations work on tensors of any rank, as long as they have the same shape:

按元素运算可在任何阶的张量上运行，只要它们有相同的形状：

实验代码：

```
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m*m
```

实验输出：

```
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.],
        [49., 64., 81.]])
```

However you can't perform elementwise operations on tensors that don't have the same shape (unless they are broadcastable, as discussed in the next section):

然而我们不能够在没有相同形态的张量上执行按元素计算（除非它们是可传播的，在下节会讨论）：

实验代码：

```
n = tensor([[1., 2, 3], [4,5,6]])
m*n
```

实验输出:

​			RuntimeError                           Traceback (most recent call last)
​			<ipython-input-12-add73c4f74e0> in <module>
​						      1 n = tensor([[1., 2, 3], [4,5,6]])
​			----> 2 m*n
​			RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

With elementwise arithmetic, we can remove one of our three nested loops: we can multiply the tensors that correspond to the `i`-th row of `a` and the `j`-th column of `b` before summing all the elements, which will speed things up because the inner loop will now be executed by PyTorch at C speed.

用按元素计算，我们能够移除三个嵌套循环的一个：我们能够在所有元素加总之前，乘以`a`的第`i`行和`b`的第`j`列的相应张量，它会提升速度，因为内部循环现在会通过PyTorch以 C 的速度执行。

To access one column or row, we can simply write `a[i,:]` or `b[:,j]`. The `:` means take everything in that dimension. We could restrict this and take only a slice of that particular dimension by passing a range, like `1:5`, instead of just `:`. In that case, we would take the elements in columns or rows 1 to 4 (the second number is noninclusive).

进入一列或行，我们能够简单的写为`a[i,:]`或`b[:,j]`。`:`表示取那个维度中的所有内容。我们能够限制这个操作，通过传递一个范围来取特定维的一个切片，如用`1:5`来替代`:`。在这种情况下，我们会取 1 到 4 列或行中的元素（第二个数值是非包含性的）

One simplification is that we can always omit a trailing colon, so `a[i,:]` can be abbreviated to `a[i]`. With all of that in mind, we can write a new version of our matrix multiplication:

一个计划的操作，我们能够总是删除一个尾部的冒号，如`a[i,:]`能够被缩写为`a[i]`。了解了这些之后，我们能够写出矩阵乘法新的版本：

实验代码：

```
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = (a[i] * b[:,j]).sum()
    return c
```

实验代码：

```
%timeit -n 20 t3 = matmul(m1,m2)
```

实验输出：

​			1.7 ms ± 88.1 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)

We're already ~700 times faster, just by removing that inner `for` loop! And that's just the beginning—with broadcasting we can remove another loop and get an even more important speed up.

只是移除了内部的`for`循环，我们就已经快了大约 ~700倍！这只是个开始，利用传播我们能够移除其它的循环并取得更重要的加速。

### Broadcasting

### 传播

As we discussed in <chapter_mnist_basics>, broadcasting is a term introduced by the [NumPy library](https://docs.scipy.org/doc/) that describes how tensors of different ranks are treated during arithmetic operations. For instance, it's obvious there is no way to add a 3×3 matrix with a 4×5 matrix, but what if we want to add one scalar (which can be represented as a 1×1 tensor) with a matrix? Or a vector of size 3 with a 3×4 matrix? In both cases, we can find a way to make sense of this operation.

我们在<第四章：mnist基础>中讨论过传播，它是由[NumPy库](https://docs.scipy.org/doc/)引入的，描述了在计算运算期间如何处理不同阶张量。例如，很明显把 3×3 矩阵和 4×5 矩阵进行相加，但是如果我们希望用一个矩阵与一个标量（可以表示为一个 1×1 张量）相加呢？或用 3×4 矩阵与大小为 3 的向量相加？在这两个案例中，我们能够寻找一个方法来搞清楚这个运算。

Broadcasting gives specific rules to codify when shapes are compatible when trying to do an elementwise operation, and how the tensor of the smaller shape is expanded to match the tensor of the bigger shape. It's essential to master those rules if you want to be able to write code that executes quickly. In this section, we'll expand our previous treatment of broadcasting to understand these rules.

当尝试做按元素运算在形状兼容时传播给出具体的规则，及小形状张量如何扩展以匹配大形状张量。如果你想能够编写快速执行的代码，熟悉那些规则是必须的。在本小节，我们把之前处理的传播展开来讲以理解这些规则。

#### Broadcasting with a scalar

#### 用标量传播

Broadcasting with a scalar is the easiest type of broadcasting. When we have a tensor `a` and a scalar, we just imagine a tensor of the same shape as `a` filled with that scalar and perform the operation:

用标题传播是最容易的传播类型。当我们有一个张量`a`和一个标题，我们只是想像一个与`a`相同形状的张量来填满标量并执行运算：

实验代码：

```
a = tensor([10., 6, -4])
a > 0
```

实验输出：

```
tensor([ True,  True, False])
```

How are we able to do this comparison? `0` is being *broadcast* to have the same dimensions as `a`. Note that this is done without creating a tensor full of zeros in memory (that would be very inefficient).

我们如何能够做这个对比？`0`被*传播*以有与`a`相关的维度。注意没有在内存中创建一个全是零的张量做完这个事情（那会非常没有效率）。

This is very useful if you want to normalize your dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar):

如果你想通过从整个数据集（一个矩阵）中减去平均值（一个标量）并除以偏差（另一个标题）来标准化是非常有用处的：

实验代码：

```
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
(m - 5) / 2.73
```

实验输出：

```
tensor([[-1.4652, -1.0989, -0.7326],
        [-0.3663,  0.0000,  0.3663],
        [ 0.7326,  1.0989,  1.4652]])
```

What if have different means for each row of the matrix? in that case you will need to broadcast a vector to a matrix.

如果对于矩阵的每行都有不同的平均值呢？在这种情况下我们会需要传播向量到矩阵。

#### Broadcasting a vector to a matrix

#### 传播向量到矩阵

We can broadcast a vector to a matrix as follows:

我们能够用下面的代码传播向量到矩阵：

实验代码：

```
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m.shape,c.shape
```

实验输出：

```
(torch.Size([3, 3]), torch.Size([3]))
```

实验代码：

```
m + c
```

实验输出：

```
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
```

Here the elements of `c` are expanded to make three rows that match, making the operation possible. Again, PyTorch doesn't actually create three copies of `c` in memory. This is done by the `expand_as` method behind the scenes:

这里`c`的元素被扩展以生成三行匹配，使得运算成为可能。同样的，PyTorch没有实际在内存中创建`c`的三个拷贝。这是通过`expand_as`方法后台完成的：

实验代码：

```
c.expand_as(m)
```

实验输出：

```
tensor([[10., 20., 30.],
        [10., 20., 30.],
        [10., 20., 30.]])
```

If we look at the corresponding tensor, we can ask for its `storage` property (which shows the actual contents of the memory used for the tensor) to check there is no useless data stored:

如果想你看一下相对应的张量，你可以请求它的`storage`属性（展示用于张量的实际内存内容）来检查没有多于的存储数据：

实验代码：

```
t = c.expand_as(m)
t.storage()
```

实验输出：

```
 10.0
 20.0
 30.0
[torch.FloatStorage of size 3]
```

Even though the tensor officially has nine elements, only three scalars are stored in memory. This is possible thanks to the clever trick of giving that dimension a *stride* of 0 (which means that when PyTorch looks for the next row by adding the stride, it doesn't move):

虽然即使正式的张量有九个元素，在内存中只存储了三个标量。这可能要归功于提供的维度  0 *步长*的聪明技巧（这表示当PyTorch通过增加步进查找下一行时，它不会移动）：

实验代码：

```
t.stride(), t.shape
```

实验输出：

```
((0, 1), torch.Size([3, 3]))
```

Since `m` is of size 3×3, there are two ways to do broadcasting. The fact it was done on the last dimension is a convention that comes from the rules of broadcasting and has nothing to do with the way we ordered our tensors. If instead we do this, we get the same result:

因为`m`是 3×3 大小，有两个方法来做传播。事实上，在最后一个维度上完成传播是来自传播规则的一个惯例，且用这个方法我们不需要安排张量做任何事情，如果我们用这个方法，我会会获得相同的结果：

实验代码：

```
c + m
```

实验输出：

```
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
```

In fact, it's only possible to broadcast a vector of size `n` with a matrix of size `m` by `n`:

事实上，这一方法只可能用`m × n`大小的矩阵来传播一个`n`大小的向量：

实验代码：

```
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```

实验输出：

```
tensor([[11., 22., 33.],
        [14., 25., 36.]])
```

This won't work:

下面的情况，这一方法就不起作用了：

实验代码：

```
c = tensor([10.,20])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```

实验输出：

​			RuntimeError                           Traceback (most recent call last)
​			<ipython-input-25-64bbbad4d99c> in <module>
​				      1 c = tensor([10.,20])
​			    	  2 m = tensor([[1., 2, 3], [4,5,6]])
​			----> 3 c+m
​			RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

If we want to broadcast in the other dimension, we have to change the shape of our vector to make it a 3×1 matrix. This is done with the `unsqueeze` method in PyTorch:

如果我们想在其它维度上做传播，我们必须改变我们向量的形状，使为成为一个 2×1 矩阵。用PyTorch的`unsqueeze`方法作为这个事情：

实验代码：

```
c = tensor([10.,20])
m = tensor([[1., 2, 3], [4,5,6]])
c = c.unsqueeze(1)
m.shape,c.shape
```

实验输出：

```
(torch.Size([2, 3]), torch.Size([2, 1]))
```

This time, `c` is expanded on the column side:

这样，`c`在列上做了扩展：

实验代码：

```
c+m
```

实验输出：

```
tensor([[11., 12., 13.],
        [24., 25., 26.]])
```

Like before, only two scalars are stored in memory:

像之前一样，只有两个标量存贮在内存中：

实验代码：

```
t = c.expand_as(m)
t.storage()
```

实验输出：

```
 10.0
 20.0
[torch.FloatStorage of size 2]
```

And the expanded tensor has the right shape because the column dimension has a stride of 0:

且扩展张量以使它有正确的形状，因为列维度有 0 的步长：

实验代码：

```
t.stride(), t.shape
```

实验输出：

```
((1, 0), torch.Size([2, 3]))
```

With broadcasting, by default if we need to add dimensions, they are added at the beginning. When we were broadcasting before, PyTorch was doing `c.unsqueeze(0)` behind the scenes:

利用传播，如果我们需要以默认的方式添加维度，它们会在开始就添加。当我们进行传播之前，PyTorchd在后台就做了`c.unsqueeze(0)`：

实验代码：

```
c = tensor([10.,20,30])
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
```

实验输出：

```
(torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```

The `unsqueeze` command can be replaced by `None` indexing:

`unsqueeze`可以用`None`索引替换：

实验代码：

```
c.shape, c[None,:].shape,c[:,None].shape
```

实验输出：

```
(torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```

You can always omit trailing colons, and `...` means all preceding dimensions:

我们可以总是忽略尾部的冒号，且`...`表示所有前面的维度：

实验代码：

```
c[None].shape,c[...,None].shape
```

实验输出：

```
(torch.Size([1, 3]), torch.Size([3, 1]))
```

With this, we can remove another `for` loop in our matrix multiplication function. Now, instead of multiplying `a[i]` with `b[:,j]`, we can multiply `a[i]` with the whole matrix `b` using broadcasting, then sum the results:

利用这个方法，我们可以在我们的矩阵乘法函数中移除另外一个`for`循环。现在，不是用`a[i]`和`b[:,j]`相乘，相替代的我们可以使用传播让`a[i]`与整个矩阵`b`相乘，然后合计结果：

实验代码：

```
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c
```

实验代码：

```
%timeit -n 20 t4 = matmul(m1,m2)
```

实验输出：

​			357 µs ± 7.2 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)

We're now 3,700 times faster than our first implementation! Before we move on, let's discuss the rules of broadcasting in a little more detail.

现在我们比第一次实现的代码快了3,700倍！在我们继续向后学习前，让我们更详细一点来讨论传播规则。

#### Broadcasting rules

#### 传播规则

When operating on two tensors, PyTorch compares their shapes elementwise. It starts with the *trailing dimensions* and works its way backward, adding 1 when it meets empty dimensions. Two dimensions are *compatible* when one of the following is true:

- They are equal.
- One of them is 1, in which case that dimension is broadcast to make it the same as the other.

当在两个张量上运算时，PyTorch会按元素对比它们的形状。它从*尾部维度*开始且它的处理方式是反向的，当它遇到空维度会加 1 。当满足下面条件之一时两个维度就是兼任的：

- 它们是相等的。
- 它们其中一个是 1 ，在这样的情况下维度被传播以使它与另一个相同。

Arrays do not need to have the same number of dimensions. For example, if you have a 256×256×3 array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with three values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

数组不需要有相同数量的维度。例如，如果我有一个RGB值的 256×256×3 数组，你想需要用不同的值来缩放图像中的每个颜色，你可以图像乘以有三个值的一维数组。根据传播规则排列这些数组的尾部坐标轴的大小，显示它们是兼容的：

```
Image  (3d tensor): 256 x 256 x 3
Scale  (1d tensor):  (1)   (1)  3
Result (3d tensor): 256 x 256 x 3
```

However, a 2D tensor of size 256×256 isn't compatible with our image:

然而，256×256 大小的2D张量与我们的图像是不兼容的：

```
Image  (3d tensor): 256 x 256 x   3
Scale  (2d tensor):  (1)  256 x 256
Error
```

In our earlier examples we had with a 3×3 matrix and a vector of size 3, broadcasting was done on the rows:

在我们之前的例子中，我们有一个 3×3 矩阵和一个大小为 3 的向量，传播在行上完成了：

```
Matrix (2d tensor):   3 x 3
Vector (1d tensor): (1)   3
Result (2d tensor):   3 x 3
```

As an exercise, try to determine what dimensions to add (and where) when you need to normalize a batch of images of size `64 x 3 x 256 x 256` with vectors of three elements (one for the mean and one for the standard deviation).

Another useful way of simplifying tensor manipulations is the use of Einstein summations convention.

做这样一个练习，当我们需要用三个元素（一个为平均值，一个为标准差）向量标准化一个 `64 x 3 x 256 x 256` 大小图像批次的时候，尝试确定添加的维度（和添加的位置）。

别一个简化张量乘法的有用方法是使用爱因斯坦求和约定。

### Einstein Summation

### 爱因斯坦求和

Before using the PyTorch operation `@` or `torch.matmul`, there is one last way we can implement matrix multiplication: Einstein summation (`einsum`). This is a compact representation for combining products and sums in a general way. We write an equation like this:

在使用PyTorch运算`@`或``torch.matmul`之前，我们有一个最后的方法可以实现矩阵乘法：爱因斯坦求和（`einsum`）。这是简洁的代表，用普通的方法来组装乘积和合计。我们写了下面的这个等式：

```
ik,kj -> ij
```

The lefthand side represents the operands dimensions, separated by commas. Here we have two tensors that each have two dimensions (`i,k` and `k,j`). The righthand side represents the result dimensions, so here we have a tensor with two dimensions `i,j`.

左侧代表的是操作数据的维度，以逗号做了分割。在这里我们有两个二维张量（`i,k`和`k,j`）。右侧代表结果维度，所以在这边我们有一个二维`i,j`的张量。

The rules of Einstein summation notation are as follows:

1. Repeated indices on the left side are implicitly summed over if they are not on the right side.
2. Each index can appear at most twice on the left side.
3. The unrepeated indices on the left side must appear on the right side.

爱因斯坦求和记法的规则如下：

1. 在左侧的重复索引被隐含求和，如果它们不在右侧话。
2. 每个索引可以在左侧最多出现两次。
3. 在左侧非重复索引必须显示在右侧。

So in our example, since `k` is repeated, we sum over that index. In the end the formula represents the matrix obtained when we put in `(i,j)` the sum of all the coefficients `(i,k)` in the first tensor multiplied by the coefficients `(k,j)` in the second tensor... which is the matrix product! Here is how we can code this in PyTorch:

因为`k`是重复的，所以在我们的例子中，我们求和那个索引。当我们输入`(i,j)`时，第一张量中所有系数`(i,k)`的合乘以第二张量中系数`(k,j)`... 它就是矩阵乘积！下面是我们可以如何在PyTorch中编写代码：

实验代码：

```
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```

Einstein summation is a very practical way of expressing operations involving indexing and sum of products. Note that you can have just one member on the lefthand side. For instance, this:

爱因斯坦求和是一个涉及到索引和乘积求和表达式运算非常实用的方法。注意在左侧我们能够只有一个成员。例如，下面所示：

```python
torch.einsum('ij->ji', a)
```

returns the transpose of the matrix `a`. You can also have three or more members. This:

返回了矩阵`a`的转置。我们也能够有三个或更多的成员，如下：

```python
torch.einsum('bi,ij,bj->b', a, b, c)
```

will return a vector of size `b` where the `k`-th coordinate is the sum of `a[k,i] b[i,j] c[k,j]`. This notation is particularly convenient when you have more dimensions because of batches. For example, if you have two batches of matrices and want to compute the matrix product per batch, you would could this:

这会返回大小为`b`的向量，第`k`坐标的位置是`a[k,i] b[i,j] c[k,j]`的合计。由于批次的原因你有更多维度的时候，这一记法是特别的方便。例如，如果你有两批次的矩阵且希望计算每个批次的矩阵乘法，你可以这样做：

```python
torch.einsum('bik,bkj->bij', a, b)
```

Let's go back to our new `matmul` implementation using `einsum` and look at its speed:

让我返回到我们使用`einsum`新的`matmul`实现，并看一下它的速度：

实验代码：

```
%timeit -n 20 t5 = matmul(m1,m2)
```

实验输出：

​			68.7 µs ± 4.06 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)

As you can see, not only is it practical, but it's *very* fast. `einsum` is often the fastest way to do custom operations in PyTorch, without diving into C++ and CUDA. (But it's generally not as fast as carefully optimized CUDA code, as you see from the results in "Matrix Multiplication from Scratch".)

正如你看到的，它不仅是实用的，而且它*非常* 快。在PyTorch中，`einsum`通常是做自定义运算最快的方法，无需深入研究C++和CUDA。（如你看到来自“从零开始实现矩阵乘法”中的结果，但它通常不会与专门针对CUDA优化的代码快。）

Now that we know how to implement a matrix multiplication from scratch, we are ready to build our neural net—specifically its forward and backward passes—using just matrix multiplications.

现在我们知道了如果从零开始实现一个矩阵乘法，我们准备只使用矩阵乘法创建我们的神经网络，特别是它的正向和反向传递。

## The Forward and Backward Passes

## 正向和反向传递

As we saw in <chapter_mnist_basics>, to train a model, we will need to compute all the gradients of a given loss with respect to its parameters, which is known as the *backward pass*. The *forward pass* is where we compute the output of the model on a given input, based on the matrix products. As we define our first neural net, we will also delve into the problem of properly initializing the weights, which is crucial for making training start properly.

如我们在<第四章：mnist基础>中学的，训练一个模型，我们将需要计算给损失对于它的参数的所以梯度，它被称为*反向传递*。*正向传递*是基于矩阵成绩计算给定输出情况下模型的输出，如我们定义的第一个神经网络，我们也会深入研究正确的初始化权重的问题，对于使得训练正确的开始它是至关重要的。

### Defining and Initializing a Layer

### 定义和初始化层

We will take the example of a two-layer neural net first. As we've seen, one layer can be expressed as `y = x @ w + b`, with `x` our inputs, `y` our outputs, `w` the weights of the layer (which is of size number of inputs by number of neurons if we don't transpose like before), and `b` is the bias vector:

我们将首先以两层神经网络为例。正如你学过的，一层可以表示为 `y = x @ w + b`，`x`是我们的输入，`y`我们的输出，`w`层的权重（如果我们没有像以前那做转置，它的大小是输入的数量乘以神经元的数量），`b`是偏差向量：

实验代码：

```
def lin(x, w, b): return x @ w + b
```

We can stack the second layer on top of the first, but since mathematically the composition of two linear operations is another linear operation, this only makes sense if we put something nonlinear in the middle, called an activation function. As mentioned at the beginning of the chapter, in deep learning applications the activation function most commonly used is a ReLU, which returns the maximum of `x` and `0`.

我们可以在第一层的上面叠加第二层，但是因为两个线性运算的数学性构成是另外一个线性运算，如果我们在它们两者这间放某个非线性方程才有意思，这被称为激活函数。正如在本章的一开始提到的，深度学习应用的激活函数大多数通常使用的是ReLU，它返回`x`和 `0` 的最大值。

We won't actually train our model in this chapter, so we'll use random tensors for our inputs and targets. Let's say our inputs are 200 vectors of size 100, which we group into one batch, and our targets are 200 random floats:

在这个章我们将不会实际的训练我们的模型，所以我们会使用随机张量作为我们的输出和目标。假定我们的输入是200个大小为100的向量，我们归合它为一个批次，及我们的目标是200个随机浮点数。

实验代码：

```
x = torch.randn(200, 100)
y = torch.randn(200)
```

For our two-layer model we will need two weight matrices and two bias vectors. Let's say we have a hidden size of 50 and the output size is 1 (for one of our inputs, the corresponding output is one float in this toy example). We initialize the weights randomly and the bias at zero:

对于我们的双层模型我们将需要两个权重矩阵和两个偏差向量。假定我们有一个隐含层大小为50和输出大小为1（在这个实验例子中相应输出是一个浮点数，这个浮点数是我们输入中的一个）。我们初始化随机权重和偏差为零：

实验代码：

```
w1 = torch.randn(100,50)
b1 = torch.zeros(50)
w2 = torch.randn(50,1)
b2 = torch.zeros(1)
```

Then the result of our first layer is simply:

那么我们第一层的结果是：

实验代码：

```
l1 = lin(x, w1, b1)
l1.shape
```

实验输出：

```
torch.Size([200, 50])
```

Note that this formula works with our batch of inputs, and returns a batch of hidden state: `l1` is a matrix of size 200 (our batch size) by 50 (our hidden size).

There is a problem with the way our model was initialized, however. To understand it, we need to look at the mean and standard deviation (std) of `l1`:

注意这个算式处理的是我们的输入批次，并返回一个隐含状态批次：`l1`是一个大小为 200 （我们的批次大小）乘 50 （我们的隐含层大小）的矩阵。

然而我们模型初始的方法有个问题。理解它，我们需要看平均值和`l1`的标准差：

实验代码：

```
l1.mean(), l1.std()
```

实验输出：

```
(tensor(0.0019), tensor(10.1058))
```

The mean is close to zero, which is understandable since both our input and weight matrices have means close to zero. But the standard deviation, which represents how far away our activations go from the mean, went from 1 to 10. This is a really big problem because that's with just one layer. Modern neural nets can have hundred of layers, so if each of them multiplies the scale of our activations by 10, by the end of the last layer we won't have numbers representable by a computer.

这个平均值接近于零，这是可理解的因为我们的输入和权重矩阵的平均值都接近于零。但是标准差，它代表我们的激活距离平均值非常的远，从 1 到 10 。这是真的是一个大问题，因为这只是一层。现代神经网络可以有上百层，所以如果它们每层激活的规模乘以 10 ，最终在最后层我们也许不会有计算机能够表示的数据。

Indeed, if we make just 50 multiplications between `x` and random matrices of size 100×100, we'll have:

的确，如果我们只做 50 次 `x` 和大小为 100×100 的随机矩阵之间的乘法，我们会得到：

实验代码：

```
x = torch.randn(200, 100)
for i in range(50): x = x @ torch.randn(100,100)
x[0:5,0:5]
```

实验输出：

```
tensor([[nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan]])
```

The result is `nan`s everywhere. So maybe the scale of our matrix was too big, and we need to have smaller weights? But if we use too small weights, we will have the opposite problem—the scale of our activations will go from 1 to 0.1, and after 50 layers we'll be left with zeros everywhere:

所有的结果都是`nan`。所以可能我们矩阵的缩放太大了，我们需要更小的权重？但是如果我们使用小权重，我们会有相反的问题：我们激活的缩放会是从 1 到 0.1 ，且 50 层后我们会得到所有都是零的结果：

实验代码：

```
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.01)
x[0:5,0:5]
```

实验输出：

```
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
```

So we have to scale our weight matrices exactly right so that the standard deviation of our activations stays at 1. We can compute the exact value to use mathematically, as illustrated by Xavier Glorot and Yoshua Bengio in ["Understanding the Difficulty of Training Deep Feedforward Neural Networks"](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). The right scale for a given layer is $1/\sqrt{n_{in}}$, where $n_{in}$​represents the number of inputs.

所以我们必须完全正确的缩放我们的权重矩阵，这样我们激活的标准差就会在 1 上。由泽维尔·格洛洛特和约书亚·本吉奥在["理解训练深度前馈神经网络的困难"](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)中做了说明，我们能够使用数学方法计算出精确的值。对于给定的层正确的缩放是 $1/\sqrt{n_{in}}$​，这里的$n_{in}$​是输入的数量。

In our case, if we have 100 inputs, we should scale our weight matrices by 0.1:

在我们的例子中，如果我们有 100 个输入，我们应该缩放权重矩阵 0.1 ：

实验代码：

```
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.1)
x[0:5,0:5]
```

实验输出：

```
tensor([[ 0.7554,  0.6167, -0.1757, -1.5662,  0.5644],
        [-0.1987,  0.6292,  0.3283, -1.1538,  0.5416],
        [ 0.6106,  0.2556, -0.0618, -0.9463,  0.4445],
        [ 0.4484,  0.7144,  0.1164, -0.8626,  0.4413],
        [ 0.3463,  0.5930,  0.3375, -0.9486,  0.5643]])
```

Finally some numbers that are neither zeros nor `nan`s! Notice how stable the scale of our activations is, even after those 50 fake layers:

最终一些数值既不是零也不是`nan`！注意我们激活的缩放是如此稳定，甚至是做了那些 50 个虚拟层之后：

实验代码：

```
x.std()
```

实验输出：

```
tensor(0.7042)
```

If you play a little bit with the value for scale you'll notice that even a slight variation from 0.1 will get you either to very small or very large numbers, so initializing the weights properly is extremely important.

如果你对缩放稍微调整一点点，甚至是 0.1 的小的变化，你将会取得要么很小要么很大的数值，所以正确的初始化权重是异常重要的。

Let's go back to our neural net. Since we messed a bit with our inputs, we need to redefine them:

让我们返回到我们的神经网。因为我们已经有点弄乱了输入值，我们需要重新定义他们：

实验代码：

```
x = torch.randn(200, 100)
y = torch.randn(200)
```

And for our weights, we'll use the right scale, which is known as *Xavier initialization* (or *Glorot initialization*):

对于我们的权重，我们将名为*Xavier初始化*（或*Glorot*初始化）做正确的缩放：

实验代码：

```
from math import sqrt
w1 = torch.randn(100,50) / sqrt(100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) / sqrt(50)
b2 = torch.zeros(1)
```

Now if we compute the result of the first layer, we can check that the mean and standard deviation are under control:

现在如果我们计算了第一层的结果，我们可以检查它的平均值和标准差是受到控制的：

实验代码：

```
l1 = lin(x, w1, b1)
l1.mean(),l1.std()
```

实验输出：

```
(tensor(-0.0050), tensor(1.0000))
```

Very good. Now we need to go through a ReLU, so let's define one. A ReLU removes the negatives and replaces them with zeros, which is another way of saying it clamps our tensor at zero:

非常好。现在我们需要学习ReLU，所以让我们定义一下。ReLU移除了负数并用零替换它们，它是另一种强制我们的张量为零的方法：

实验代码：

```
def relu(x): return x.clamp_min(0.)
```

We pass our activations through this:

我们传递激活通过这个函数：

实验代码：

```
l2 = relu(l1)
l2.mean(),l2.std()
```

实验输出：

```
(tensor(0.3961), tensor(0.5783))
```

And we're back to square one: the mean of our activations has gone to 0.4 (which is understandable since we removed the negatives) and the std went down to 0.58. So like before, after a few layers we will probably wind up with zeros:

我们返回初始点：我们激活的平均值已经到了 0.4（这是可理解的，因为我们移除和负数）且std降到了 0.58 。所以像以前一样，几个层后我们可能将以零结束：

实验代码：

```
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * 0.1))
x[0:5,0:5]
```

实验输出：

```
tensor([[0.0000e+00, 1.9689e-08, 4.2820e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.6701e-08, 4.3501e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.0976e-08, 3.0411e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.8457e-08, 4.9469e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.9949e-08, 4.1643e-08, 0.0000e+00, 0.0000e+00]])
```

This means our initialization wasn't right. Why? At the time Glorot and Bengio wrote their article, the popular activation in a neural net was the hyperbolic tangent (tanh, which is the one they used), and that initialization doesn't account for our ReLU. Fortunately, someone else has done the math for us and computed the right scale for us to use. In ["Delving Deep into Rectifiers: Surpassing Human-Level Performance"](https://arxiv.org/abs/1502.01852) (which we've seen before—it's the article that introduced the ResNet), Kaiming He et al. show that we should use the following scale instead: $\sqrt{2 / n_{in}}$, where $n_{in}$​ is the number of inputs of our model. Let's see what this gives us:

这意味着我们的初始不正确。为什么？在泽维尔·格洛洛特和约书亚·本吉奥编写他们的文章时，神经网络流行的激活是双曲正切（双曲正切tanh，是他们使用的一种），且初始化没有考虑ReLU。幸运的是，其他人已经为我们做了数学计算并为我们完成了正确的缩放以供我们使用。在何凯明等人的文章中[“深入研究矫正器：超越人类水平的表现”](https://arxiv.org/abs/1502.01852)（在之前我们已经学习过，它是引入ResNet的文章），提供了我们应该使用这样的缩放作为替代： $\sqrt{2 / n_{in}}$，这里的  $n_{in}$ 是我们模型输入的数量。让我们看看这个方法给我们提供了什么：

实验代码：

```
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * sqrt(2/100)))
x[0:5,0:5]
```

实验输出：

```
tensor([[0.2871, 0.0000, 0.0000, 0.0000, 0.0026],
        [0.4546, 0.0000, 0.0000, 0.0000, 0.0015],
        [0.6178, 0.0000, 0.0000, 0.0180, 0.0079],
        [0.3333, 0.0000, 0.0000, 0.0545, 0.0000],
        [0.1940, 0.0000, 0.0000, 0.0000, 0.0096]])
```

That's better: our numbers aren't all zeroed this time. So let's go back to the definition of our neural net and use this initialization (which is named *Kaiming initialization* or *He initialization*):

这个结果好太多了：这次我们的数值并不都是零。所以返回我们神经网络的定义并使用这个初始化方法（它被命名为*凯明初始化*或*何初始化*）：

实验代码：

```
x = torch.randn(200, 100)
y = torch.randn(200)
```

实验代码：

```
w1 = torch.randn(100,50) * sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) * sqrt(2 / 50)
b2 = torch.zeros(1)
```

Let's look at the scale of our activations after going through the first linear layer and ReLU:

让我们一下通过第一个线性层和ReLU后激活的缩放:

实验代码：

```
l1 = lin(x, w1, b1)
l2 = relu(l1)
l2.mean(), l2.std()
```

实验输出：

```
(tensor(0.5661), tensor(0.8339))
```

Much better! Now that our weights are properly initialized, we can define our whole model:

好太多了！现在我们的权重正确的初始化了，我们能够定义整个模型了：

实验代码：

```
def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3
```

This is the forward pass. Now all that's left to do is to compare our output to the labels we have (random numbers, in this example) with a loss function. In this case, we will use the mean squared error. (It's a toy problem, and this is the easiest loss function to use for what is next, computing the gradients.)

这是一个正向传递。 现在剩下的工作是用损失函数把我们的输出与已有的标注做对比（在这个例子中是随机数值）。在这个例子中，我们会使用均方差误差。（这是一个实验问题，且这是用于下一步计算梯度的最容易的损失函数。）

The only subtlety is that our outputs and targets don't have exactly the same shape—after going though the model, we get an output like this:

唯一精妙之处是我们的输出和目标没有完全相同的形状，执行完整个模型后，我们获得了如下的输出：

实验代码：

```
out = model(x)
out.shape
```

实验输出：

```
torch.Size([200, 1])
```

To get rid of this trailing 1 dimension, we use the `squeeze` function:

舍弃这个数为 1 的维度，我们使用`squeeze`函数：

实验代码：

```
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
```

And now we are ready to compute our loss:

现在我们准备计算损失：

实验代码：

```
loss = mse(out, y)
```

That's all for the forward pass—let's now look at the gradients.

这就是正向传递的全部内容，现在我们看一下梯度。

### Gradients and the Backward Pass

### 梯度和反向传递

We've seen that PyTorch computes all the gradients we need with a magic call to `loss.backward`, but let's explore what's happening behind the scenes.



Now comes the part where we need to compute the gradients of the loss with respect to all the weights of our model, so all the floats in `w1`, `b1`, `w2`, and `b2`. For this, we will need a bit of math—specifically the *chain rule*. This is the rule of calculus that guides how we can compute the derivative of a composed function:

$$(g \circ f)'(x) = g'(f(x)) f'(x)$$

> j: I find this notation very hard to wrap my head around, so instead I like to think of it as: if `y = g(u)` and `u=f(x)`; then `dy/dx = dy/du * du/dx`. The two notations mean the same thing, so use whatever works for you.

Our loss is a big composition of different functions: mean squared error (which is in turn the composition of a mean and a power of two), the second linear layer, a ReLU and the first linear layer. For instance, if we want the gradients of the loss with respect to `b2` and our loss is defined by:

```
loss = mse(out,y) = mse(lin(l2, w2, b2), y)
```

The chain rule tells us that we have:

$$\frac{\text{d} loss}{\text{d} b_{2}} = \frac{\text{d} loss}{\text{d} out} \times \frac{\text{d} out}{\text{d} b_{2}} = \frac{\text{d}}{\text{d} out} mse(out, y) \times \frac{\text{d}}{\text{d} b_{2}} lin(l_{2}, w_{2}, b_{2})$$

To compute the gradients of the loss with respect to 𝑏2b2, we first need the gradients of the loss with respect to our output 𝑜𝑢𝑡out. It's the same if we want the gradients of the loss with respect to 𝑤2w2. Then, to get the gradients of the loss with respect to 𝑏1b1 or 𝑤1w1, we will need the gradients of the loss with respect to 𝑙1l1, which in turn requires the gradients of the loss with respect to 𝑙2l2, which will need the gradients of the loss with respect to 𝑜𝑢𝑡out.

So to compute all the gradients we need for the update, we need to begin from the output of the model and work our way *backward*, one layer after the other—which is why this step is known as *backpropagation*. We can automate it by having each function we implemented (`relu`, `mse`, `lin`) provide its backward step: that is, how to derive the gradients of the loss with respect to the input(s) from the gradients of the loss with respect to the output.

Here we populate those gradients in an attribute of each tensor, a bit like PyTorch does with `.grad`.

The first are the gradients of the loss with respect to the output of our model (which is the input of the loss function). We undo the `squeeze` we did in `mse`, then we use the formula that gives us the derivative of 𝑥2x2: 2𝑥2x. The derivative of the mean is just $1/n$ where $n$ is the number of elements in our input:

In [ ]:

```
def mse_grad(inp, targ): 
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
```

For the gradients of the ReLU and our linear layer, we use the gradients of the loss with respect to the output (in `out.g`) and apply the chain rule to compute the gradients of the loss with respect to the input (in `inp.g`). The chain rule tells us that `inp.g = relu'(inp) * out.g`. The derivative of `relu` is either 0 (when inputs are negative) or 1 (when inputs are positive), so this gives us:

In [ ]:

```
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g
```

The scheme is the same to compute the gradients of the loss with respect to the inputs, weights, and bias in the linear layer:

In [ ]:

```
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)
```

We won't linger on the mathematical formulas that define them since they're not important for our purposes, but do check out Khan Academy's excellent calculus lessons if you're interested in this topic.