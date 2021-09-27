# A Neural Net from the Foundations

# 基础的神经网络

This chapter begins a journey where we will dig deep into the internals of the models we used in the previous chapters. We will be covering many of the same things we've seen before, but this time around we'll be looking much more closely at the implementation details, and much less closely at the practical issues of how and why things are as they are.

本章节会开始一段路程，在这里我们将使用之前章节提供的内容深入挖掘模型的内部。我们会解释之前我们已经学习过的很多相同内容，但这次的范围我们会更仔细的关注于细节的实现，同时更少贴近于为什么事情是那样的实际问题。

We will build everything from scratch, only using basic indexing into a tensor. We'll write a neural net from the ground up, then implement backpropagation manually, so we know exactly what's happening in PyTorch when we call `loss.backward`. We'll also see how to extend PyTorch with custom *autograd* functions that allow us to specify our own forward and backward computations.

我们将从零开始创建所有内容，只使用基础索引到张量中。我们将从基础开始编写一个神经网络，然后手动实现反向传播，所以当我们调用`loss.backward`时就能够准确的知道PyTorch里发生了什么。我们也会学习扩展PyTorch，用自定义*autograd*函数允许我们详细说明我们自己的前向和反向计算。

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

这个区别来自什么地方？PyTorch不是用Python编写的矩阵乘法，而是C++以使它很快。无论什么时候我们在张量上做计算，我们将需要*向量化*它们，所以我们可以采纳PyTorch的速度优势，通常使用两个技术：元素计算和广播。

### Elementwise Arithmetic

### 元素计算

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

用元素计算，我们能够移除三个嵌套循环的一个：我们能够在所有元素加总之前，乘以`a`的第`i`行和`b`的第`j`列的相应张量，它会提升速度，因为内部循环现在会通过PyTorch以 C 的速度执行。

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

只是移除了内部的`for`循环，我们就已经快了大约 ~700倍！这只是个开始，利用广播我们能够移除其它的循环并取得更重要的加速。

### Broadcasting

### 广播

As we discussed in <chapter_mnist_basics>, broadcasting is a term introduced by the [NumPy library](https://docs.scipy.org/doc/) that describes how tensors of different ranks are treated during arithmetic operations. For instance, it's obvious there is no way to add a 3×3 matrix with a 4×5 matrix, but what if we want to add one scalar (which can be represented as a 1×1 tensor) with a matrix? Or a vector of size 3 with a 3×4 matrix? In both cases, we can find a way to make sense of this operation.

我们在<第四章：mnist基础>中讨论过广播，它是由[NumPy库](https://docs.scipy.org/doc/)引入的，描述了在计算运算期间如何处理不同阶张量。例如，很明显把 3×3 矩阵和 4×5 矩阵进行相加，但是如果我们希望用一个矩阵与一个标量（可以表示为一个 1×1 张量）相加呢？或用 3×4 矩阵与大小为 3 的向量相加？在这两个案例中，我们能够寻找一个方法来搞清楚这个运算。

Broadcasting gives specific rules to codify when shapes are compatible when trying to do an elementwise operation, and how the tensor of the smaller shape is expanded to match the tensor of the bigger shape. It's essential to master those rules if you want to be able to write code that executes quickly. In this section, we'll expand our previous treatment of broadcasting to understand these rules.

当尝试做按元素运算在形状兼容时广播给出具体的规则，及小形状张量如何扩展以匹配大形状张量。如果你想能够编写快速执行的代码，熟悉那些规则是必须的。在本小节，我们把之前处理的广播展开来讲以理解这些规则。

#### Broadcasting with a scalar

#### 用标量广播

Broadcasting with a scalar is the easiest type of broadcasting. When we have a tensor `a` and a scalar, we just imagine a tensor of the same shape as `a` filled with that scalar and perform the operation:

用标题广播是最容易的广播类型。当我们有一个张量`a`和一个标题，我们只是想像一个与`a`相同形状的张量来填满标量并执行运算：

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

我们如何能够做这个对比？`0`被*广播*以有与`a`相关的维度。注意没有在内存中创建一个全是零的张量做完这个事情（那会非常没有效率）。

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

如果对于矩阵的每行都有不同的平均值呢？在这种情况下我们会需要广播向量到矩阵。

#### Broadcasting a vector to a matrix

#### 广播向量到矩阵

We can broadcast a vector to a matrix as follows:

我们能够用下面的代码广播向量到矩阵：

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

因为`m`是 3×3 大小，有两个方法来做广播。事实上，在最后一个维度上完成广播是来自广播规则的一个惯例，且用这个方法我们不需要安排张量做任何事情，如果我们用这个方法，我会会获得相同的结果：

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

事实上，这一方法只可能用`m × n`大小的矩阵来广播一个`n`大小的向量：

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

如果我们想在其它维度上做广播，我们必须改变我们向量的形状，使为成为一个 2×1 矩阵。用PyTorch的`unsqueeze`方法作为这个事情：

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

利用广播，如果我们需要以默认的方式添加维度，它们会在开始就添加。当我们进行广播之前，PyTorchd在后台就做了`c.unsqueeze(0)`：

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

利用这个方法，我们可以在我们的矩阵乘法函数中移除另外一个`for`循环。现在，不是用`a[i]`和`b[:,j]`相乘，相替代的我们可以使用广播让`a[i]`与整个矩阵`b`相乘，然后合计结果：

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

现在我们比第一次实现的代码快了3,700倍！在我们继续向后学习前，让我们更详细一点来讨论广播规则。

#### Broadcasting rules

#### 广播规则

When operating on two tensors, PyTorch compares their shapes elementwise. It starts with the *trailing dimensions* and works its way backward, adding 1 when it meets empty dimensions. Two dimensions are *compatible* when one of the following is true:

- They are equal.
- One of them is 1, in which case that dimension is broadcast to make it the same as the other.

当在两个张量上运算时，PyTorch会按元素对比它们的形状。它从*尾部维度*开始且它的处理方式是反向的，当它遇到空维度会加 1 。当满足下面条件之一时两个维度就是兼任的：

- 它们是相等的。
- 它们其中一个是 1 ，在这样的情况下维度被广播以使它与另一个相同。

Arrays do not need to have the same number of dimensions. For example, if you have a 256×256×3 array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with three values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

数组不需要有相同数量的维度。例如，如果我有一个RGB值的 256×256×3 数组，你想需要用不同的值来缩放图像中的每个颜色，你可以图像乘以有三个值的一维数组。根据广播规则排列这些数组的尾部坐标轴的大小，显示它们是兼容的：

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

在我们之前的例子中，我们有一个 3×3 矩阵和一个大小为 3 的向量，广播在行上完成了：

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

现在我们知道了如果从零开始实现一个矩阵乘法，我们准备只使用矩阵乘法创建我们的神经网络，特别是它的前向和反向传递。

## The Forward and Backward Passes

## 前向和反向传递

As we saw in <chapter_mnist_basics>, to train a model, we will need to compute all the gradients of a given loss with respect to its parameters, which is known as the *backward pass*. The *forward pass* is where we compute the output of the model on a given input, based on the matrix products. As we define our first neural net, we will also delve into the problem of properly initializing the weights, which is crucial for making training start properly.

如我们在<第四章：mnist基础>中学的，训练一个模型，我们将需要计算给损失对于它的参数的所以梯度，它被称为*反向传递*。*前向传递*是基于矩阵成绩计算给定输出情况下模型的输出，如我们定义的第一个神经网络，我们也会深入研究正确的初始化权重的问题，对于使得训练正确的开始它是至关重要的。

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

这是一个前向传递。 现在剩下的工作是用损失函数把我们的输出与已有的标注做对比（在这个例子中是随机数值）。在这个例子中，我们会使用均方差误差。（这是一个实验问题，且这是用于下一步计算梯度的最容易的损失函数。）

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

这就是前向传递的全部内容，现在我们看一下梯度。

### Gradients and the Backward Pass

### 梯度和反向传递

We've seen that PyTorch computes all the gradients we need with a magic call to `loss.backward`, but let's explore what's happening behind the scenes.

我们已经学习过要想PyTorch计算所有的梯度，我们需要一个神奇的调用`loss.backward`，现在让我们研究一下在这个功能后发生了什么。

Now comes the part where we need to compute the gradients of the loss with respect to all the weights of our model, so all the floats in `w1`, `b1`, `w2`, and `b2`. For this, we will need a bit of math—specifically the *chain rule*. This is the rule of calculus that guides how we can compute the derivative of a composed function:

现在学习一下需要计算关于我们模型所有权重的损失梯度部分，因此所有的内容都在 `w1`, `b1`, `w2`, 和`b2`中。为此，我们将需要一些计算知识，具体的*链式法则*。这是我们可以如何计算梯度合成函数的导数的微积分规则：

$$(g \circ f)'(x) = g'(f(x)) f'(x)$$

> j: I find this notation very hard to wrap my head around, so instead I like to think of it as: if `y = g(u)` and `u=f(x)`; then `dy/dx = dy/du * du/dx`. The two notations mean the same thing, so use whatever works for you.

$$(g \circ f)'(x) = g'(f(x)) f'(x)$$

> 杰：我发现这个表达式理解的难度很好，所以我喜欢这样理解：如果 `y = g(u)` 和 `u=f(x)`；那么 `dy/dx = dy/du * du/dx` 。这两种表达式是相同的意思，所以对你来说无论在什么情况下，用这种方式思考。

Our loss is a big composition of different functions: mean squared error (which is in turn the composition of a mean and a power of two), the second linear layer, a ReLU and the first linear layer. For instance, if we want the gradients of the loss with respect to `b2` and our loss is defined by:

我们的损失是一个不同的大组合函数：均方误差（其依次是平均数和二幂的组合），第二线性层，一个ReLU 及第一线性层。例如，如果我们想求关于`b2`的损失的梯度，那么我们损失的定义是：

```
loss = mse(out,y) = mse(lin(l2, w2, b2), y)
```

The chain rule tells us that we have:

$$\frac{\text{d} loss}{\text{d} b_{2}} = \frac{\text{d} loss}{\text{d} out} \times \frac{\text{d} out}{\text{d} b_{2}} = \frac{\text{d}}{\text{d} out} mse(out, y) \times \frac{\text{d}}{\text{d} b_{2}} lin(l_{2}, w_{2}, b_{2})$$

链式法则告诉我们会有：

$$\frac{\text{d} loss}{\text{d} b_{2}} = \frac{\text{d} loss}{\text{d} out} \times \frac{\text{d} out}{\text{d} b_{2}} = \frac{\text{d}}{\text{d} out} mse(out, y) \times \frac{\text{d}}{\text{d} b_{2}} lin(l_{2}, w_{2}, b_{2})$$

To compute the gradients of the loss with respect to $b_{2}$, we first need the gradients of the loss with respect to our output $out$. It's the same if we want the gradients of the loss with respect to $w_{2}$. Then, to get the gradients of the loss with respect to $b_{1}$ or $w_{1}$, we will need the gradients of the loss with respect to $l_{1}$, which in turn requires the gradients of the loss with respect to $l_{2}$, which will need the gradients of the loss with respect to $out$.

对计算关于 $b_{2}$ 的损失的梯度，我们首先需要计算关于我们的输出 $out$ 的损失梯度。如果我们希望计算关于 $w_{2}$ 的损失梯度也是同样的。然后，获取$b_{1}$ 或 $w_{1}$ 的损失的梯度，我们会需要关于 $l_{1}$ 的损失的梯度，依次 $l_{1}$ 需要  $l_{2}$ 的损失的梯度 ，而 $l_{2}$ 需要关于 $out$ 的损失的梯度。

So to compute all the gradients we need for the update, we need to begin from the output of the model and work our way *backward*, one layer after the other—which is why this step is known as *backpropagation*. We can automate it by having each function we implemented (`relu`, `mse`, `lin`) provide its backward step: that is, how to derive the gradients of the loss with respect to the input(s) from the gradients of the loss with respect to the output.

所以计算我们需要更新的所有梯度，我们需要一层接一层的从模型的输出开始并*逆向*处理，这就是为什么这一步被称为*反向广播*。我们可以通过实现的 (`relu`, `mse`, `lin`) 提供给它反向步骤的每个函数自动的实现它：即，如何驱动从关于输出的损失的梯度到输入的损失的梯度。

Here we populate those gradients in an attribute of each tensor, a bit like PyTorch does with `.grad`.

在这里我们在每个张量的属性中填充那些梯度，这有点像PyTorch 处理`.grad`。

The first are the gradients of the loss with respect to the output of our model (which is the input of the loss function). We undo the `squeeze` we did in `mse`, then we use the formula that gives us the derivative of $x^{2}$: $2x$. The derivative of the mean is just $1/n$ where $n$ is the number of elements in our input:

首先是关于我们模型输出的损失的梯度（它是损失函数的输入）。我们撤销我们在 `mse`中做的 `squeeze`，然后我们使用公式提供给我们的 $x^{2}$: $2x$ 的导数。平均值的导数正好是  $1/n$ ，这里的 $n$ 是在我们输入中元素的数量：

实验代码:

```
def mse_grad(inp, targ): 
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
```

For the gradients of the ReLU and our linear layer, we use the gradients of the loss with respect to the output (in `out.g`) and apply the chain rule to compute the gradients of the loss with respect to the input (in `inp.g`). The chain rule tells us that `inp.g = relu'(inp) * out.g`. The derivative of `relu` is either 0 (when inputs are negative) or 1 (when inputs are positive), so this gives us:

对于 ReLU 和我们线性层的梯度，我们使用关于输出（在 `out.g`中）的损失的梯度和应用链式法则来计算关于输入（在`inp.g`中）损失的梯度。链式法则告诉了我们 `inp.g = relu'(inp) * out.g`。`relu` 的导数是 0（当输入是负数时）或 1 （当输入是正数时），所以，这提供给我们：

实验代码:

```
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g
```

The scheme is the same to compute the gradients of the loss with respect to the inputs, weights, and bias in the linear layer:

在线性层中计算关于输入、权重和偏差的损失的梯度是同样的方案：

实验代码:

```
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)
```

We won't linger on the mathematical formulas that define them since they're not important for our purposes, but do check out Khan Academy's excellent calculus lessons if you're interested in this topic.

我们逗留在数学公式的定义上，因为他们对于我们的学习目标是不重要的，但是如果你对这个话题有兴趣，可以访问可汗学院优秀的微积分课程。

### Sidebar: SymPy

### 侧边栏：SymPy

SymPy is a library for symbolic computation that is extremely useful library when working with calculus. Per the [documentation](https://docs.sympy.org/latest/tutorial/intro.html):

> : Symbolic computation deals with the computation of mathematical objects symbolically. This means that the mathematical objects are represented exactly, not approximately, and mathematical expressions with unevaluated variables are left in symbolic form.

SymPy 是一个符号计算库，当用于微积分时它是极其有用的库，根据[文档](https://docs.sympy.org/latest/tutorial/intro.html):

> ：符号计算处理数学目标符号的计算。这表示数据目标被准确的描述，不是近似的，且有未评估变量的数据表达是在左侧以符号的形式表示。

To do symbolic computation, we first define a *symbol*, and then do a computation, like so:

对于做符号计算，我们首先定义一个 *符号*，然后做计算，如下：

实验代码:

```
from sympy import symbols,diff
sx,sy = symbols('sx sy')
diff(sx**2, sx)
```

实验输出: $2sx$

Here, SymPy has taken the derivative of `x**2` for us! It can take the derivative of complicated compound expressions, simplify and factor equations, and much more. There's really not much reason for anyone to do calculus manually nowadays—for calculating gradients, PyTorch does it for us, and for showing the equations, SymPy does it for us!

在这里，SymPy 为我们求了 `x**2` 的导数！它能够求复杂构成表达式、简化的、因子议程等的导数。现如今每个人真的没有太多理由手动的来计算微积分。对于计算梯度，PyTorch 为我们进行计算，SymPy 为我们做展示方程！

### End sidebar

### 侧边栏结束

Once we have have defined those functions, we can use them to write the backward pass. Since each gradient is automatically populated in the right tensor, we don't need to store the results of those `_grad` functions anywhere—we just need to execute them in the reverse order of the forward pass, to make sure that in each function `out.g` exists:

一旦我们定义了那些函数，我们就能够使用它们来编写反向传递。因为每个梯度是自动的填充在正确的张量中，无论何处我们都不需要存贮那些`_grad`函数的结果，我们只需要以前向传递的相反顺序执行他们，确保在每个函数中 `out.g`是存在的：

实验代码:

```
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

And now we can access the gradients of our model parameters in `w1.g`, `b1.g`, `w2.g`, and `b2.g`.

现在我们能够访问 `w1.g`，`b1.g`，`w2.g` 和 `b2.g`中我们模型参数的梯度了。

We have successfully defined our model—now let's make it a bit more like a PyTorch module.

我们成功定义了我们的模型，现在让我们让它更像一点PyTorch 模型吧。

### Refactoring the Model

### 重构模型

The three functions we used have two associated functions: a forward pass and a backward pass. Instead of writing them separately, we can create a class to wrap them together. That class can also store the inputs and outputs for the backward pass. This way, we will just have to call `backward`:

我们使用的三个函数有两个是有关联的：一个前向传递，一个是反向传递。相对于分别编写他们，我们能够创建一个类把他们合并在一起。这个类也能够存贮反向传递的输入和输出。这样，我们只需要调用 `backward` 了：

实验代码:

```
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```

`__call__` is a magic name in Python that will make our class callable. This is what will be executed when we type `y = Relu()(x)`. We can do the same for our linear layer and the MSE loss:

`__call__`在 Python 中是一个神奇的命名，这会使得我们的类可调用。当我们键入 `y = Relu()(x)` 时它会执行此内容。我们能够对我们的线性层和 MSE 损失做同样的操作：

实验代码:

```
class Lin():
    def __init__(self, w, b): self.w,self.b = w,b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out
    
    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```

实验代码:

```
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        x = (self.inp.squeeze()-self.targ).unsqueeze(-1)
        self.inp.g = 2.*x/self.targ.shape[0]
```

Then we can put everything in a model that we initiate with our tensors `w1`, `b1`, `w2`, `b2`:

我们能够在模型中放置任何内容，用我们张量 `w1`， `b1`， `w2`， `b2`我们做初始化：

实验代码:

```
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```

What is really nice about this refactoring and registering things as layers of our model is that the forward and backward passes are now really easy to write. If we want to instantiate our model, we just need to write:

这个重构的真正的好处是，注册内容作为我们模型的层是现在让前向和反向传递真正的容易编写。如果我们希望实例化我们的模型，我们只需要这样写：

实验代码:

```
model = Model(w1, b1, w2, b2)
```

The forward pass can then be executed with:

然后前向传递能够这样被执行：

实验代码:

```
loss = model(x, y)
```

And the backward pass with:

和反向传递：

实验代码:

```
model.backward()
```

### Going to PyTorch

### 用 PyTorch 实现

The `Lin`, `Mse` and `Relu` classes we wrote have a lot in common, so we could make them all inherit from the same base class:

我们编写的`Lin`, `Mse` 和 `Relu` 类有一些共同点，所以我们能够使得他们从相同的基类继承：

实验代码:

```
class LayerFunction():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self):  raise Exception('not implemented')
    def bwd(self):      raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```

Then we just need to implement `forward` and `bwd` in each of our subclasses:

然后我们只需要在我们的每个子类中实现 `forward` 和 `bwd`：

实验代码:

```
class Relu(LayerFunction):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```

实验代码:

```
class Lin(LayerFunction):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = out.g.sum(0)
```

实验代码:

```
class Mse(LayerFunction):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ): 
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

The rest of our model can be the same as before. This is getting closer and closer to what PyTorch does. Each basic function we need to differentiate is written as a `torch.autograd.Function` object that has a `forward` and a `backward` method. PyTorch will then keep trace of any computation we do to be able to properly run the backward pass, unless we set the `requires_grad` attribute of our tensors to `False`.

剩下的我们模型与之前是相同的。这越来越接近 PyTorch 做的内容。我们需要差异化每个基础函，编写为 `torch.autograd.Function` 目标，它有 一个`forward` 和 一个 `backward` 方法。其后PyTorch 会对我们做的任何计算保持记录，以能够合适的运行反向传递，除非我们设置我们张量的 `requires_grad` 属性为 `False`。

Writing one of these is (almost) as easy as writing our original classes. The difference is that we choose what to save and what to put in a context variable (so that we make sure we don't save anything we don't need), and we return the gradients in the `backward` pass. It's very rare to have to write your own `Function` but if you ever need something exotic or want to mess with the gradients of a regular function, here is how to write one:

编写这样的函数与我们所编写的最初的类是（差不多）是一样的。差异是我们选择保存什么和放置什么环境变量（这样我们确保我们保存需要的内容），并且我们返回在 `backward` 传递中的梯度。这是非常罕见的必须编写你自己的`函数`，但是如果你需要一些奇异的内容或想去把一个常规函数的梯度弄乱，下面展示了如何编写一个函数：

实验代码:

```
from torch.autograd import Function

class MyRelu(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.)
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i>0).float()
```

The structure used to build a more complex model that takes advantage of those `Function`s is a `torch.nn.Module`. This is the base structure for all models, and all the neural nets you have seen up until now inherited from that class. It mostly helps to register all the trainable parameters, which as we've seen can be used in the training loop.

利用那些函数构建一个更复杂模型的架构是`torch.nn.Module`。这是适用所有模型的基础架构，我们迄今为止看到的所有的神经网络都来自这个类。它主要用于注册所有的可训练参数，如我们所见能够用于训练循环。

To implement an `nn.Module` you just need to:

- Make sure the superclass `__init__` is called first when you initialize it.
- Define any parameters of the model as attributes with `nn.Parameter`.
- Define a `forward` function that returns the output of your model.

实现一个`nn.Module`你只需要：

- 当我们初始化的时候，确保子类`__init__` 首先被调用。
- 用 `nn.Parameter` 定义模型的任意参数作为属性。
- 定义一个 `forward` 函数，返回你的模型的输出。

As an example, here is the linear layer from scratch:

作为一个实例，下面是从头定义的线性层：

实验代码:

```
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
    
    def forward(self, x): return x @ self.weight.t() + self.bias
```

As you see, this class automatically keeps track of what parameters have been defined:

如你所见，这个类自动的记录已经被定义的参数：

实验代码:

```
lin = LinearLayer(10,2)
p1,p2 = lin.parameters()
p1.shape,p2.shape
```

实验输出:

```
(torch.Size([2, 10]), torch.Size([2]))
```

It is thanks to this feature of `nn.Module` that we can just say `opt.step()` and have an optimizer loop through the parameters and update each one.

这要感谢于 `nn.Module` 的这个特性，我们可以只声明 `opt.step()`，就有了一个贯穿参数的优化器循环，并对每一个参数进行更新。

Note that in PyTorch, the weights are stored as an `n_out x n_in` matrix, which is why we have the transpose in the forward pass.

在 PyTorch 中要注意，权重被存贮为一个 `n_out x n_in` 矩阵，这就是为什么在反向传递中我们有转置：

By using the linear layer from PyTorch (which uses the Kaiming initialization as well), the model we have been building up during this chapter can be written like this:

通过使用来自 PyTorch 中的线性层（它同样使用了  Kaiming 初始化），在本章期间我们创建起来的模型，能够编写以下的样子：

实验代码:

```
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse
        
    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```

fastai provides its own variant of `Module` that is identical to `nn.Module`, but doesn't require you to call `super().__init__()` (it does that for you automatically):

fastai 提供了它自己的 `Module` 变种，它与 `nn.Module` 是相同的，但是不需要你调用 `super().__init__()` （它为你自动的做这个事情）：

实验代码:

```
class Model(Module):
    def __init__(self, n_in, nh, n_out):
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse
        
    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```

In the last chapter, we will start from such a model and see how to build a training loop from scratch and refactor it to what we've been using in previous chapters.

在最后一章，我们会从这样的一个模型开始，并学习如何从头开始创建一个训练循环，并把它重构为我们之前章节所使用的内容。

## Conclusion

## 总结

In this chapter we explored the foundations of deep learning, beginning with matrix multiplication and moving on to implementing the forward and backward passes of a neural net from scratch. We then refactored our code to show how PyTorch works beneath the hood.

在本章节我们探索了深度学习的基础，从矩阵乘法开始，一直到从头开始实现一个神经网络的反向和前向传递。然后我们重构我们的代码来展示PyTorch 在后台如何处理的。

Here are a few things to remember:

- A neural net is basically a bunch of matrix multiplications with nonlinearities in between.
- Python is slow, so to write fast code we have to vectorize it and take advantage of techniques such as elementwise arithmetic and broadcasting.
- Two tensors are broadcastable if the dimensions starting from the end and going backward match (if they are the same, or one of them is 1). To make tensors broadcastable, we may need to add dimensions of size 1 with `unsqueeze` or a `None` index.
- Properly initializing a neural net is crucial to get training started. Kaiming initialization should be used when we have ReLU nonlinearities.
- The backward pass is the chain rule applied multiple times, computing the gradients from the output of our model and going back, one layer at a time.
- When subclassing `nn.Module` (if not using fastai's `Module`) we have to call the superclass `__init__` method in our `__init__` method and we have to define a `forward` function that takes an input and returns the desired result.

这里有一些内容需要记住：

- 神经网络基本上是一堆矩阵乘法在中间带有非线性。
- Python 运行慢，所以要写出快的代码，我们必须矢量化它并利用如元素算术和广播方法。
- 两个张量如果从末尾开始并反向匹配（如果他们是相同，或他们其中一个是  1），可广播两个张量。
- 合适的初始化一个神经网络对于训练的开始是至关重要的。当我们有 ReLU 非线性时，应该使用Kaiming 初始化。
- 反向传递是多次应用链式法则，从我们模型的输出开始计算梯度并向回走，一次一层。
- 当子类 `nn.Module`（如果不使用 fastai 的 Module）时，在我们的`__init__`方法中我们必须调用子类的`__init__`方法，并我们必须定义一个需要一个输入和返回期望结果的 `forward` 函数。

## Questionnaire

## 练习题

1. Write the Python code to implement a single neuron.
2. 编写 Python 代码来来实现一个单神经元。
3. Write the Python code to implement ReLU.
4. 编写 Python 代码来实现 ReLU。
5. Write the Python code for a dense layer in terms of matrix multiplication.
6. 依据矩阵乘法为稠密层编写 Python 代码。
7. Write the Python code for a dense layer in plain Python (that is, with list comprehensions and functionality built into Python).
8. 用纯 Python 为稠密层编写 Python 代码（即，用列表解释和内置的 Python 功能）。
9. What is the "hidden size" of a layer?
10. 什么是层的“隐含大小”？
11. What does the `t` method do in PyTorch?
12. 在 PyTorch 中 `t` 方法做了什么？
13. Why is matrix multiplication written in plain Python very slow?
14. 为什么用纯 Python编写的矩阵乘法非常慢？
15. In `matmul`, why is `ac==br`?
16. 在 `matmul` 中，为什么 `ac==br` ？
17. In Jupyter Notebook, how do you measure the time taken for a single cell to execute?
18. 在 Jupyter Notebook 中，我们如何测量单一单元格执行所花费的时间？
19. What is "elementwise arithmetic"?
20. 什么是“元素计算”？
21. Write the PyTorch code to test whether every element of `a` is greater than the corresponding element of `b`.
22. 编写 PyTorch 代码来测试 `a` 的每个元素是否比 `b` 相应的元素更大。
23. What is a rank-0 tensor? How do you convert it to a plain Python data type?
24. 什么是 0 阶 张量？你如何转换它为纯 Python 数据类型？
25. What does this return, and why? `tensor([1,2]) + tensor([1])`
26. 这个返回做了什么，为什么？`tensor([1,2]) + tensor([1])`
27. What does this return, and why? `tensor([1,2]) + tensor([1,2,3])`
28. 这个返回做了什么，为什么？`tensor([1,2]) + tensor([1,2,3])`
29. How does elementwise arithmetic help us speed up `matmul`?
30. 元素计算如何帮助我们加速了 `matmul` ？
31. What are the broadcasting rules?
32. 广播的规则是什么？
33. What is `expand_as`? Show an example of how it can be used to match the results of broadcasting.
34. 什么是 `expand_as`？举一个例子，它如何能够用来匹配广播的结果。
35. How does `unsqueeze` help us to solve certain broadcasting problems?
36. `unsqueeze`帮助我们如何解决某些广播问题？
37. How can we use indexing to do the same operation as `unsqueeze`?
38. 我们如何使用索引来做 `unsqueeze` 同样的操作？
39. How do we show the actual contents of the memory used for a tensor?
40. 我们如何展示用于一个张量的内存的准确内容？
41. When adding a vector of size 3 to a matrix of size 3×3, are the elements of the vector added to each row or each column of the matrix? (Be sure to check your answer by running this code in a notebook.)
42. 当把一个大小为 3 的向量加到一个大小为 3×3 的矩阵时，微量的元素是加到矩阵的每行，还是每列？（确保在 notebook 中通过运行这个代码来检查你的答案）
43. Do broadcasting and `expand_as` result in increased memory use? Why or why not?
44. 做广播和 `expand_as` 会导致内存使用的增加？为什么会或为什么不会？
45. Implement `matmul` using Einstein summation.
46. 使用爱因斯坦求和约定实现 `matmul` 。
47. What does a repeated index letter represent on the left-hand side of einsum?
48. 在爱因斯坦求和左侧的重复索引字母代表什么？
49. What are the three rules of Einstein summation notation? Why?
50. 爱因斯坦求和记法的三个规则是什么？为什么？
51. What are the forward pass and backward pass of a neural network?
52. 什么是神经网络的前向传递和反向传递？
53. Why do we need to store some of the activations calculated for intermediate layers in the forward pass?
54. 在前向传递中我们为什么需要存贮一些中间层激活计算？
55. What is the downside of having activations with a standard deviation too far away from 1?
56. 标准偏差距离 1 太远的的激活的缺点是什么？
57. How can weight initialization help avoid this problem?
58. 权重初始如何能够帮助避免这个问题？
59. What is the formula to initialize weights such that we get a standard deviation of 1 for a plain linear layer, and for a linear layer followed by ReLU?
60. 初始权重的公式是什么，使得我们获得普通线性层 1 的标准差，和 ReLU后 的线性层的标准差？
61. Why do we sometimes have to use the `squeeze` method in loss functions?
62. 在损失函数中为什么我们有时必须使用 `squeeze` 方法？
63. What does the argument to the `squeeze` method do? Why might it be important to include this argument, even though PyTorch does not require it?
64. 对于`squeeze`方法的论点做了什么？为什么包含这个论点是可能是重要的，即使 PyTorch 并不需要它？
65. What is the "chain rule"? Show the equation in either of the two forms presented in this chapter.
66. 什么是“链式法则”？展示本章节中两个展现形式中的任何一个。
67. Show how to calculate the gradients of `mse(lin(l2, w2, b2), y)` using the chain rule.
68. 展示使用链式法则如何计算 `mse(lin(l2, w2, b2), y)` 的梯度。
69. What is the gradient of ReLU? Show it in math or code. (You shouldn't need to commit this to memory—try to figure it using your knowledge of the shape of the function.)
70. ReLU 的梯度是什么？用数学或代码来展示它。（你应该不需要耗费时间来记忆，利用你的函数形状的知识尝试想出它）
71. In what order do we need to call the `*_grad` functions in the backward pass? Why?
72. 在反射传递中我们需要以什么样的顺序调用 `*_grad` 函数？为什么？
73. What is `__call__`?
74. `__call__` 是什么？
75. What methods must we implement when writing a `torch.autograd.Function`?
76. 当编写一个 `torch.autograd.Function` 我们必须实现什么方法？
77. Write `nn.Linear` from scratch, and test it works.
78. 从头开始编写 `nn.Linear`，并测试它的工作原理。
79. What is the difference between `nn.Module` and fastai's `Module`?
80. `nn.Module` 和 fastai 的 `Module` 之间的差异是什么？

### Further Research

### 深入研究

1. Implement ReLU as a `torch.autograd.Function` and train a model with it.
2. If you are mathematically inclined, find out what the gradients of a linear layer are in mathematical notation. Map that to the implementation we saw in this chapter.
3. Learn about the `unfold` method in PyTorch, and use it along with matrix multiplication to implement your own 2D convolution function. Then train a CNN that uses it.
4. Implement everything in this chapter using NumPy instead of PyTorch.

1. 实现 ReLU 作为一个`torch.autograd.Function` 并用它训练一个模型。
2. 如果你是数学方向的，用数学记法找出一个线性层的梯度是什么。映射到我们在本章节中看到的所实现的内容上。
3. 学习 PyTorch 中 `unfold` 方法，且用它连同矩阵乘法一起实现你自己的 2D 卷积函数。然后用它训练一个 CNN。
4. 使用 NumPy 而不是 PyTorch 来实现本章节中的所有内容。