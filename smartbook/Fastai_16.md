# The Training Process

# 训练过程

You now know how to create state-of-the-art architectures for computer vision, natural language processing, tabular analysis, and collaborative filtering, and you know how to train them quickly. So we're done, right? Not quite yet. We still have to explore a little bit more the training process.

现在你知道了如何为计算机视觉、自然语言处理、表格分析和协同过滤创建先进的架构了，并知道如何快速的训练他们。所以我们完成了学习，是这样的吗？还不完全是。我们必须继续再多研究一点训练过程。

We explained in <chapter_mnist_basics> the basis of stochastic gradient descent: pass a mini-batch to the model, compare it to our target with the loss function, then compute the gradients of this loss function with regard to each weight before updating the weights with the formula:

我们在<第四章：mnist基础>中讲解了随机梯度下降的基础知识：传递一个最小批次给模型，用损失函数与我们的目标做对比，然后计算关于每个权重的损失函数的梯度，然后用公式更新权重：

```python
new_weight = weight - lr * weight.grad
```

We implemented this from scratch in a training loop, and also saw that PyTorch provides a simple `nn.SGD` class that does this calculation for each parameter for us. In this chapter we will build some faster optimizers, using a flexible foundation. But that's not all we might want to change in the training process. For any tweak of the training loop, we will need a way to add some code to the basis of SGD. The fastai library has a system of callbacks to do this, and we will teach you all about it.

在一个训练循环中我们从零开始实现了这个操作，且也看到PyTorch提供了一个方便的`nn.SGD`类，它为我们对每个参数做这个计算。在本章我们会使用灵活的基础知识构建一些更快的优化器。但是这并不是我们可能希望在训练循环中改变的全部。训练循环的任何调整，我们需要一个方法增加某些代码到SGD的基础代码上。fastai库有一个回调系统来做这个事情，我们会教你关于它的一切知识。

Let's start with standard SGD to get a baseline, then we will introduce the most commonly used optimizers.

让我们从标准的SGD开始来获得一个基线，然后我们将会介绍最常用的优化器。

## Establishing a Baseline

## 评估一个基线

First, we'll create a baseline, using plain SGD, and compare it to fastai's default optimizer. We'll start by grabbing Imagenette with the same `get_data` we used in <chapter_resnet>:

首先，我们将会使用普通的SGD创建一个基线，并与fastai默认优化器做对比。我们会使用在<第十四章：残差网络>中同样使用的`get_data`函数来抓取Imagenette开始：

实验代码：

```
#hide_input
def get_data(url, presize, resize):
    path = untar_data(url)
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, 
        splitter=GrandparentSplitter(valid_name='val'),
        get_y=parent_label, item_tfms=Resize(presize),
        batch_tfms=[*aug_transforms(min_scale=0.5, size=resize),
                    Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(path, bs=128)
```

```
dls = get_data(URLs.IMAGENETTE_160, 160, 128)
```

We'll create a ResNet-34 without pretraining, and pass along any arguments received:

我们将不用预训练创建一个ResNet-34，并传递接收到的任意参数：

实验代码：

```
def get_learner(**kwargs):
    return cnn_learner(dls, resnet34, pretrained=False,
                    metrics=accuracy, **kwargs).to_fp16()
```

Here's the default fastai optimizer, with the usual 3e-3 learning rate:

下面是fastai默认优化器，使用了常用的3e-3学习率：

实验代码：

```
learn = get_learner()
learn.fit_one_cycle(3, 0.003)
```

实验输出:

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   2.571932 |   2.685040 | 0.322548 | 00:11 |
|     1 |   1.904674 |   1.852589 | 0.437452 | 00:11 |
|     2 |   1.586909 |   1.374908 | 0.594904 | 00:11 |

Now let's try plain SGD. We can pass `opt_func` (optimization function) to `cnn_learner` to get fastai to use any optimizer:

现在我们用普通的SGD尝试一下。我们可以传递`opt_func`（优化函数）给`cnn_learner`，让fastai使用任意优化器：

实验代码:

```
learn = get_learner(opt_func=SGD)
```

The first thing to look at is `lr_find`:

首要查看的内容是`lr_find`：

实验代码:

```
learn.lr_find()
```

实验输出:

```
(0.017378008365631102, 3.019951861915615e-07)
```

 ![img](data:image/png;base64,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)

It looks like we'll need to use a higher learning rate than we normally use:

它看起来好像我们需要使用一个比常用的学习率要更高的学习率：

实验代码:

```
learn.fit_one_cycle(3, 0.03, moms=(0,0,0))
```

实验输出:

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   2.969412 |   2.214596 | 0.242038 | 00:09 |
|     1 |   2.442730 |   1.845950 | 0.362548 | 00:09 |
|     2 |   2.157159 |   1.741143 | 0.408917 | 00:09 |

Because accelerating SGD with momentum is such a good idea, fastai does this by default in `fit_one_cycle`, so we turn it off with `moms=(0,0,0)`. (We'll be discussing momentum shortly.)

因为用动量加速SGD是一个绝佳的想法，fastai在`fit_one_cycle`中以默认的方式做这个事情，所以我们用 `moms=(0,0,0)`来关闭它。（我们将会简短的讨论动量。）

Clearly, plain SGD isn't training as fast as we'd like. So let's learn some tricks to get accelerated training!

很明显，普通的SGD的训练不如我们想要的快。所以让我们学习一些加速训练的技巧！

## A Generic Optimizer

## 通用优化器

To build up our accelerated SGD tricks, we'll need to start with a nice flexible optimizer foundation. No library prior to fastai provided such a foundation, but during fastai's development we realized that all the optimizer improvements we'd seen in the academic literature could be handled using *optimizer callbacks*. These are small pieces of code that we can compose, mix and match in an optimizer to build the optimizer `step`. They are called by fastai's lightweight `Optimizer` class. These are the definitions in `Optimizer` of the two key methods that we've been using in this book:

我们需要从一个非常好的灵活的基础库开始，来建立我们的加速SGD技巧。之前没有库提供了像fastai这样的基础库，

```python
def zero_grad(self):
    for p,*_ in self.all_params():
        p.grad.detach_()
        p.grad.zero_()

def step(self):
    for p,pg,state,hyper in self.all_params():
        for cb in self.cbs:
            state = _update(state, cb(p, **{**state, **hyper}))
        self.state[p] = state
```

As we saw when training an MNIST model from scratch, `zero_grad` just loops through the parameters of the model and sets the gradients to zero. It also calls `detach_`, which removes any history of gradient computation, since it won't be needed after `zero_grad`.

The more interesting method is `step`, which loops through the callbacks (`cbs`) and calls them to update the parameters (the `_update` function just calls `state.update` if there's anything returned by `cb`). As you can see, `Optimizer` doesn't actually do any SGD steps itself. Let's see how we can add SGD to `Optimizer`.

Here's an optimizer callback that does a single SGD step, by multiplying `-lr` by the gradients and adding that to the parameter (when `Tensor.add_` in PyTorch is passed two parameters, they are multiplied together before the addition):

实验代码：

```
def sgd_cb(p, lr, **kwargs): p.data.add_(-lr, p.grad.data)
```

We can pass this to `Optimizer` using the `cbs` parameter; we'll need to use `partial` since `Learner` will call this function to create our optimizer later:

实验代码：

```
opt_func = partial(Optimizer, cbs=[sgd_cb])
```

Let's see if this trains:

实验代码：

```
learn = get_learner(opt_func=opt_func)
learn.fit(3, 0.03)
```

实验输出：

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   2.730918 |   2.009971 | 0.332739 | 00:09 |
|     1 |   2.204893 |   1.747202 | 0.441529 | 00:09 |
|     2 |   1.875621 |   1.684515 | 0.445350 | 00:09 |

It's working! So that's how we create SGD from scratch in fastai. Now let's see what "momentum" is.