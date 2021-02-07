# Data Munging with fastai's Mid-Level API

# 用fastai中级API做数据整理

We have seen what `Tokenizer` and `Numericalize` do to a collection of texts, and how they're used inside the data block API, which handles those transforms for us directly using the `TextBlock`. But what if we want to only apply one of those transforms, either to see intermediate results or because we have already tokenized texts? More generally, what can we do when the data block API is not flexible enough to accommodate our particular use case? For this, we need to use fastai's *mid-level API* for processing data. The data block API is built on top of that layer, so it will allow you to do everything the data block API does, and much much more.

我们已经学习了`Tokenizer`和`Numericalize`对于文本全集的作用，及他们如何使用内部的数据块API，其能够使用`TextBlock`直接为我们处理那些转换。但要么看中间结果，要么因为我们已经有了标记文本，如果我们只想应用那些转换中的一个会怎样？更一般的来说，当数据块API的灵活性不足以提供给镒特定使用案例时我们能做什么？为此，为了处理数据我们需要使用fastai的*中级API*。数据块API是建立该层的顶端，所以它允许你做数据块API所能做的一切事，且更多更多。

## Going Deeper into fastai's Layered API

## 深入fastai的层API

The fastai library is built on a *layered API*. In the very top layer there are *applications* that allow us to train a model in five lines of codes, as we saw in <chapter_intro>. In the case of creating `DataLoaders` for a text classifier, for instance, we used the line:

fastai库建立在*层API*之上。在最顶层有一些*应用*允许我们用五行代码来训练模型，我们在<章节：概述>中看过。例如，为文本分类器创建`DataLoaders`例子中，我们使用的代码行：

```
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
```

The factory method `TextDataLoaders.from_folder` is very convenient when your data is arranged the exact same way as the IMDb dataset, but in practice, that often won't be the case. The data block API offers more flexibility. As we saw in the last chapter, we can get the same result with:

当你的数据以IMDb数据集的方式做完全相同的组合时，工厂方法`TextDataLoaders.from_folder`是非常方便的。但实际上，情况往往并非如此。数据块API提供了更多的灵活性。在上一章节中我们所学的，我们用下面的代码能够获得相同的结果：

```
path = untar_data(URLs.IMDB)
dls = DataBlock(
    blocks=(TextBlock.from_folder(path),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path)
```

But it's sometimes not flexible enough. For debugging purposes, for instance, we might need to apply just parts of the transforms that come with this data block. Or we might want to create a `DataLoaders` for some application that isn't directly supported by fastai. In this section, we'll dig into the pieces that are used inside fastai to implement the data block API. Understanding these will enable you to leverage the power and flexibility of this mid-tier API.

但有时候它还不足够的灵活。例如，以调试为目的，我们可能需要应用的仅仅是随附这个数据块的部分转换。或者我们可能想为一些fastai没有直接支持的应用创建一个`DataLoaders`。在本部分，我们深入研究使用fastai内部来实现数据块API的部分。

> note: Mid-Level API: The mid-level API does not only contain functionality for creating `DataLoaders`. It also has the *callback* system, which allows us to customize the training loop any way we like, and the *general optimizer*. Both will be covered in <chapter_accel_sgd>.

> 注释：中级API：中级API不仅仅包含创建`DataLoaders`的功能。它也有回调系统，允许我们定义任何我们喜欢的训练循环和常用优化器。这两者会在<章节:加速随机梯度下降>所体现。

### Transforms

### 转换

When we studied tokenization and numericalization in the last chapter, we started by grabbing a bunch of texts:

当我们在上一章节学习了标记化和数值化时，我们从抓取一堆文本开始的：

```
files = get_text_files(path, folders = ['train', 'test'])
txts = L(o.open().read() for o in files[:2000])
```

We then showed how to tokenize them with a `Tokenizer`:

然后我们展示了如何用`Tokenizer`来标记它们：

```
tok = Tokenizer.from_folder(path)
tok.setup(txts)
toks = txts.map(tok)
toks[0]
```

Out: (#374) ['xxbos' , 'xxmaj' , 'well' , ',' , '"' , 'cube' , '"' , '(' , '1997' , ')'...]

and how to numericalize, including automatically creating the vocab for our corpus:

及如何数据化，包括为我们的语料库自动创建词汇表：

```
num = Numericalize()
num.setup(toks)
nums = toks.map(num)
nums[0][:10]
```

Out: tensor([   2,    8,   76,   10,   23, 3112,   23,   34, 3113,   33])

The classes also have a `decode` method. For instance, `Numericalize.decode` gives us back the string tokens:

这些类也有一个`decode`方法。例如，`Numericalize.decode`给我们返回了标记串：

```
nums_dec = num.decode(nums[0][:10]); nums_dec
```

Out: (#10) ['xxbos' , 'xxmaj' , 'well' , ',' , '"' , 'cube' , '"' , '(' , '1997' , ')']

and `Tokenizer.decode` turns this back into a single string (it may not, however, be exactly the same as the original string; this depends on whether the tokenizer is *reversible*, which the default word tokenizer is not at the time we're writing this book):

`Tokenizer.decode`转换这个返回为一个单一的字符串（虽然它可能不会完成与原始字符串相同。这依赖于标记器是否是*可逆的*，在我们编写这本书的时候它默认不是可逆的单词标记器）：

```
tok.decode(nums_dec)
```

Out: 'xxbos xxmaj well , " cube " ( 1997 )'

`decode` is used by fastai's `show_batch` and `show_results`, as well as some other inference methods, to convert predictions and mini-batches into a human-understandable representation.

`decode`被fastai的`show_batch`和`show_results`所使用，和一些其它推断方法，把预测和小批次转换为人类所有理解的描述。

For each of `tok` or `num` in the preceding example, we created an object, called the `setup` method (which trains the tokenizer if needed for `tok` and creates the vocab for `num`), applied it to our raw texts (by calling the object as a function), and then finally decoded the result back to an understandable representation. These steps are needed for most data preprocessing tasks, so fastai provides a class that encapsulates them. This is the `Transform` class. Both `Tokenize` and `Numericalize` are `Transform`s.

在之前的例子中我们为`tok`或`num`每个都创建了一个对象，称为`setup`方法（如果需要`tok`的话它会训练标记器和为`num`创建词汇表），应用它到我们的原生文本（通过以函数的方式调用对象），其后完成编码结果返回可理解的描述。这些步骤是大多数据处理任务所需要的，因此fastai提供了一个类封装它们。这就`Transform`类。`Tokenize`和`Numericalize`两者都是`Transform`。

In general, a `Transform` is an object that behaves like a function and has an optional `setup` method that will initialize some inner state (like the vocab inside `num`) and an optional `decode` that will reverse the function (this reversal may not be perfect, as we saw with `tok`).

通常来说，`Transform`是一个对象，其表现像一个函数和有一个可选的会初始化一些内部状态（像`num`内部的词汇表）的`setup`方法，和一个可选的会反转函数的`decode`（这个反转可能不是很完美，正如我们看到的`tok`）。

A good example of `decode` is found in the `Normalize` transform that we saw in <chapter_sizing_and_tta>: to be able to plot the images its `decode` method undoes the normalization (i.e., it multiplies by the standard deviation and adds back the mean). On the other hand, data augmentation transforms do not have a `decode` method, since we want to show the effects on images to make sure the data augmentation is working as we want.

在<章节：数据尺寸和测试数据增强>中我们看到在`Normalize`转换中能找到一个很好的`decode`例子：为了能够画出图像，它的`decode`方法取消了归一化（即，它乘以离差并加回平均值）。另一方面，数据增强转换没有`decode`方法，因为我们希望显示图像上的效果，来确保数据增加如我们所预期的在工作。

A special behavior of `Transform`s is that they always get applied over tuples. In general, our data is always a tuple `(input,target)` (sometimes with more than one input or more than one target). When applying a transform on an item like this, such as `Resize`, we don't want to resize the tuple as a whole; instead, we want to resize the input (if applicable) and the target (if applicable) separately. It's the same for batch transforms that do data augmentation: when the input is an image and the target is a segmentation mask, the transform needs to be applied (the same way) to the input and the target.

`Transform`的一个特定行为是，它们总是应用于元组。通常来说，我们数据总是一个元组`(input,target)`（有时候会有多个输入或多个目标）。当应用一个转换在数据项上，如要做`Resize`，我们不希望对整个元组调整尺寸大小，相反我们希望分开调整输入（如果可应用的话）和目标（如果可应用的话）尺寸大小。对于做数据增强的批量转换也是一样的：当输入是一个图像，且目标是一个分割蒙版时，转换需要应用（同样的方法）到输入和目标。

We can see this behavior if we pass a tuple of texts to `tok`:

如果我们传递文本元组给`tok`，我们能够看到这个行为：

```
tok((txts[0], txts[1]))
```

Out: ((#374) ['xxbos' , 'xxmaj' , 'well' , ',' , '"' , 'cube' , '"' , '(' , '1997' , ')'...],  (#207) ['xxbos' , 'xxmaj' , 'conrad' , 'xxmaj' , 'hall' , 'went' , 'out' , 'with' , 'a' , 'bang'...])