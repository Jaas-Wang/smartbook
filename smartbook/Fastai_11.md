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