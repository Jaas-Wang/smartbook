# Other Computer Vision Problems

# 其它计算机视觉问题

In the previous chapter you learned some important practical techniques for training models in practice. Considerations like selecting learning rates and the number of epochs are very important to getting good results.

在上一章节我们学习了一些在实践中训练模型的一些重要特定技术。考虑的因素像选择学习率和周期数对于获得良好结果是非常重要的。

In this chapter we are going to look at two other types of computer vision problems: multi-label classification and regression. The first one is when you want to predict more than one label per image (or sometimes none at all), and the second is when your labels are one or several numbers—a quantity instead of a category.

在本章节，我们将学习另外两个类型的计算机视觉问题：多标签分类和回归。第一个问题是当我们想要预测单张图像有多个标签时（或有时间根本什么都没有）遇到到，第二个问题是当你的标注是一个或多个时：数量而不是类别。

In the process will study more deeply the output activations, targets, and loss functions in deep learning models.

在这个过程中会学习在深度学习模型中更深层的输出激活、目标和损失函数。

## Multi-Label Classification

## 多标签分类

Multi-label classification refers to the problem of identifying the categories of objects in images that may not contain exactly one type of object. There may be more than one kind of object, or there may be no objects at all in the classes that you are looking for.

多标签分类适用于那种在图像中不可能只包含一个目标类型的识别问题。有可能会超过一个目标类型，或在分类中你根本没有找到目标。

For instance, this would have been a great approach for our bear classifier. One problem with the bear classifier that we rolled out in <chapter_production> was that if a user uploaded something that wasn't any kind of bear, the model would still say it was either a grizzly, black, or teddy bear—it had no ability to predict "not a bear at all." In fact, after we have completed this chapter, it would be a great exercise for you to go back to your image classifier application, and try to retrain it using the multi-label technique, then test it by passing in an image that is not of any of your recognized classes.

例如，我们的熊分类器这一方法也许已经是一个非常好的方法了。在<章节：产品>中我们遇到了一个熊分类器的问题，如果用户上传不包含任何种类熊的图像，模型依然会说它是灰熊、黑熊或泰迪熊的一种。它不具备预测“根本没有熊”的能力。实际上，当我们完成本章节后，你重回图像分类器应用会是很好的练习。用多标签技术尝试重训练模型，然后通过传递给模型一张不包含任何你的识别分类的图像，对它进行测试。

In practice, we have not seen many examples of people training multi-label classifiers for this purpose—but we very often see both users and developers complaining about this problem. It appears that this simple solution is not at all widely understood or appreciated! Because in practice it is probably more common to have some images with zero matches or more than one match, we should probably expect in practice that multi-label classifiers are more widely applicable than single-label classifiers.

在实践中，我们没有看到很多人们以此为目的训练多标签分类器的例子，但是我们常常看到用户和开发人员抱怨这个问题。表明这个简单的解决方案根本没有被广泛的理解或认可！由于在实践中可能更常见的是很多零匹配或超过一个匹配项的图片，相比单标签分类器，我们可能应该期望在实践中多标签分类器更更广泛的应用。

First, let's see what a multi-label dataset looks like, then we'll explain how to get it ready for our model. You'll see that the architecture of the model does not change from the last chapter; only the loss function does. Let's start with the data.

首先，我们看一下多标签数据集看起来是什么样子，然后我们会解释如何为我们的模型把它准备好。你会在看到从上一章节开始，模型的架构没有做变化，只是变了损失函数。让我们从数据开始。

### The Data

### 数据

For our example we are going to use the PASCAL dataset, which can have more than one kind of classified object per image.

We begin by downloading and extracting the dataset as per usual:

我们的例子会使用PASCAL数据集，这个数据集对每张图像有超过一个种类的分类对象。与平常一样我们开始下载并抽取数据集：

```
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
```

This dataset is different from the ones we have seen before, in that it is not structured by filename or folder but instead comes with a CSV (comma-separated values) file telling us what labels to use for each image. We can inspect the CSV file by reading it into a Pandas DataFrame:

这个数据集与我们以前看到的那些是不同的，也就是说它没有通过文件名或文件夹结构化，而是用CSV（逗号分割值）文件告诉我们每一张图像所用的标签是什么。我们能够把它读取到Pandas DataFrame里，来查看这个CSV文件：

```
df = pd.read_csv(path/'train.csv')
df.head()
```

<table style="width: 370px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 10px;" align="center"></td>
    <td  style="width: 100px;" align="center">fname</td>
    <td  style="width: 160px;" align="center">labels</td>
    <td  style="width: 100px;" align="center">is_valid</td>
  </tr>
  <tr>
    <td align="center" font-weight="bold"><strong>0</strong></td>
    <td align="right">000005.jpg</td>
  	<td align="right">chair</td>
  	<td align="right">True</td>
  </tr>
  </tr>
    <td align="center"><strong>1</strong></td>
    <td align="right">000007.jpg</td>
  	<td align="right">car</td>
  	<td align="right">True</td>
  </tr>
  </tr>
    <td align="center"><strong>2</strong></td>
    <td align="right">000009.jpg</td>
  	<td align="right">horse person</td>
  	<td align="right">True</td>
  </tr>
  </tr>
    <td align="center"><strong>3</strong></td>
    <td align="right">000012.jpg</td>
  	<td align="right">car</td>
  	<td align="right">False</td>
  </tr>
  </tr>
    <td align="center" ><strong>4</strong></td>
    <td align="right">000016.jpg</td>
  	<td align="right">bicycle</td>
  	<td align="right">True</td>
  </tr>
</table>

As you can see, the list of categories in each image is shown as a space-delimited string.

正如你能看到的，每张图像的分类列表被显示为一个空格分割字符串。

### Sidebar: Pandas and DataFrames

### 侧边栏：Pandas and DataFrames

No, it’s not actually a panda! *Pandas* is a Python library that is used to manipulate and analyze tabular and time series data. The main class is `DataFrame`, which represents a table of rows and columns. You can get a DataFrame from a CSV file, a database table, Python dictionaries, and many other sources. In Jupyter, a DataFrame is output as a formatted table, as shown here.

You can access rows and columns of a DataFrame with the `iloc` property, as if it were a matrix:

是的，实际上它并不是熊猫！*Pandas*是一个Python库，用于操作和分析表格形式和时间序列数据。主要的类是`DataFrame`，它相当于一个有行和列的表。你能够从CSV文件、数据库表、Python字典和一些其它资源获取DataFrame。在Jupyter中，一个DataFrame输出的是一个格式化后的表，正如下面显示的。

你利用`iloc`特性能够获取一个DataFrame的行和列，好像它是一个矩阵：

```
df.iloc[:,0]
```

Out: ${\begin{array}{l,c}
0 &      000005.jpg \\
1&       000007.jpg  \\
2&       000009.jpg \\
3&       000012.jpg  \\
4&       000016.jpg  \\
&           ...      \\
5006&    009954.jpg \\
5007&    009955.jpg \\
5008&    009958.jpg  \\
5009&    009959.jpg  \\
5010&    009961.jpg  \end{array}\\
\begin{array}{l}Name: fname, Length: 5011, dtype: object
 \end{array}}$

```
df.iloc[0,:]
# Trailing :s are always optional (in numpy, pytorch, pandas, etc.),
#   so this is equivalent:
df.iloc[0]
```

Out: $\begin{array}{l,r} 
fname   &    000005.jpg\\
labels  &         chair\\
is\_valid &         True\end{array}$
$\begin{array}{l}&&Name: 0,& dtype: object
\end{array}$

You can also grab a column by name by indexing into a DataFrame directly:

你也能够通过列名直接索引到DataFrame中去来抓起一列：

```
df['fname']
```

Out: ${\begin{array}{l,c} 0  &     000005.jpg\\
1      & 000007.jpg\\
2      & 000009.jpg\\
3      & 000012.jpg\\
4      & 000016.jpg\\
       &    ...    \\
5006   & 009954.jpg\\
5007   & 009955.jpg\\
5008   & 009958.jpg\\
5009   & 009959.jpg\\
5010   & 009961.jpg\end{array}
\\
\begin{array}{l,l}Name: fname,& Length: 5011,& dtype: object\end{array}}$

You can create new columns and do calculations using columns:

你能够创建一个新列并用列做计算：

```
tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
tmp_df
```

<table style="width: 30px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 10px;" align="center"></td>
    <td  style="width: 10px;" align="center"><strong>a</strong></td>
    <td  style="width: 10px;" align="center"><strong>b</strong></td>
  </tr>
  <tr>
    <td align="center" font-weight="bold"><strong>0</strong></td>
    <td align="right">1</td>
  	<td align="right">3</td>
  </tr>
  <tr>
    <td align="center" font-weight="bold"><strong>1</strong></td>
    <td align="right">2</td>
  	<td align="right">4</td>
  </tr>
</table>

```
tmp_df['c'] = tmp_df['a']+tmp_df['b']
tmp_df
```

<table style="width: 30px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 10px;" align="center"></td>
    <td  style="width: 10px;" align="center"><strong>a</strong></td>
    <td  style="width: 10px;" align="center"><strong>b</strong></td>
    <td  style="width: 10px;" align="center"><strong>c</strong></td> 
  </tr>
  <tr>
    <td align="center" font-weight="bold"><strong>0</strong></td>
    <td align="right">1</td>
  	<td align="right">3</td>
    <td align="right">4</td>
  </tr>
  <tr>
    <td align="center" font-weight="bold"><strong>1</strong></td>
    <td align="right">2</td>
  	<td align="right">4</td>
  	<td align="right">6</td>
  </tr>
</table>

Pandas is a fast and flexible library, and an important part of every data scientist’s Python toolbox. Unfortunately, its API can be rather confusing and surprising, so it takes a while to get familiar with it. If you haven’t used Pandas before, we’d suggest going through a tutorial; we are particularly fond of the book [*Python for Data Analysis*](http://shop.oreilly.com/product/0636920023784.do) by Wes McKinney, the creator of Pandas (O'Reilly). It also covers other important libraries like `matplotlib` and `numpy`. We will try to briefly describe Pandas functionality we use as we come across it, but will not go into the level of detail of McKinney’s book.

Pandas是一个快速和灵活的库，它是每个数据科学的Python工具箱一个很重要的部分。不幸的是，它的API让人比较混乱和惊讶，所以需要花一段时间来熟悉它。如果你之前没有用过Pandas，我们建议你通过一个指引来学习使用它。我们特别喜欢由Pandas的创始人韦斯·麦金尼编写的书[Pyhon数据分析](http://shop.oreilly.com/product/0636920023784.do)（欧莱礼媒体发行）。它也覆盖了其它重要的库，如`matploatlib`和`numpy`的内容。我们会尝试简短的描述我们所使用的Pandas功能设计，但是不会达到麦金尼著作的细节水平。

### End sidebar

### 侧边栏结束

Now that we have seen what the data looks like, let's make it ready for model training.

既然我们已经知道了数据的样子，那让我们为模型训练做好准备。

### Constructing a DataBlock

### 创建一个数据块

How do we convert from a `DataFrame` object to a `DataLoaders` object? We generally suggest using the data block API for creating a `DataLoaders` object, where possible, since it provides a good mix of flexibility and simplicity. Here we will show you the steps that we take to use the data blocks API to construct a `DataLoaders` object in practice, using this dataset as an example.

我们如果从一个`DataFrame`对象转换为一个`DataLoaders`对象？对于创建一个`DataLoaders`对象，我们通常建议尽可能使用数据块API，因为它提供了柔性和简洁的完美融合。这里我们使用这个数据集做为一个例子，给你展示实践中使用数据块API来构建一个`DataLoaders`对象的步骤。

As we have seen, PyTorch and fastai have two main classes for representing and accessing a training set or validation set:

- `Dataset`:: A collection that returns a tuple of your independent and dependent variable for a single item
- `DataLoader`:: An iterator that provides a stream of mini-batches, where each mini-batch is a tuple of a batch of independent variables and a batch of dependent variables

正如你看到的，PyTorch和fastai有两个主要的类代表和访问训练集和验证集：

- `Dataset`：一个集合，返回单一数据自变量和因变量的一个元组
- `DataLoader`：一个迭代器，提供一串最小批次，每一最小批次是一个自变量批次和因变量批次的元组

On top of these, fastai provides two classes for bringing your training and validation sets together:

- `Datasets`:: An object that contains a training `Dataset` and a validation `Dataset`
- `DataLoaders`:: An object that contains a training `DataLoader` and a validation `DataLoader`

在这些之上，fastai提供了两个类把你的训练和验证集汇集在一起：

- `Datasets`：一个对象，包含一个训练`Dataset`和一个验证`Dataset`
- `DataLoaders`：一个对象，包含一个训练`DataLoader`和一个验证`DataLoader`

Since a `DataLoader` builds on top of a `Dataset` and adds additional functionality to it (collating multiple items into a mini-batch), it’s often easiest to start by creating and testing `Datasets`, and then look at `DataLoaders` after that’s working.

因为`DataLoader`构建在`Dataset`之上并增加了附加功能（把多个数据项收集在一个最小批次内），通常最简单的开始创建和测试`Datasets`，然后在运行后查看`DataLoaders`。

When we create a `DataBlock`, we build up gradually, step by step, and use the notebook to check our data along the way. This is a great way to make sure that you maintain momentum as you are coding, and that you keep an eye out for any problems. It’s easy to debug, because you know that if a problem arises, it is in the line of code you just typed!

Let’s start with the simplest case, which is a data block created with no parameters:

当你创建一个`数据块`时，我们要一步一步的逐步建立，并用notebook来逐步检查我们的数据。这是一个确保你保持编码动力的一个绝佳方法，然后随时关注出现的任何问题。它很容易调试，因为你知道如果一个问题产生，这个问题就在你刚刚敲击完成的代码行内！

让我们从不传参创建一个数据块这个最简单的例子开始：

```
dblock = DataBlock()
```

We can create a `Datasets` object from this. The only thing needed is a source—in this case, our DataFrame:

这样我们就能够创建一个`Datasets`对象。在这个例子中唯一需要的一个源是—我们的DataFrame：

```
dsets = dblock.datasets(df)
```

This contains a `train` and a `valid` dataset, which we can index into:

它包含了一个我们能够索引到的`训练`和`验证`数据集，

```
len(dsets.train),len(dsets.valid)
```

Out: (4009, 1002)

```
x,y = dsets.train[0]
x,y
```

Out: ${\begin{array}{llr}
			(fname  &     008663.jpg\\
			\ \ labels & car person\\
			\ \ is\_valid  & False
	  \end{array}
	  \\
 	  \begin{array}{llr}
			\ \ Name: 4346, &dtype: object,
	  \end{array}
	  \\
 	  \begin{array}{llllr}
			 \ \ fname &008663.jpg\\
			 \ \ labels    &  car person\\
 			 \ \ is\_valid &  False
	  \end{array}
	  \\
 	  \begin{array}{llr}
			\ \  Name: 4346,& dtype: object)
	  \end{array}
	}$

As you can see, this simply returns a row of the DataFrame, twice. This is because by default, the data block assumes we have two things: input and target. We are going to need to grab the appropriate fields from the DataFrame, which we can do by passing `get_x` and `get_y` functions:

正如你看到的，这个例子返回了两次数据帧的一行数据。因为这是默认的，数据块假设我们有两件事：输入和目标。我们需要从DataFrame合适的区域中抓取，可以通过传递`get_x`和`get_y`函数来做：

```
x['fname']
```

Out: '008663.jpg'

```
dblock = DataBlock(get_x = lambda r: r['fname'], get_y = lambda r: r['labels'])
dsets = dblock.datasets(df)
dsets.train[0]
```

Out: ('005620.jpg', 'aeroplane')

As you can see, rather than defining a function in the usual way, we are using Python’s `lambda` keyword. This is just a shortcut for defining and then referring to a function. The following more verbose approach is identical:

正如你所看到的，我们使用了Python的`lambda`关键字，而不是在普通的方法中定义一个函数。这只是定义的一个快捷方式，然后引用一个函数。下面是相同的更为冗长的方法：

```
def get_x(r): return r['fname']
def get_y(r): return r['labels']
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```

Out: ('002549.jpg', 'tvmonitor')

Lambda functions are great for quickly iterating, but they are not compatible with serialization, so we advise you to use the more verbose approach if you want to export your `Learner` after training (lambdas are fine if you are just experimenting).

We can see that the independent variable will need to be converted into a complete path, so that we can open it as an image, and the dependent variable will need to be split on the space character (which is the default for Python’s `split` function) so that it becomes a list:

Lambda函数对于快速迭代是极好的，但它们不兼任序列化，所以如果你想训练后输出你的`Learner`，我们建议你使用更为冗长的方法。（如果你只是做个尝试，lambdas是非常好的）

```
def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```

Out: (Path('/home/jhoward/.fastai/data/pascal_2007/train/002844.jpg'), ['train'])

To actually open the image and do the conversion to tensors, we will need to use a set of transforms; block types will provide us with those. We can use the same block types that we have used previously, with one exception: the `ImageBlock` will work fine again, because we have a path that points to a valid image, but the `CategoryBlock` is not going to work. The problem is that block returns a single integer, but we need to be able to have multiple labels for each item. To solve this, we use a `MultiCategoryBlock`. This type of block expects to receive a list of strings, as we have in this case, so let’s test it out:

实际打开图片并做张量转换，我们会需要做一些列的转换。块类型会提供我们那些步骤。我们能够使用先前已经使用过的相同块类型，但有一个例外：`ImageBlock`会运行良好，因为我们有一个指向有效图像的路径，但是`CategoryBlock`将不会运行。问题是块返回一个单整形，但我们需要能够对每个数据项有多标签。为了解决这个问题，我们使用`MultiCateoryBlock`类型。这个块类型要求接收字符串列表，正如在这个例子中我们已经有的数据那样，让我们测试一下它的输出：

```
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```

Out: (PILImage mode=RGB size=500x375,
         TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))

As you can see, our list of categories is not encoded in the same way that it was for the regular `CategoryBlock`. In that case, we had a single integer representing which category was present, based on its location in our vocab. In this case, however, we instead have a list of zeros, with a one in any position where that category is present. For example, if there is a one in the second and fourth positions, then that means that vocab items two and four are present in this image. This is known as *one-hot encoding*. The reason we can’t easily just use a list of category indices is that each list would be a different length, and PyTorch requires tensors, where everything has to be the same length.

> jargon: One-hot encoding: Using a vector of zeros, with a one in each location that is represented in the data, to encode a list of integers.

正如你看到的，我们的分类列表没有用常规的`CategoryBlock`规则方法编码。在那个例子中，基于在我们词汇表中它的位置，我们有一个单整形代表现在分类。虽然在这个例子中，有一个很多0列表，列表中1所在的位置是当前的分类。例如，如果1在第二和第四的位置，这就表示在这个图像中，代表两个和四个词汇数据项。这被称为*独热编码*。我们不能轻易的只用一个分类列表索引，原因是出每一列会是不一样的长度，而PyTorch需要张量，张量里所有事物必须是相同的长度。

> 术语：独热编码：使用包含一个1的0失量（1所在的每个位置是数据中所代表的类型），来编码整型列表。

Let’s check what the categories represent for this example (we are using the convenient `torch.where` function, which tells us all of the indices where our condition is true or false):

在这个例子中，我们检查一下分类代表的是什么（我们使用`torch.where`函数转换，这会告诉我们的条件是真还是假的所有索引）：

```
idxs = torch.where(dsets.train[0][1]==1.)[0]
dsets.train.vocab[idxs]
```

Out: (#1) ['dog']

With NumPy arrays, PyTorch tensors, and fastai’s `L` class, we can index directly using a list or vector, which makes a lot of code (such as this example) much clearer and more concise.

利用NumPy数组，PyTorch张量和fastai的`L`类，我们能够使用列表或失量生成一些更清晰和更简洁的代码（如这个例子）直接索引。

We have ignored the column `is_valid` up until now, which means that `DataBlock` has been using a random split by default. To explicitly choose the elements of our validation set, we need to write a function and pass it to `splitter` (or use one of fastai's predefined functions or classes). It will take the items (here our whole DataFrame) and must return two (or more) lists of integers:

到目前为止，我们已经忽略了`is_valid`列，这表示`DataBlock`已经通过默认的方式使用了随机分割。来明确的选择我们验证集的元素，我们需要编写一个函数并把它传递给`splitter`（或使用fastai先前定义的函数和类的一种）。它会携带数据项（在这里是我们整个DataFrame）并必须返回两个（或更多）整形列表： 

```
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]
```

Out: (PILImage mode=RGB size=500x333,
         TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

As we have discussed, a `DataLoader` collates the items from a `Dataset` into a mini-batch. This is a tuple of tensors, where each tensor simply stacks the items from that location in the `Dataset` item.

正如我们已经讨论过的，一个`DataLoader`从一个`Dataset`收集数据项到一个最小批次中。这是一个张量元组，每个张量简单的堆砌来自`Dataset`项中该位置的数据条目。

Now that we have confirmed that the individual items look okay, there's one more step we need to ensure we can create our `DataLoaders`, which is to ensure that every item is of the same size. To do this, we can use `RandomResizedCrop`:

现在我们已经确信独立的数据项是没有问题的，还有一步需要确保我们能够创建每个数据项是相同尺寸的`DataLoaders`。为了做到这一点，我们可以使用`RandomResizedCrop`方法：

```
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)
```

And now we can display a sample of our data:

现在我们能够展示一个我们数据的实例：

```
dls.show_batch(nrows=1, ncols=3)
```

Out: <img src="./_v_images/mulcat.png" alt="mulcat" style="zoom:100%;" />

Remember that if anything goes wrong when you create your `DataLoaders` from your `DataBlock`, or if you want to view exactly what happens with your `DataBlock`, you can use the `summary` method we presented in the last chapter.

记住，当创建来自你`DataBlock`的`DataLoaders`时，如果有任何问题，或如果你希望准确的看到你的`DataBlock`发生了什么，你可以使用在上一章节我们讲解的`summary`方法。

Our data is now ready for training a model. As we will see, nothing is going to change when we create our `Learner`, but behind the scenes, the fastai library will pick a new loss function for us: binary cross-entropy.

我们现在准备好了训练模型的数据。我们会看到，当我们创建我们的`Learner`时没有什么东西会改变，但是背后的景象是，fastai库会为我们选取一个新的损失函数：二值交叉熵。

### Binary Cross-Entropy

### 二值交叉熵

Now we'll create our `Learner`. We saw in <chapter_mnist_basics> that a `Learner` object contains four main things: the model, a `DataLoaders` object, an `Optimizer`, and the loss function to use. We already have our `DataLoaders`, we can leverage fastai's `resnet` models (which we'll learn how to create from scratch later), and we know how to create an `SGD` optimizer. So let's focus on ensuring we have a suitable loss function. To do this, let's use `cnn_learner` to create a `Learner`, so we can look at its activations:

现在来创建我们的`Learner`。在<章节：mnist基础>中我们看了`Learner`对象包含四个主要内容：模型、`DataLoaders`对象、优化器和要使用的损失函数。我们已经有了`DataLoaders`，能够利用fastai的`resnet`模型（后面我们会学习如何从零开始创建这一模型），我们知道如何创建一个`SGD`优化器。所以让我们聚焦在确保我们会有一个合适的损失函数。为此，让我们使用`cnn_learner`来创建一个`Learner`，以便我们能够查看它的激活情况：

```
learn = cnn_learner(dls, resnet18)
```

We also saw that the model in a `Learner` is generally an object of a class inheriting from `nn.Module`, and that we can call it using parentheses and it will return the activations of a model. You should pass it your independent variable, as a mini-batch. We can try it out by grabbing a mini batch from our `DataLoader` and then passing it to the model:

在学习器里我们也会看到模型通常是从`nn.Module`继承的一个类对象，并且我们能够使用圆括号调用它，同时它会返回模型的激活。你应该把你的自变量作为最小批次传给它。我们能够尝试从我们的`DataLoader`中抓取一个最小批次，然后把它传给模型：

```
x,y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
activs.shape
```

Out: torch.Size([64, 20])

Think about why `activs` has this shape—we have a batch size of 64, and we need to calculate the probability of each of 20 categories. Here’s what one of those activations looks like:

想一下，为什么激活是这个形状。我们有一个大小为64的批次，同时我们需要计算20个分类的每个概率。下面是那些激活其中一个的样子：

```
activs[0]
```

Out: TensorImage([ 0.7476, -1.1988,  4.5421, -1.5915, -0.6749,  0.0343, -2.4930, -0.8330, -0.3817, -1.4876, -0.1683,  2.1547, -3.4151,         -1.1743,  0.1530, -1.6801, -2.3067,  0.7063, -1.3358, -0.3715], grad_fn=<AliasBackward>)

> note: Getting Model Activations: Knowing how to manually get a mini-batch and pass it into a model, and look at the activations and loss, is really important for debugging your model. It is also very helpful for learning, so that you can see exactly what is going on.

> 注释：获取模型激活：知道如何手动的获取一个最小批次，并把它传递到模型中去，然后查看激活和损失情况，这对于调试你的模型是非常重要的。它对学习也非常有帮助。因此你能够准确的看到到底发生了什么。

They aren’t yet scaled to between 0 and 1, but we learned how to do that in <chapter_mnist_basics>, using the `sigmoid` function. We also saw how to calculate a loss based on this—this is our loss function from <chapter_mnist_basics>, with the addition of `log` as discussed in the last chapter:

他们还没有缩放到0到1之间，但在<章节：mnist基础>中我们知道如何利用`sigmoid`函数来做。我们也看了如何基于此计算损失函数（这是来自<章节：mnist基础>的损失函数，并增加了在上一章节讨论过的`对数`）：

```
def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, inputs, 1-inputs).log().mean()
```

Note that because we have a one-hot-encoded dependent variable, we can't directly use `nll_loss` or `softmax` (and therefore we can't use `cross_entropy`):

- `softmax`, as we saw, requires that all predictions sum to 1, and tends to push one activation to be much larger than the others (due to the use of `exp`); however, we may well have multiple objects that we're confident appear in an image, so restricting the maximum sum of activations to 1 is not a good idea. By the same reasoning, we may want the sum to be *less* than 1, if we don't think *any* of the categories appear in an image.
- `nll_loss`, as we saw, returns the value of just one activation: the single activation corresponding with the single label for an item. This doesn't make sense when we have multiple labels.

请注意，因为我们有了独热编码因变量，我们不能直接使用`nll_loss`或`softmax`（并且因而我们不能使用`交叉熵`）：

- `softmax`，我们知道它的预测合计需要为1，并倾向输出一个远大于其它的激活（由于`指数函数`的使用）。而然，我们可能会有多个目标我们是确认会出现在图像中，所以限制激活的最大合计为1就不是一个好主意了。基于同样的原因，如果我们不认为*其它*的分类显示在图像中，我们可能希望合计*小于*1。
- `nll_loss`，我们知道它只返回一个激活的值：单激活对应于对于每个数据项的单标签。当我们有多个标签时它就没有意义了。

On the other hand, the `binary_cross_entropy` function, which is just `mnist_loss` along with `log`, provides just what we need, thanks to the magic of PyTorch's elementwise operations. Each activation will be compared to each target for each column, so we don't have to do anything to make this function work for multiple columns.

> j: One of the things I really like about working with libraries like PyTorch, with broadcasting and elementwise operations, is that quite frequently I find I can write code that works equally well for a single item or a batch of items, without changes. `binary_cross_entropy` is a great example of this. By using these operations, we don't have to write loops ourselves, and can rely on PyTorch to do the looping we need as appropriate for the rank of the tensors we're working with.

换个角度说，`binary_cross_entropy`函数只是一个结合了`log`的`mnist_loss`函数，它正好提供了我们所需要的，这要感谢PyTorch神奇的元素操作。每个激活会与每一列的每个目标做对比，因此我们不必做任何事情，就能使得这个函数处理多列。

> 杰：我真正喜欢使用如PyTorch这种具有广播和元素操作的库的原因是，我经常发现我能够编写出对于单一数据项或批次数据项运行同样良好的代码，且不需要改代码。`binary_cross_entropy`是这样一个非常棒的例子。通过使用这些操作，我不必自己编写循环，我们需要对正在处理的数据具有合适的张量阶，且能够依赖PyTorch来做这个循环。

PyTorch already provides this function for us. In fact, it provides a number of versions, with rather confusing names!

PyTorch已经为我们提供了这个函数。实际上，它提供了很多版本，且让人相当迷惑的命名。

`F.binary_cross_entropy` and its module equivalent `nn.BCELoss` calculate cross-entropy on a one-hot-encoded target, but do not include the initial `sigmoid`. Normally for one-hot-encoded targets you'll want `F.binary_cross_entropy_with_logits` (or `nn.BCEWithLogitsLoss`), which do both sigmoid and binary cross-entropy in a single function, as in the preceding example.

`F.binary_cross_entropy`和它的模块相当于`nn.BCELoss`在独热编码目标上计算交叉熵，但是没有包含初始的`sigmoid`。通常对于独热编码目标你会想到`F.binary_cross_entropy_with_logits`（或`nn.BCEWithLogitsLoss`），正如之前的例子那样，在其单一函数里包含了sigmoid和二值交叉熵两者。

The equivalent for single-label datasets (like MNIST or the Pet dataset), where the target is encoded as a single integer, is `F.nll_loss` or `nn.NLLLoss` for the version without the initial softmax, and `F.cross_entropy` or `nn.CrossEntropyLoss` for the version with the initial softmax.

相同的，对于单标签数据集（如MNIST或宠物数据集），其目标做为一个单整形进行了编码，对于`F.nll_loss`或`nn.NLLLoss`的版本是没有包含初始的softmax，而`F.cross_entropy`或`nn.CrossEntropyLoss`的版本包含了softmax。

Since we have a one-hot-encoded target, we will use `BCEWithLogitsLoss`:

因为我们有独热编码目标，我们会使用`BCEWithLogitsLoss`：

```
loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss
```

Out: TensorImage(1.0342, grad_fn=<AliasBackward>)

We don't actually need to tell fastai to use this loss function (although we can if we want) since it will be automatically chosen for us. fastai knows that the `DataLoaders` has multiple category labels, so it will use `nn.BCEWithLogitsLoss` by default.

我们实际上并不需要告诉fastai来使用这个损失函数（虽然如果我们想的话，我们可以这么做），因为它会自动为我们进行选择。fastai知道`DataLoaders`有多分类标签，所以它会默认的使用`nn.BCEWithLogitsLoss`。

One change compared to the last chapter is the metric we use: because this is a multilabel problem, we can't use the accuracy function. Why is that? Well, accuracy was comparing our outputs to our targets like so:

与上一章节相比有一个改变是我们用的指标，因为这是一个多标签问题，我们不能使用精度函数。这是为什么呢？好吧，精度是我们的输出与我们目标的对比，如下所求：

```python
def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred = inp.argmax(dim=axis)
    return (pred == targ).float().mean()
```

The class predicted was the one with the highest activation (this is what `argmax` does). Here it doesn't work because we could have more than one prediction on a single image. After applying the sigmoid to our activations (to make them between 0 and 1), we need to decide which ones are 0s and which ones are 1s by picking a *threshold*. Each value above the threshold will be considered as a 1, and each value lower than the threshold will be considered a 0:

分类预测的是最高激活的那个值（这就是`argmax`做的事情）。在这里它不奏效是因为在一张图像上我们有超过一个的预测。在我们的激活应用sigmoid后（使得激活在0到1之间），我需要通过选择一个*阈值*来决定哪些是0，哪些是1。每个大于阈值的值会被识做为1，每个小于阈值的会被识做为0：

```python
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()
```

If we pass `accuracy_multi` directly as a metric, it will use the default value for `threshold`, which is 0.5. We might want to adjust that default and create a new version of `accuracy_multi` that has a different default. To help with this, there is a function in Python called `partial`. It allows us to *bind* a function with some arguments or keyword arguments, making a new version of that function that, whenever it is called, always includes those arguments. For instance, here is a simple function taking two arguments:

如果我们直接传递`accuracy_multi`做为一个指标，它会有默认为0.5的阈值。我们可能希望调整这个默认值，并创建一个与默认值不同的新的`accuracy_multi`版本。为了完成这个工作，在Python中有一个名为`partial`函数。它允许我们*绑定*一个有一些参数或关键值参数的函数，来对这个函数生成 一个新的版本，无论何时调用它，总会包含那些参数。这里有一个取两个参数的函数例子：

```
def say_hello(name, say_what="Hello"): return f"{say_what} {name}."
say_hello('Jeremy'),say_hello('Jeremy', 'Ahoy!')
```

Out: ('Hello Jeremy.', 'Ahoy! Jeremy.')

We can switch to a French version of that function by using `partial`:

通过使用`partial`我们能够转换为一个这个函数的法语版本：

```
f = partial(say_hello, say_what="Bonjour")
f("Jeremy"),f("Sylvain")
```

Out: ('Bonjour Jeremy.', 'Bonjour Sylvain.')

We can now train our model. Let's try setting the accuracy threshold to 0.2 for our metric:

我们现在能够训练我们的模型了。对于我们的指标，让我们尝试设置精度阈值为0.2：

```
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
```

| epoch | train_loss | valid_loss | accuracy_multi |  time |
| ----: | ---------: | ---------: | -------------: | ----: |
|     0 |   0.942663 |   0.703737 |       0.233307 | 00:08 |
|     1 |   0.821548 |   0.550827 |       0.295319 | 00:08 |
|     2 |   0.604189 |   0.202585 |       0.816474 | 00:08 |
|     3 |   0.359258 |   0.123299 |       0.944283 | 00:08 |

| epoch | train_loss | valid_loss | accuracy_multi |  time |
| ----: | ---------: | ---------: | -------------: | ----: |
|     0 |   0.135746 |   0.123404 |       0.944442 | 00:09 |
|     1 |   0.118443 |   0.107534 |       0.951255 | 00:09 |
|     2 |   0.098525 |   0.104778 |       0.951554 | 00:10 |

Picking a threshold is important. If you pick a threshold that's too low, you'll often be failing to select correctly labeled objects. We can see this by changing our metric, and then calling `validate`, which returns the validation loss and metrics:

选取一个阈值是非常重要的。如果你选取的阈值太低，你会经常无法选择正确的标签目标。我们能够通过改变我们的指标来观察，然后调用`validate`，返回验证的损失和指标：

```
learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()
```

Out: (#2) [0.10477833449840546,0.9314740300178528]

If you pick a threshold that's too high, you'll only be selecting the objects for which your model is very confident:

如果你选择的阈值太高，你只会看到对你的模型非常确信的选择目标：

```
learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()
```

Out: (#2) [0.10477833449840546,0.9429482221603394]

We can find the best threshold by trying a few levels and seeing what works best. This is much faster if we just grab the predictions once:

我们能够通过尝试一些等级和查看最佳实践查找最佳阈值。如果我们只抓取一次预测，这会更快些：

```
preds,targs = learn.get_preds()
```

Then we can call the metric directly. Note that by default `get_preds` applies the output activation function (sigmoid, in this case) for us, so we'll need to tell `accuracy_multi` to not apply it:

然后我们能够直接调用指标。注意，通过默认的方式`get_preds`会为我们应用输出激活函数（在这个例子中是sigmoid），所以我们需要告诉`accuracy_multi`不用应用它：

```
accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)
```

Out: TensorImage(0.9567)

We can now use this approach to find the best threshold level:

现在我们能使用这一方法来查找最优的阈值水平：

```
xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs,accs);
```

Out: ![img](data:image/png;base64,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)

In this case, we're using the validation set to pick a hyperparameter (the threshold), which is the purpose of the validation set. Sometimes students have expressed their concern that we might be *overfitting* to the validation set, since we're trying lots of values to see which is the best. However, as you see in the plot, changing the threshold in this case results in a smooth curve, so we're clearly not picking some inappropriate outlier. This is a good example of where you have to be careful of the difference between theory (don't try lots of hyperparameter values or you might overfit the validation set) versus practice (if the relationship is smooth, then it's fine to do this).

在这个例子中，我们使用验证集来选择一个超参（阈值），这就是验证集的用途。有时学习们会表达他们的顾虑，我们在验证集上可能会*过拟*，因为我们尝试了太多的值来看那个是最好的。然而，正如你在图中看到的，在这个例子中阈值改变的结果是一个平滑的曲线，所以我们显然不会选择那些离群值。这是一个好的例子，我们必须关注理论（不要尝试太多超参值，否则你可能会过拟验证集）与实践（如果关联是平滑的，那么这样做就好了）间的差别。

This concludes the part of this chapter dedicated to multi-label classification. Next, we'll take a look at a regression problem.

本章节介绍多标签分类的部分结束了。下面，我们会看一下回归问题。

## Regression

## 回归

It's easy to think of deep learning models as being classified into domains, like *computer vision*, *NLP*, and so forth. And indeed, that's how fastai classifies its applications—largely because that's how most people are used to thinking of things.

非常容易想到，深度学习模型做为分类进入如计算机视觉、自然语言处理等相关领域。的确，这就是fastai分类它的应用方式。很大程度上这是因为绝大多数人惯性思考事物的方式。

But really, that's hiding a more interesting and deeper perspective. A model is defined by its independent and dependent variables, along with its loss function. That means that there's really a far wider array of models than just the simple domain-based split. Perhaps we have an independent variable that's an image, and a dependent that's text (e.g., generating a caption from an image); or perhaps we have an independent variable that's text and dependent that's an image (e.g., generating an image from a caption—which is actually possible for deep learning to do!); or perhaps we've got images, texts, and tabular data as independent variables, and we're trying to predict product purchases... the possibilities really are endless.

然而事实上，这隐藏了很多有趣且深层的观点。一个模型是通过自变量和因变量连同它的损失函数来定义的。这表示相比简单基于领域分割，这会有真的有很广泛的模型系列。可能我们有一个图像的自变量和文本的因变量（即，根据这个图像生成一段文本）；或可能我们有一个文本的自变量和一个图像的因变量（即，根据文本生成一张图像 ，它是深度学习可能真正要做的！）；或可能我们已经获取了图像、文本和表格数据做为自己变量，我们尝试预测产品的购买情况...这可能真的是无穷无尽。

To be able to move beyond fixed applications, to crafting your own novel solutions to novel problems, it helps to really understand the data block API (and maybe also the mid-tier API, which we'll see later in the book). As an example, let's consider the problem of *image regression*. This refers to learning from a dataset where the independent variable is an image, and the dependent variable is one or more floats. Often we see people treat image regression as a whole separate application—but as you'll see here, we can treat it as just another CNN on top of the data block API.

为了能够超越固定应用，把你新颖的解决方案应用到与从不同的问题上，这有帮于正直理解数据块API（也可能是中层API，我们会在本书稍后看到）。让我们思考一下*图像回归* 问题这个例子。这是指从一个自变量为图像的数据集中学习，因变量为一个或多个浮点数。我们通常看到人员把图像回归视为一个完全独立的应用，但是在这里你会看到，我们能够只视它为数据块API之上的别一个卷积神经网络。

We're going to jump straight to a somewhat tricky variant of image regression, because we know you're ready for it! We're going to do a key point model. A *key point* refers to a specific location represented in an image—in this case, we'll use images of people and we'll be looking for the center of the person's face in each image. That means we'll actually be predicting *two* values for each image: the row and column of the face center.

我们准备直接跳到复杂的图像回归变量，因为我们知道你已经准备好了！我们会做一个关键点模型。*一个关键点* 指的是表示在图像中一个特定位置。在这个例子中，我们会使用人像，且在每一幅图像中我们会查找人脸的中心位置。这表示我们实际上对每张图像要预测两个变量：脸部中心位置的行和列。

### Assemble the Data

### 组装数据

We will use the [Biwi Kinect Head Pose dataset](https://icu.ee.ethz.ch/research/datsets.html) for this section. We'll begin by downloading the dataset as usual:

在本小节我们会使用[Biwi Kinect头部姿态数据集](https://icu.ee.ethz.ch/research/datsets.html)。像往常一样我们会通过下载数据集开始：

```
path = untar_data(URLs.BIWI_HEAD_POSE)
```

```
#hide
Path.BASE_PATH = path
```

Let's see what we've got!

我们看一下获得了什么！

```
path.ls().sorted()
```

Out: (#50) [Path('01'),Path('01.obj'),Path('02'),Path('02.obj'),Path('03'),Path('03.obj'),Path('04'),Path('04.obj'),Path('05'),Path('05.obj')...]

There are 24 directories numbered from 01 to 24 (they correspond to the different people photographed), and a corresponding *.obj* file for each (we won't need them here). Let's take a look inside one of these directories:

有24个从01到24的数字命名目录（它们相对应不同人的照片），且对每个目录有一个相对就的*.obj*文件（我们在这里并不需要它们）。让我们看看其中一个目录里面的内容：

```
(path/'01').ls().sorted()
```

Out: (#1000) [Path('01/depth.cal'),Path('01/frame_00003_pose.txt'),Path('01/frame_00003_rgb.jpg'),Path('01/frame_00004_pose.txt'),Path('01/frame_00004_rgb.jpg'),Path('01/frame_00005_pose.txt'),Path('01/frame_00005_rgb.jpg'),Path('01/frame_00006_pose.txt'),Path('01/frame_00006_rgb.jpg'),Path('01/frame_00007_pose.txt')...]

Inside the subdirectories, we have different frames, each of them come with an image (*_rgb.jpg*) and a pose file (*_pose.txt*). We can easily get all the image files recursively with `get_image_files`, then write a function that converts an image filename to its associated pose file:

子目录的内部，我们有不同的基础结构，每个子目录有一个图像（*_rgb.jpg*）和一个姿态文件（*_pose.txt*）。使用`get_image_files`我们能够很容易的递归获取所有图像文件，然后编写一个函数转换图像文件名到它相关联的姿态文件：

```
img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])
```

Out: Path('13/frame_00349_pose.txt')

Let's take a look at our first image:

我们看一下第一张图像：

```
im = PILImage.create(img_files[0])
im.shape
```

Out: (480, 640)

```
im.to_thumb(160)
```

Out: ![img](data:image/png;base64,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)

The Biwi dataset website used to explain the format of the pose text file associated with each image, which shows the location of the center of the head. The details of this aren't important for our purposes, so we'll just show the function we use to extract the head center point:

Biwi数据集网站曾经解释过关联每张图像的姿态文本文件的格式，展示了头部中央位置。这个细节对于我们的目标并不重要，所以我们只会展示用于抽取头部中心点的函数：

```
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
```

This function returns the coordinates as a tensor of two items:

这个函数返回了两个数据项的张量坐标：

```
get_ctr(img_files[0])
```

Out: tensor([384.6370, 259.4787])

We can pass this function to `DataBlock` as `get_y`, since it is responsible for labeling each item. We'll resize the images to half their input size, just to speed up training a bit.

我们能够把这个函数作为`get_y`传递给`数据块（DataBlock）`，因为对于标记的每个数据项它是可靠的。我们会改变图像到它们输入尺寸一半的大小，只是为了稍微加快一点训练速度。

One important point to note is that we should not just use a random splitter. The reason for this is that the same people appears in multiple images in this dataset, but we want to ensure that our model can generalize to people that it hasn't seen yet. Each folder in the dataset contains the images for one person. Therefore, we can create a splitter function that returns true for just one person, resulting in a validation set containing just that person's images.

需要注意的一个重要的点是我们不应该只使用一个随机分拆器。原因是在这个数据集中同一个人会显示在多张图像中，但我们希望确保我们的模型能够泛化还没有看到的那些人。在这个数据集中每个文件夹包含一个人的图像。因此，我们能够创建一个分拆器函数，对于只是一个人的返回真，在验证集只包含那个人图像的结果。

The only other difference from the previous data block examples is that the second block is a `PointBlock`. This is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates as it does to the images:

与之前的数据块例子的唯一其它区别是第二个块是一个`点块（PointBlock）`。这是有必要的，这样fastai就知道标签代表的是坐标。因此，它知道在做数据扩充时，它应该对这些坐标做像图像那样的同样扩充：

```
biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
```

> important: Points and Data Augmentation: We're not aware of other libraries (except for fastai) that automatically and correctly apply data augmentation to coordinates. So, if you're working with another library, you may need to disable data augmentation for these kinds of problems.
>
> 重要：点和数据扩充：我们不知道其它库（除了fastai）可自动和正确的应用数据扩充到坐标上。所以，如果你正在使用其它库，你需要禁用这些类型问题的数据扩充。

Before doing any modeling, we should look at our data to confirm it seems okay:

做任何建模前，我们应该查看我们的数据，确认它是正常的：

```
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
```

Out: <img src="./_v_images/show_batch.png" alt="show_batch" style="zoom:100%;" />

That's looking good! As well as looking at the batch visually, it's a good idea to also look at the underlying tensors (especially as a student; it will help clarify your understanding of what your model is really seeing):

看起来很好！除了批次查看，查看底层张量也是一个好注意（尤其是一名学生，它会帮助你清晰理解，你的模型真正看到的内容）

```
xb,yb = dls.one_batch()
xb.shape,yb.shape
```

Out: (torch.Size([64, 3, 240, 320]), torch.Size([64, 1, 2]))

Make sure that you understand *why* these are the shapes for our mini-batches.

Here's an example of one row from the dependent variable:

确保你理解*为什么*这些是我们最小指批次的形状。

这有一个来自因变量一行的例子：

```
yb[0]
```

Out: TensorPoint([[-0.3375,  0.2193]], device='cuda:6')

As you can see, we haven't had to use a separate *image regression* application; all we've had to do is label the data, and tell fastai what kinds of data the independent and dependent variables represent.

正如你可以看到的，我们不必使用一个单独的*图像回归*应用。我们必须要做的是标注数据，并告诉fastai自变量和因变量代表什么类型的数据。

It's the same for creating our `Learner`. We will use the same function as before, with one new parameter, and we will be ready to train our model.

它与我们创建学习器相同。我们会使用之前用过的，带有一个新参数的相同函数，同时我们将准备训练我们的模型。

### Training a Model

### 训练一个模型

As usual, we can use `cnn_learner` to create our `Learner`. Remember way back in <chapter_intro> how we used `y_range` to tell fastai the range of our targets? We'll do the same here (coordinates in fastai and PyTorch are always rescaled between -1 and +1):

像往常一样，我们能够使用`cnn_learner`来创建我们的`学习器`。记得在<章节：概述>里我们如何使用`y_range`告诉fastai我们目标范围方法吗？在这里我们会做同样的操作（在fastai和PyTorchr中的坐标刻度总是在-1和+1之间）：

```
learn = cnn_learner(dls, resnet18, y_range=(-1,1))
```

`y_range` is implemented in fastai using `sigmoid_range`, which is defined as:

在fastai中`y_range`利用`sigmoid_range`来实现，下面是它的代码定义：

```
def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
```

This is set as the final layer of the model, if `y_range` is defined. Take a moment to think about what this function does, and why it forces the model to output activations in the range `(lo,hi)`.

Here's what it looks like:

如果定义了`y_range`，这就是设置模型的最后层。花点时间想想为这个函数做了什么，并且为什么它聚焦说模型输出激活在`(lo,hi)`这个范围。

看起来是下面这个样子：

```
plot_function(partial(sigmoid_range,lo=-1,hi=1), min=-4, max=4)
```

Out: ![img](data:image/png;base64,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)

We didn't specify a loss function, which means we're getting whatever fastai chooses as the default. Let's see what it picked for us:

我们没有具体说明损失函数，这表示我们让fastai默认选择。我们来看一下它为我们做出的选择：

```
dls.loss_func
```

Out: FlattenedLoss of MSELoss()

This makes sense, since when coordinates are used as the dependent variable, most of the time we're likely to be trying to predict something as close as possible; that's basically what `MSELoss` (mean squared error loss) does. If you want to use a different loss function, you can pass it to `cnn_learner` using the `loss_func` parameter.

这很有意义，因为当坐标被用于因变量，大多情况下我们可能尝试尽可能预测某物，这是`MSELoss`（均方误差损失）的基础。如果我想使用不同的损失函数，你能够把=使用`loss_func`参数把它传递给`cnn_learner`。

Note also that we didn't specify any metrics. That's because the MSE is already a useful metric for this task (although it's probably more interpretable after we take the square root).

We can pick a good learning rate with the learning rate finder:

同时要注意的，我们不会具体说明任何指标。这是因为MSE对于这个任务已经是一个有用的指标（虽然我们取平均根后，它可能更好解释的）。

利用学习率查找器我们能够选取一个好的学习率：

```
learn.lr_find()
```

Out: SuggestedLRs(lr_min=0.005754399299621582, lr_steep=0.033113110810518265)

Out: ![img](data:image/png;base64,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)

We'll try an LR of 1e-2:

我们会尝试一个值为1e-2（0.01）的学习率（LR）：

```
lr = 1e-2
learn.fine_tune(3, lr)
```

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.049630 |   0.007602 | 00:42 |

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.008714 |   0.004291 | 00:53 |
|     1 |   0.003213 |   0.000715 | 00:53 |
|     2 |   0.001482 |   0.000036 | 00:53 |

Generally when we run this we get a loss of around 0.0001, which corresponds to an average coordinate prediction error of:



```
math.sqrt(0.0001)
```

Out: 0.01

This sounds very accurate! But it's important to take a look at our results with `Learner.show_results`. The left side are the actual (*ground truth*) coordinates and the right side are our model's predictions:

听起来非常精确！但重要的是用`Learner.show_results`看一下结果。左侧是实际的（真实数据组）坐标，右侧是我们模型的预测：

```
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
```

Out:<img src="./_v_images/learnshow_results.png" alt="learnshow_results" style="zoom:100%;" />

It's quite amazing that with just a few minutes of computation we've created such an accurate key points model, and without any special domain-specific application. This is the power of building on flexible APIs, and using transfer learning! It's particularly striking that we've been able to use transfer learning so effectively even between totally different tasks; our pretrained model was trained to do image classification, and we fine-tuned for image regression.

这非常神奇，仅仅几分钟的计算我们就能创建如此精确的关键的点模型，且没有任何具体专业领域应用。我就是构建在灵活的API之上，并利用迁移学习的力量！这非常惊人，甚至在完全不间的任务之间，我们能够如此有效率的使用迁移学习。我们的预训练模型被训练用于图像分类，而我们的微调是用于图像回归。

## Conclusion

## 结尾

In problems that are at first glance completely different (single-label classification, multi-label classification, and regression), we end up using the same model with just different numbers of outputs. The loss function is the one thing that changes, which is why it's important to double-check that you are using the right loss function for your problem.

乍一看这些问题是完全不同的（单标签分类、多标签分类和回归），我们最终使用了只有不同数量的输出相同模型。损失函数是可改变的一项，这就是为什么对于你的问题使用正确的损失函数进行复核是非常重要的。

fastai will automatically try to pick the right one from the data you built, but if you are using pure PyTorch to build your `DataLoader`s, make sure you think hard when you have to decide on your choice of loss function, and remember that you most probably want:

- `nn.CrossEntropyLoss` for single-label classification
- `nn.BCEWithLogitsLoss` for multi-label classification
- `nn.MSELoss` for regression

根据你构建的数据，fastai会尝试自动选择正确的损失函数，但如果你用纯PyTorch来构建你的`DataLodaer`，当你不得不决定做你的损失函数选择时，一定要认真思考：

- `nn.CrossEntropyLoss`是单标签分类损失函数
- `nn.BCEWithLogitsLoss`是多标签分类损失函数
- `nn.MSELoss`是回归损失函数

## Questionnaire

## 练习题

1. How could multi-label classification improve the usability of the bear classifier?
2. 多标签分类如何能够改善熊分类器的可用性？
3. How do we encode the dependent variable in a multi-label classification problem?
4. 在多标签分类问题上我们如何编码因变量？
5. How do you access the rows and columns of a DataFrame as if it was a matrix?
6. 如果它是一个矩阵，你怎么取DataFrame的行和列值？
7. How do you get a column by name from a DataFrame?
8. 对于DataFrame你怎么通过名称取列值？
9. What is the difference between a `Dataset` and `DataLoader`?
10. `Dataset`和`DataLoader`之间的区别是什么？
11. What does a `Datasets` object normally contain?
12. `Datasets`对象通常包含什么内容？
13. What does a `DataLoaders` object normally contain?
14. `DataLoaders`对象通常包含什么内容？
15. What does `lambda` do in Python?
16. 在Python中`lambda`做了什么？
17. What are the methods to customize how the independent and dependent variables are created with the data block API?
18. 用数据块API创建自变量和因变量的方法是什么？
19. Why is softmax not an appropriate output activation function when using a one hot encoded target?
20. 当使用独热编码目标时，为什么softmax不是合适的输出激活？
21. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
22. 当使用独热编码目标时，为什么`nll_loss`不是合适的损失函数？
23. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
24. `nn.BCELoss`和`nn.BCEWithLogitsLoss`之间的区别是什么？
25. Why can't we use regular accuracy in a multi-label problem?
26. 在多标签问题中我们为什么不能使用常规精度？
27. When is it okay to tune a hyperparameter on the validation set?
28. 在验证集上什么时候调超参是合适的？
29. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
30. 在fastai中`y_range`是如何实现的？（看看你是否能够在不偷看的情况下，自己实现并测试它）
31. What is a regression problem? What loss function should you use for such a problem?
32. 回归问题是什么？对于这类问题我应该使用什么损失函数？
33. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?
34. 你需要做什么来确保fastai库在你的输入图像和目标点坐标上应用相同的数据扩充？

### Further Research

### 深入研究

1. Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.
2. 阅读Pandas DataFrames的教程，并尝试一些方法对你感兴趣的方法。看本书的网站推荐教程。
3. Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.
4. 利用多标签分类重训练熊分类器。看看你是否能够使它在不包含熊的图片上有效运行，同时在网页应用上显示相关信息。在有两个不同品类的熊图像上尝试一下，检查利用多标签分类在单标签数据集上的是否影响了精度。