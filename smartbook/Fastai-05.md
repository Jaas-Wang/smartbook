# Image Classification

# 图像分类

Now that you understand what deep learning is, what it's for, and how to create and deploy a model, it's time for us to go deeper! In an ideal world deep learning practitioners wouldn't have to know every detail of how things work under the hood… But as yet, we don't live in an ideal world. The truth is, to make your model really work, and work reliably, there are a lot of details you have to get right, and a lot of details that you have to check. This process requires being able to look inside your neural network as it trains, and as it makes predictions, find possible problems, and know how to fix them.

现在你理解了深度学习是什么，为了什么，且如何创建和部署一个模型，对我们来说是时候更深一步了！在理想的世界里深度学习行业者不需要知道事物深层次如何运作的所有细节...但到目前为止，我们并不是生活在一个理想的世界里。真像是，要使得你的模型真正的工作且可行，有太多的细节你必须正确处理，很多细节你需要检查。这一过程需要能够查看你所训练的神经网络内部，并在它做出预测后查找可能的问题，及要知道如何来修复它们。

So, from here on in the book we are going to do a deep dive into the mechanics of deep learning. What is the architecture of a computer vision model, an NLP model, a tabular model, and so on? How do you create an architecture that matches the needs of your particular domain? How do you get the best possible results from the training process? How do you make things faster? What do you have to change as your datasets change?

所以，在本书从现在开始，我们要深入了解深度学习的机理。计算机视觉、自然语言处理、表格等模型的架构是什么？你如何创建一个匹配你的特定领域所需要的架构？你如何通过训练处理得到一个可能最好的结果？你如何处理事物更快？当你的数据集改变了你需要做什么样的变化？

We will start by repeating the same basic applications that we looked at in the first chapter, but we are going to do two things:

- Make them better.
- Apply them to a wider variety of types of data.

我们将会通过重复相同的基础应用开始，这些我们在第一章节已经看到过了，但我们将要做两件事：

- 使它们更好。
- 应用它们在一个各类繁多的数据类型。

In order to do these two things, we will have to learn all of the pieces of the deep learning puzzle. This includes different types of layers, regularization methods, optimizers, how to put layers together into architectures, labeling techniques, and much more. We are not just going to dump all of these things on you, though; we will introduce them progressively as needed, to solve actual problems related to the projects we are working on.

为了做这两件事情，我们将必须学习深度学习拼图的所有部分。这包含层的不同类型，正则化方法，优化器，如果把层合并到架构里，标注技术，及更多的其它内容。然而，我们不会把所有的内容只是倾倒给你。我们会通过解决与我们正在工作的项目相关的实际问题，然后根据需要逐步的介绍他们，

## From Dogs and Cats to Pet Breeds

## 从猫狗到宠物品种

In our very first model we learned how to classify dogs versus cats. Just a few years ago this was considered a very challenging task—but today, it's far too easy! We will not be able to show you the nuances of training models with this problem, because we get a nearly perfect result without worrying about any of the details. But it turns out that the same dataset also allows us to work on a much more challenging problem: figuring out what breed of pet is shown in each image.

在我们最先的模型，我们学会了如何来分类狗和猫。仅仅在几年前这被认为是一项非常有挑战的任务，而然在今天，它太容易了！在这个问题上我们无法给你展示模型训练的细微差别。因为我们取得了近乎完美的结果，而不担心任何细节。但事实上同样的数据集也允许我们处理更多挑战性的问题：弄清楚显示在每张图像中的宠物品种是什么。

In <chapter_intro> we presented the applications as already-solved problems. But this is not how things work in real life. We start with some dataset that we know nothing about. We then have to figure out how it is put together, how to extract the data we need from it, and what that data looks like. For the rest of this book we will be showing you how to solve these problems in practice, including all of the intermediate steps necessary to understand the data that you are working with and test your modeling as you go.

在<章节：概述>里我们展示了作为所解决问题的应用。但是不是现实生活中事物的运作方式。我们从一些完全不了解的数据集开始。然后计算出如何把他们聚合在一起，如何从其中抽取我们所需要的数据，且数据看起来像什么。本书的剩余部分我们会呈现给你如何解决这些实践中的问题，包括所有必须的中间步骤来理解我们正在处理的数据并随时测试你的模型

We already downloaded the Pet dataset, and we can get a path to this dataset using the same code as in <chapter_intro>:

我们已经下载了宠物数据集，并我们能够利用与<章节：概述>中同样的代码来获取得这个数据集的路径：

```
from fastai.vision.all import *
path = untar_data(URLs.PETS)
```

Now if we are going to understand how to extract the breed of each pet from each image we're going to need to understand how this data is laid out. Such details of data layout are a vital piece of the deep learning puzzle. Data is usually provided in one of these two ways:

- Individual files representing items of data, such as text documents or images, possibly organized into folders or with filenames representing information about those items
- A table of data, such as in CSV format, where each row is an item which may include filenames providing a connection between the data in the table and data in other formats, such as text documents and images

现在如果我们要理解如何从每张图像中抽取每个宠物的种类，我们就需要理解这个数据框架是怎么的。如数据框架的细节是深度学习拼图中一个非常重要的部分。数据通常以如下两种方式中的其中一种来提供：

- 独立的文件代表数据项目，例如文本文档或图像，尽量分组到目录中或用文件名代表这些数据项目的信息
- 一个表格数据，例如CSV格式，每行是一个可以包含文件名的数据项，这把表格中的数据和以其它格式存放的数据（如文本文档和图像）做了关联

There are exceptions to these rules—particularly in domains such as genomics, where there can be binary database formats or even network streams—but overall the vast majority of the datasets you'll work with will use some combination of these two formats.

To see what is in our dataset we can use the `ls` method:

对于这些规则有一些例外，如基因组学等特定领域，在这些领域提供二进制化数据库格式，甚至会是网络流。但是总体来说绝大多数我们将要使用的数据集会使用这两种模型中的某种进行组装。

我们能够使用`ls`方法来查看我们的数据集里有什么：

```
#hide
Path.BASE_PATH = path
```

```
path.ls()
```

Out: (#3) [Path('annotations'),Path('images'),Path('models')]

We can see that this dataset provides us with *images* and *annotations* directories. The [website](https://www.robots.ox.ac.uk/~vgg/data/pets/) for the dataset tells us that the *annotations* directory contains information about where the pets are rather than what they are. In this chapter, we will be doing classification, not localization, which is to say that we care about what the pets are, not where they are. Therefore, we will ignore the *annotations* directory for now. So, let's have a look inside the *images* directory:

我们能够看到提供给我们这个数据集有*图像*和*释文*目录。对于数据集[网站](https://www.robots.ox.ac.uk/~vgg/data/pets/)告诉了我们*释文*目录所包含的信息是哪里的宠物而不是他们是什么。在本章节，我们会处理分类，不是本地化，也就是说我们关心宠物是什么，不是他们是哪里的。因而，从现在我们会忽略*释文*目录。所以，让我们看一下*图像*目录的内部：

```
(path/"images").ls()
```

Out: (#7394) [Path('images/great_pyrenees_173.jpg'),Path('images/wheaten_terrier_46.jpg'),Path('images/Ragdoll_262.jpg'),Path('images/german_shorthaired_3.jpg'),Path('images/american_bulldog_196.jpg'),Path('images/boxer_188.jpg'),Path('images/staffordshire_bull_terrier_173.jpg'),Path('images/basset_hound_71.jpg'),Path('images/staffordshire_bull_terrier_37.jpg'),Path('images/yorkshire_terrier_18.jpg')...]

Most functions and methods in fastai that return a collection use a class called `L`. `L` can be thought of as an enhanced version of the ordinary Python `list` type, with added conveniences for common operations. For instance, when we display an object of this class in a notebook it appears in the format shown there. The first thing that is shown is the number of items in the collection, prefixed with a `#`. You'll also see in the preceding output that the list is suffixed with an ellipsis. This means that only the first few items are displayed—which is a good thing, because we would not want more than 7,000 filenames on our screen!

在fastai中大多数函数和方法返回一个名为L的集合类。L 能够被认为原生Python `list`类型的增强版，对于普通操作增加了方便性。例如，当在我们一个笔记中展示这个类的对象时（它以上述所展示的格式呈现）。首先展示的是带有井号（ # ）前缀的集合中数据项的数值。你也会看到在先前输出中带有省略号（ ... ）的列表后缀。这意味着只有前部的少量数据项被显示，这是个好事情，因为我们不想在屏幕上看到七千多个文件名！

By examining these filenames, we can see how they appear to be structured. Each filename contains the pet breed, and then an underscore (`_`), a number, and finally the file extension. We need to create a piece of code that extracts the breed from a single `Path`. Jupyter notebooks make this easy, because we can gradually build up something that works, and then use it for the entire dataset. We do have to be careful to not make too many assumptions at this point. For instance, if you look carefully you may notice that some of the pet breeds contain multiple words, so we cannot simply break at the first `_` character that we find. To allow us to test our code, let's pick out one of these filenames:

通过审视这些文件名，我们会发现他们所展示的结构是怎么样的。每个文件名都包含了宠物的品种，然后是下划线（ _ ）及一个数字，最终是文件的扩展名。我们需要构建的代码是从这单一`路径`内抽取品种。Jupyter notebooks会让这一工作很容易，因为我们能够逐步的建立起运行代码，然后把它应用到整个数据集上。在这一点上我们必须小心不要做太多假设。例如，如果你仔细看你会注意到一些宠物品种包含多个词，我们不能用所发现的第一个下划线来简单的分离它。我们来测试一个代码，挑出这些文件里其中一个文件名：

```
fname = (path/"images").ls()[0]
```

The most powerful and flexible way to extract information from strings like this is to use a *regular expression*, also known as a *regex*. A regular expression is a special string, written in the regular expression language, which specifies a general rule for deciding if another string passes a test (i.e., "matches" the regular expression), and also possibly for plucking a particular part or parts out of that other string.

从字符串中抽取信息最强大和灵活的方法是正如这里所使用的*正则表达式*，也被称为*regex*。正则表达式是一种用正则表达语言编写的特殊字符串，用特定通用规则来决定其它字符串是否通过测试（即，“匹配”正则表达），也可以取出其它字符串中的一个或多个特定部分。

In this case, we need a regular expression that extracts the pet breed from the filename.

在这个例子中，我们用正则表达示从文件名中抽取宠物品种。

We do not have the space to give you a complete regular expression tutorial here,but there are many excellent ones online and we know that many of you will already be familiar with this wonderful tool. If you're not, that is totally fine—this is a great opportunity for you to rectify that! We find that regular expressions are one of the most useful tools in our programming toolkit, and many of our students tell us that this is one of the things they are most excited to learn about. So head over to Google and search for "regular expressions tutorial" now, and then come back here after you've had a good look around. The [book's website](https://book.fast.ai/) also provides a list of our favorites.

在这里我们没有时间给你一个完整的正则表达式的指导，但是有很多优秀的在线教程，并且我们知道对于其中的一些优秀的工具你已经熟知了。如果你不知道，完全没有关系，对你来说这是一个很好的修正机会！我们发现在我们程序工具箱里正则表达式是最有用的工具之一并且我们的很多学生告诉我们学习它是他们的最激动的事情之一。所以现在到谷歌并搜索“正则表达式指引”吧，然后充分看完后再返回到这里。[本书网站](https://book.fast.ai/) 也提供了我们收藏的列表。

> a: Not only are regular expressions dead handy, but they also have interesting roots. They are "regular" because they were originally examples of a "regular" language, the lowest rung within the Chomsky hierarchy, a grammar classification developed by linguist Noam Chomsky, who also wrote *Syntactic Structures*, the pioneering work searching for the formal grammar underlying human language. This is one of the charms of computing: it may be that the hammer you reach for every day in fact came from a spaceship.
>
> 亚：正则表达式不仅仅很好用，他们也有一个很有趣的词根。他们是“正则”，因为他们源于一种“正则”语言的例子，在乔姆斯基等级里它是最低等级，这个等级是由语言学家诺姆·乔姆斯基开发的一个语法分类，他也编写了*句法结构*，基于人类语言对于正式说法的先驱研究工作。这是极具魅力的计算之一：它可能是你每天要伸手去拿，而实际上来是自太空飞船的锤子。

When you are writing a regular expression, the best way to start is just to try it against one example at first. Let's use the `findall` method to try a regular expression against the filename of the `fname` object:

当你正在编写一个正式表达式时，首先依据一个例子进行尝试是最好的开始方法。让我们用`findall`方法来尝试一个正则表达式处理`fname`对象的文件名：

```
re.findall(r'(.+)_\d+.jpg$', fname.name)
```

Out: ['great_pyrenees']

This regular expression plucks out all the characters leading up to the last underscore character, as long as the subsequence characters are numerical digits and then the JPEG file extension.

只要后续字符是数字数值，再然后是JPEG扩展文件名，这个正则表达式就会摘出从开始直到最后一个下划线前的所有字符。

Now that we confirmed the regular expression works for the example, let's use it to label the whole dataset. fastai comes with many classes to help with labeling. For labeling with regular expressions, we can use the `RegexLabeller` class. In this example we use the data block API we saw in <chapter_production> (in fact, we nearly always use the data block API—it's so much more flexible than the simple factory methods we saw in <chapter_intro>):

现在根据这个例子，我们确认正则表达式起做用了，让我们用它来标注整个数据集。fastai提供了很多类来帮助处理标注。对于使用正则表达式做标注，我们能够用`RegexLabeller`类。在这个例子中，我们使用在<章节：产品>中看到的数据块API（实际上，我们几乎一直使用数据块API，它比我们在<章节：概述>中看到的简单工厂方法要灵活的多）：

```
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path/"images")
```

One important piece of this `DataBlock` call that we haven't seen before is in these two lines:

`DataBlock`一个重要调用部分的两行代码，我们之前没有看到过：

```python
item_tfms=Resize(460),
batch_tfms=aug_transforms(size=224, min_scale=0.75)
```

These lines implement a fastai data augmentation strategy which we call *presizing*. Presizing is a particular way to do image augmentation that is designed to minimize data destruction while maintaining good performance.

这些代码行执行的是一个fastai数据增强策略，我们称为*填孔处理*。填孔处理是一个处理图像扩展的特定方法，这个设计用来当维持好的性能时最小化数据破坏。

## Presizing

## 填孔处理

We need our images to have the same dimensions, so that they can collate into tensors to be passed to the GPU. We also want to minimize the number of distinct augmentation computations we perform. The performance requirement suggests that we should, where possible, compose our augmentation transforms into fewer transforms (to reduce the number of computations and the number of lossy operations) and transform the images into uniform sizes (for more efficient processing on the GPU).

我们需要图像有相同的维度，所以他们就能够整理到张量中传递给GPU。我们也希望最小化我们执行的不同增强计算的数量。性能需求建议下我们应该尽可能压缩我们的增强转换组成为更少的转换（压缩计算数量和有损操作数量）并转换图像为相同尺寸（在GPU上更有效的处理）。

The challenge is that, if performed after resizing down to the augmented size, various common data augmentation transforms might introduce spurious empty zones, degrade data, or both. For instance, rotating an image by 45 degrees fills corner regions of the new bounds with emptyness, which will not teach the model anything. Many rotation and zooming operations will require interpolating to create pixels. These interpolated pixels are derived from the original image data but are still of lower quality.

这样做的挑战是，如果调整尺寸到增强尺寸后执行，很多常见的增强转换会导致虚假的空域，或数据质量退化，或同时出现这两种问题。例如， 通过45度转换一张图片，填充新的空白边界的角区域，不会教会模型任何东西。很多转换和缩放操作会需要插值来创建像素。这些插值像素是从原始图像数据衍生得来的，但依然质量是很低的。

To work around these challenges, presizing adopts two strategies that are shown in <presizing>:

1. Resize images to relatively "large" dimensions—that is, dimensions significantly larger than the target training dimensions.
2. Compose all of the common augmentation operations (including a resize to the final target size) into one, and perform the combined operation on the GPU only once at the end of processing, rather than performing the operations individually and interpolating multiple times.

来处理这些挑战，填孔处理采用了两个策略，这些策略<训练集上的填孔处理>上有所展示：

1. 调整图像到相对应的“大”量度，即，量度比目标训练维度明显更大。
2. 组合所有常见增强操作（包括调整到最终的目标尺寸）为一步，并在最终处理时，GPU上只执行组合操作一次，而不是独立的执行各操作及插值多次。

The first step, the resize, creates images large enough that they have spare margin to allow further augmentation transforms on their inner regions without creating empty zones. This transformation works by resizing to a square, using a large crop size. On the training set, the crop area is chosen randomly, and the size of the crop is selected to cover the entire width or height of the image, whichever is smaller.

调整大小的第一步，创建足够大的图像，他们有备用边缘以允许未来在内部区域增强转换，而不会产生空域。这一转换工作通过利用大剪切尺寸来调整图像尺寸为正方形。在训练集上，剪切面积是随机选择的，且剪切的尺寸是通过覆盖整个图像的宽或高，以最小者为准进行选定的。

In the second step, the GPU is used for all data augmentation, and all of the potentially destructive operations are done together, with a single interpolation at the end.

在第二步，GPU用于对所有数据增强，且所有潜在的破坏操作一起完成，并最终用一次插值。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00060.png" alt="Presizing on the training set" width="600" caption="Presizing on the training set" id="presizing" />
  </p>
  <p align="center">图：训练集上的填孔处理</p>
</div>

This picture shows the two steps:

1. *Crop full width or height*: This is in `item_tfms`, so it's applied to each individual image before it is copied to the GPU. It's used to ensure all images are the same size. On the training set, the crop area is chosen randomly. On the validation set, the center square of the image is always chosen.
2. *Random crop and augment*: This is in `batch_tfms`, so it's applied to a batch all at once on the GPU, which means it's fast. On the validation set, only the resize to the final size needed for the model is done here. On the training set, the random crop and any other augmentations are done first.

这张图像展示了两个步骤：

1. *宽或高全尺寸剪切* ：这个步骤在`item_tfms`中，所以在剪切到GPU前它用于每张独立的图像。它用于确保所有图像是相同的尺寸。在训练集上，剪切面积是随机选择的。在验证集上一直选择的是图像中心区域。
2. *随机剪切和增强* ：这个步骤在`batch_tfms`中，所以它立刻用于GPU上的一个批次，这意味着它很快。在验证集上，仅仅调整到模型最终所需要的大小。在训练集上，随机剪切和任何其它增强是首要做的。

To implement this process in fastai you use `Resize` as an item transform with a large size, and `RandomResizedCrop` as a batch transform with a smaller size. `RandomResizedCrop` will be added for you if you include the `min_scale` parameter in your `aug_transforms` function, as was done in the `DataBlock` call in the previous section. Alternatively, you can use `pad` or `squish` instead of `crop` (the default) for the initial `Resize`.

在fastai中来实施这一过程，我们用`Resize`做为对一个大尺寸的图像项转换，`RandomResizedCrop`作为一个更小尺寸的批次转换。如果在你的`aug_transforms`函数中包含了`min_scale`参数，会为你添加`RandomResizedCrop`，正如在前一部分`DataBlock`中调用的那样。或者，对于最初的`Resize`你能够使用`pad`或`squish`替代`crop`（默认）。

<interpolations> shows the difference between an image that has been zoomed, interpolated, rotated, and then interpolated again (which is the approach used by all other deep learning libraries), shown here on the right, and an image that has been zoomed and rotated as one operation and then interpolated just once on the left (the fastai approach), shown here on the left.

<interpolations>代码展示了他们的差异，一张图像被放大、插值、旋转，然后再次插值（所有其它深度学习库所使用的方法）被展示在右侧，及一张图像放大和旋转做为一次操作然后只是插值一次（这是fastai方法）被展示在左侧。

```
#hide_input
#id interpolations
#caption A comparison of fastai's data augmentation strategy (left) and the traditional approach (right).
dblock1 = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_y=parent_label,
                   item_tfms=Resize(460))
dls1 = dblock1.dataloaders([(Path.cwd()/'images'/'grizzly.jpg')]*100, bs=8)
dls1.train.get_idxs = lambda: Inf.ones
x,y = dls1.valid.one_batch()
_,axs = subplots(1, 2)

x1 = TensorImage(x.clone())
x1 = x1.affine_coord(sz=224)
x1 = x1.rotate(draw=30, p=1.)
x1 = x1.zoom(draw=1.2, p=1.)
x1 = x1.warp(draw_x=-0.2, draw_y=0.2, p=1.)

tfms = setup_aug_tfms([Rotate(draw=30, p=1, size=224), Zoom(draw=1.2, p=1., size=224),
                       Warp(draw_x=-0.2, draw_y=0.2, p=1., size=224)])
x = Pipeline(tfms)(x)
#x.affine_coord(coord_tfm=coord_tfm, sz=size, mode=mode, pad_mode=pad_mode)
TensorImage(x[0]).show(ctx=axs[0])
TensorImage(x1[0]).show(ctx=axs[1]);
```

Out:<img src="/Users/Y.H/Documents/GitHub/smartbook/smartbook/_v_images/crop-image.png" alt="crop-image" style="zoom:100%;" />

You can see that the image on the right is less well defined and has reflection padding artifacts in the bottom-left corner; also, the grass iat the top left has disappeared entirely. We find that in practice using presizing significantly improves the accuracy of models, and often results in speedups too.

你能够看到右侧的图像不是很好，且在左下角反映出人工填充痕迹，并且图像左上角的青草完全消失不见了。在实践中我们发现，使用填孔处理可极大的改善模型精度，且通常也会加快速度。

The fastai library also provides simple ways to check your data looks right before training a model, which is an extremely important step. We'll look at those next.

fastai库提供了一些简单方法，在训练模型前来检查你的数据看起来是否正确，这是异常重要的步骤。后续我们会看到这些方法。

### Checking and Debugging a DataBlock

### 检查并调试一个DataBlock

We can never just assume that our code is working perfectly. Writing a `DataBlock` is just like writing a blueprint. You will get an error message if you have a syntax error somewhere in your code, but you have no guarantee that your template is going to work on your data source as you intend. So, before training a model you should always check your data. You can do this using the `show_batch` method:

永远不能假定我们的代码会完美的运行。编写一个`DataBlock`只是就像写一个规划。如果在你的代码中的某个地方有语法错误你会收到错误消息，但你不能保证你的模板会如你计划的那样在你的数据资源上运行。所以，开始训练一个模型前，你应该一直检查你的数据。你可以用`show_batch`方法来进行检查：

```
dls.show_batch(nrows=1, ncols=3)
```

Out: <img src="/Users/Y.H/Documents/GitHub/smartbook/smartbook/_v_images/batch_1.png" alt="batch_1" style="zoom:100%;" />

Take a look at each image, and check that each one seems to have the correct label for that breed of pet. Often, data scientists work with data with which they are not as familiar as domain experts may be: for instance, I actually don't know what a lot of these pet breeds are. Since I am not an expert on pet breeds, I would use Google images at this point to search for a few of these breeds, and make sure the images look similar to what I see in this output.

看一下每一张图像，并检查每张图像是否有正确的宠物品种标注。通常，数据科学家处理的数据，他们可能并会与领域专家一样熟悉：例如，我实际上并不知道这些宠物品种是什么。因为我不是一个宠物品种专家，在这一点上我也许会用谷歌图片来搜索一些这些品种，并确保在输出时与我们看到这些图像看起来是相似的。

If you made a mistake while building your `DataBlock`, it is very likely you won't see it before this step. To debug this, we encourage you to use the `summary` method. It will attempt to create a batch from the source you give it, with a lot of details. Also, if it fails, you will see exactly at which point the error happens, and the library will try to give you some help. For instance, one common mistake is to forget to use a `Resize` transform, so you en up with pictures of different sizes and are not able to batch them. Here is what the summary would look like in that case (note that the exact text may have changed since the time of writing, but it will give you an idea):

如果在创建你的`DataBlock`时你犯了一个错误，在这一步这前你极有可能不会看到它。来调试一下，我们鼓励你使用`summary`方法。它会尝试从你给定的资源上创建一个批次，并带有很多细节。此外，如果它失败了，你会准确的看到哪个点产生了错误，并且库会尝试给你一些帮助。例如，一个常发生的错误是会忘记使用一个`Resize`转换，所以最终你会用不同尺寸的图像且无法批处理它们。在本下例中summary可能看起来是这样的（注解：由于编写时间原因，准确的文字描述可能发生了变化，但它会给你一个启示）：

```python
#hide_output
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))
pets1.summary(path/"images")
```

