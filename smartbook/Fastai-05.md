# Image Classification

# 图像分类

Now that you understand what deep learning is, what it's for, and how to create and deploy a model, it's time for us to go deeper! In an ideal world deep learning practitioners wouldn't have to know every detail of how things work under the hood… But as yet, we don't live in an ideal world. The truth is, to make your model really work, and work reliably, there are a lot of details you have to get right, and a lot of details that you have to check. This process requires being able to look inside your neural network as it trains, and as it make s predictions, find possible problems, and know how to fix them.

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

Out:<img src="./_v_images/crop-image.png" alt="crop-image" style="zoom:100%;" />

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

Out: <img src="./_v_images/batch_1.png" alt="batch_1" style="zoom:100%;" />

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

 ```python
Setting-up type transforms pipelines
Collecting items from /Users/Y.H/.fastai/data/oxford-iiit-pet/images
Found 7390 items
2 datasets of sizes 5912,1478
Setting up Pipeline: PILBase.create
Setting up Pipeline: partial -> Categorize -- {'vocab': None, 'add_na': False}

Building one sample
  Pipeline: PILBase.create
    starting from
      /Users/Y.H/.fastai/data/oxford-iiit-pet/images/saint_bernard_60.jpg
    applying PILBase.create gives
      PILImage mode=RGB size=375x500
  Pipeline: partial -> Categorize -- {'vocab': (#37) ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue'...], 'add_na': False}
    starting from
      /Users/Y.H/.fastai/data/oxford-iiit-pet/images/saint_bernard_60.jpg
    applying partial gives
      saint_bernard
    applying Categorize -- {'vocab': (#37) ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue'...], 'add_na': False} gives
      TensorCategory(30)

Final sample: (PILImage mode=RGB size=375x500, TensorCategory(30))

Setting up after_item: Pipeline: ToTensor
Setting up before_batch: Pipeline: 
Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}

Building one batch
Applying item_tfms to the first sample:
  Pipeline: ToTensor
    starting from
      (PILImage mode=RGB size=375x500, TensorCategory(30))
    applying ToTensor gives
      (TensorImage of size 3x500x375, TensorCategory(30))

Adding the next 3 samples

No before_batch transform to apply

Collating items in a batch
Error! It's not possible to collate your items in a batch
Could not collate the 0-th members of your tuples because got the following shapes
torch.Size([3, 500, 375]),torch.Size([3, 333, 500]),torch.Size([3, 352, 500]),torch.Size([3, 500, 354])
 ```

You can see exactly how we gathered the data and split it, how we went from a filename to a *sample* (the tuple (image, category)), then what item transforms were applied and how it failed to collate those samples in a batch (because of the different shapes).

你能够准确的看到我们怎样收集数据并分割它，我们希望怎样从一个文件名到一个*样本*（元组（图像，分类）），然后数据项转换应用用它收集那些在一个批次中的样本是如何失败的（因为形状的不同）。

Once you think your data looks right, we generally recommend the next step should be using to train a simple model. We often see people put off the training of an actual model for far too long. As a result, they don't actually find out what their baseline results look like. Perhaps your probem doesn't need lots of fancy domain-specific engineering. Or perhaps the data doesn't seem to train the model all. These are things that you want to know as soon as possible. For this initial test, we'll use the same simple model that we used in <chapter_intro>:

一旦你认为你的数据看起来是好的，通常我们建议接下来的步骤应该是用来训练一个简单的模型。我们经常看到人们被一个实际训练的模型拖延太久了。这导致他们并不能实际找出他们基线结果是什么样子。也许你的问题并不是需要太多花哨的特定领域工程。或者也许数据似乎不能全部来训练模型。这些事情你知道的越快越好。对于这个初始测试，我们会用在<章节：概述>中使用过的，同样简单的模型：

```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
```

| epoch | train_loss | valid_loss | error_rate |  time |
| ----: | ---------: | ---------: | ---------: | ----: |
|     0 |   1.551305 |   0.322132 |   0.106225 | 00:19 |

| epoch | train_loss | valid_loss | error_rate |  time |
| ----: | ---------: | ---------: | ---------: | ----: |
|     0 |   0.529473 |   0.312148 |   0.095399 | 00:23 |
|     1 |   0.330207 |   0.245883 |   0.080514 | 00:24 |

As we've briefly discussed before, the table shown when we fit a model shows us the results after each epoch of training. Remember, an epoch is one complete pass through all of the images in the data. The columns shown are the average loss over the items of the training set, the loss on the validation set, and any metrics that we requested—in this case, the error rate.

正如我们之前做过的简短讨论，当我们拟合一个模型时每个周期训练后展示给我们如表格中显示的结果。记住，一个周期是完整遍历数据中所有的图像一个过程。列展示的是训练集数据项的平均损失，验证集上的损失，以及在本例中我们需要的那些指标：错误率。

Remember that *loss* is whatever function we've decided to use to optimize the parameters of our model. But we haven't actually told fastai what loss function we want to use. So what is it doing? fastai will generally try to select an appropriate loss function based on what kind of data and model you are using. In this case we have image data and a categorical outcome, so fastai will default to using *cross-entropy loss*.

记住*损失*是我们已经决定用于来优化我们模型参数的那种函数。但实际上我们并没有告诉fastai我们想用损失函数。那么它做了什么？fastai通常会基于数据类型和你所使用的模型，尝试选择一个合适的损失函数。在这个例子中，我们有图像数据和一个分类输出，所以fastai会默认使用*交叉熵损失函数*（即 *cross-entropy loss*）。

## Cross-Entropy Loss

## 交叉熵损失函数

*Cross-entropy loss* is a loss function that is similar to the one we used in the previous chapter, but (as we'll see) has two benefits:

- It works even when our dependent variable has more than two categories.
- It results in faster and more reliable training.

在之前的章节我们使用过一种损失函数，*交叉熵损失函数*与它相似，但（正如我们会看到的）有两个好处：

- 即使当我们的因变量有超过两个类别时它也可以工作。
- 其结果是更快和更可靠的训练。

In order to understand how cross-entropy loss works for dependent variables with more than two categories, we first have to understand what the actual data and activations that are seen by the loss function look like.

为了理解超过两个类别的因变量交叉熵损失函数如何工作的，我们首先必须要理解通过损失函数看到的真实数据和激活的情况。

### Viewing Activations and Labels

### 观察激活和标注

Let's take a look at the activations of our model. To actually get a batch of real data from our `DataLoaders`, we can use the `one_batch` method:

让我们看一下我们模型的激活。来自我们的`DataLoaders`实际取得的一个批次真实数据，我们可以使用`one_batch`方法：

```
x,y = dls.one_batch()
```

As you see, this returns the dependent and independent variables, as a mini-batch. Let's see what is actually contained in our dependent variable:

正如你看到的，这返回的是一个最小批次的因变量和自变量。让我们看一下在因变量里实际包含了什么：

```
y
```

Out:$\begin{matrix}TensorCategory([& 0,  &5,& 23,& 36,&  5, &20, &29, &34, &33,& 32,& 31,\\
 &24, &12, &36,&  8,& 26, &30, & 2,& 12,& 17,&  7,& 23, \\
&12,& 29, &21, & 4, &35,& 33, & 0, &20, &26,& 30, & 3, \\
& 6, &36, & 2, &17, &32,& 11, & 6, & 3, &30, & 5,& 26,\\
& 26,& 29,&  7,& 36,&31,& 26,& 26, & 8, &13,& 30,& 11,  \end{matrix}\\
\begin{matrix}&&&&&&&&&12, &36,& 31, &34,& 20,& 15,&  8, & 8,& 23&], device='cuda:5') \end{matrix}$

Our batch size is 64, so we have 64 rows in this tensor. Each row is a single integer between 0 and 36, representing our 37 possible pet breeds. We can view the predictions (that is, the activations of the final layer of our neural network) using `Learner.get_preds`. This function either takes a dataset index (0 for train and 1 for valid) or an iterator of batches. Thus, we can pass it a simple list with our batch to get our predictions. It returns predictions and targets by default, but since we already have the targets, we can effectively ignore them by assigning to the special variable `_`:

我们批处理尺寸大小是64，所以在这个张量里我们有64行。每一行是介于从0到36之间的整数，代表了我们37个可能宠物品种。我们能够用`Learner.get_preds`来观察预测情况（即我们的神经网络最后一层的激活）。这个函数要么接受一个数据集的索引（0代表训练，1代表验证）要么接受批次的迭代器。因而，我们能够传递给它一个包含我们批次的简单列表来得到我们的预测。它默认会返回预测和目标，但因为我们已经有了目标，我们能够通过分配一个特定变量`_`来有效的忽略它们：

```
preds,_ = learn.get_preds(dl=[(x,y)])
preds[0]
```

Out: $\begin{matrix}tensor([
&9.9911e-01,& 5.0433e-05, &3.7515e-07, &8.8590e-07, &8.1794e-05,& 1.8991e-05, &9.9280e-06, \\
&5.4656e-07, &6.7920e-06, &2.3486e-04, &3.7872e-04, &2.0796e-05,& 4.0443e-07,& 1.6933e-07, \\
&2.0502e-07, &3.1354e-08,&9.4115e-08, &2.9782e-06,& 2.0243e-07,& 8.5262e-08, &1.0900e-07,\\
 &1.0175e-07, &4.4780e-09,& 1.4285e-07, &1.0718e-07, &8.1411e-07, &3.6618e-07,& 4.0950e-07, \\
&3.8525e-08, &2.3660e-07, &5.3747e-08, &2.5448e-07,&6.5860e-08,& 8.0937e-05, &2.7464e-07,\\
 &5.6760e-07,& 1.5462e-08 ]) \end{matrix}$

The actual predictions are 37 probabilities between 0 and 1, which add up to 1 in total:

实际预测是37个从0到1之间的概率，他们合计为1：

```
len(preds[0]),preds[0].sum()
```

Out: (37, tensor(1.0000))

To transform the activations of our model into predictions like this, we used something called the *softmax* activation function.

转换我们模型的激活为像这样的预测，我们使用了叫做*softmax*的激活函数。

### Softmax

### Softmax激活函数

In our classification model, we use the softmax activation function in the final layer to ensure that the activations are all between 0 and 1, and that they sum to 1.

Softmax is similar to the sigmoid function, which we saw earlier. As a reminder sigmoid looks like this:

在我们的分类模型中，我们在最后的层使用softmax激活函数以确保激活都处于从0到1之间，且它们的合计为1.

Softmax与我们早先看到的S函数类似。提醒一下S函数看起来像这个样子：

```
plot_function(torch.sigmoid, min=-4,max=4)
```

Out: ![img](data:image/png;base64,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)

We can apply this function to a single column of activations from a neural network, and get back a column of numbers between 0 and 1, so it's a very useful activation function for our final layer.

我们能够把这一函数应用到神经网络单一列的激活，并获取一列介于从0到1之间的数值，所以对于我们最后的层它是一个非常有用的激活函数。

Now think about what happens if we want to have more categories in our target (such as our 37 pet breeds). That means we'll need more activations than just a single column: we need an activation *per category*. We can create, for instance, a neural net that predicts 3s and 7s that returns two activations, one for each class—this will be a good first step toward creating the more general approach. Let's just use some random numbers with a standard deviation of 2 (so we multiply `randn` by 2) for this example, assuming we have 6 images and 2 possible categories (where the first column represents 3s and the second is 7s):

现在想一下，如果我们想在我们目标中（例如我们有37个宠物品种）有更多的类别会发生什么。这意味着我们会需要更多的激活而不仅仅是一个单列：我们需要一个*宠物分类*的激活。例如，我们能够创建一个神经网络来返回两个激活来预测数字3和数字7，每个类别一个激活。这是来创建更加通用方法的好的开端。在本例中我们只用标准差为2的一些随机数（所以我们用2乘以`randn`），假设我们有6张图像和2个分类（第一列代表数字3，第二列代表数字7）：

```
#hide
torch.random.manual_seed(42);
acts = torch.randn((6,2))*2
acts
```

Out: $\begin{array}{r} tensor(
		&[[ &0.6734, & 0.2576&],\\
       & [ &0.4689, & 0.4607&],\\
       & [&-2.2457, &-0.3727&],\\
       & [& 4.4164, &-1.2760&],\\
       & [& 0.9233, & 0.5347&],\\
       & [& 1.0698, & 1.6187&]]&)\end{array}$

We can't just take the sigmoid of this directly, since we don't get rows that add to 1 (i.e., we want the probability of being a 3 plus the probability of being a 7 to add up to 1):

我们不能指示直接采纳S函数，因为我们不能取所有的行且总计为1（即，我们认为数字3的概率加上数字7的概率总计为1）：

```
acts.sigmoid()
```

Out: $\begin{array}{r} tensor(
		&[[&0.6623, &0.5641&],\\
        &[&0.6151, &0.6132&],\\
        &[&0.0957, &0.4079&],\\
        &[&0.9881, &0.2182&],\\
        &[&0.7157, &0.6306&],\\
        &[&0.7446, &0.8346&]]&) \end{array}$

In <chapter_mnist_basics>, our neural net created a single activation per image, which we passed through the `sigmoid` function. That single activation represented the model's confidence that the input was a 3. Binary problems are a special case of classification problems, because the target can be treated as a single boolean value, as we did in `mnist_loss`. But binary problems can also be thought of in the context of the more general group of classifiers with any number of categories: in this case, we happen to have two categories. As we saw in the bear classifier, our neural net will return one activation per category.

在<章节：MNIST基础>中，我们创建了每张图像单激活的神经网络，我们通过`sigmoid`函数传递。单激活代表模型置信度为输入数据是数字3。二值问题是一个特定的分类问题事例，因为目标能够作为单布尔值进行处理，正如我们在`mnist_loss`做的那样。但二值问题也能被认为任意数值类别情况下更加通用的组分类器：在这个例子中，我们恰巧有两个类别。正如在熊分类器中我们看到的，我们的神经网络会返回每个类别的一个激活。

So in the binary case, what do those activations really indicate? A single pair of activations simply indicates the *relative* confidence of the input being a 3 versus being a 7. The overall values, whether they are both high, or both low, don't matter—all that matters is which is higher, and by how much.

那么在二值事例中，那些激活真正表明了什么呢？一对激活简单的指示输入的*相对*置信度是数字3不是数字7。整个数值，两者是否都高或都低，并不是问题，问题在于哪个更高，高多少。

We would expect that since this is just another way of representing the same problem, that we would be able to use `sigmoid` directly on the two-activation version of our neural net. And indeed we can! We can just take the *difference* between the neural net activations, because that reflects how much more sure we are of the input being a 3 than a 7, and then take the sigmoid of that:

我们会期望，因为这只是代表相同问题的别一个方法，我们能够在我们的神经网络的双激活版本上直接使用`sigmoid`函数。并且我们真的能够这样！我们只需要求神经神经激活间的*差*，因为这反映了我们输入的数字为3比为7更确信多少，然后求它的S函数：

```
(acts[:,0]-acts[:,1]).sigmoid()
```

Out: tensor([0.6025, 0.5021, 0.1332, 0.9966, 0.5959, 0.3661])

The second column (the probability of it being a 7) will then just be that value subtracted from 1. Now, we need a way to do all this that also works for more than two columns. It turns out that this function, called `softmax`, is exactly that:

第二列（它是数字7的概率）只是从1减去的值。现在我们需要一个方法也做所有的这些超过两列的处理。被称为`softmax`的方法准确来说是这样的：

```python
def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
```

> jargon: Exponential function (exp): Literally defined as `e**x`, where `e` is a special number approximately equal to 2.718. It is the inverse of the natural logarithm function. Note that `exp` is always positive, and it increases *very* rapidly!
>
> 术语：指数函数（exp）：字面上的定义是`e**x`，`e`是一个特定近似等于2.718的数值。它是自然对数函数的逆函数。注释`exp`一直是正值，并且它的增长*非常* 快速！

Let's check that `softmax` returns the same values as `sigmoid` for the first column, and those values subtracted from 1 for the second column:

我们来查看一下，`softmax`返回了`sigmoid`第一列同样的值，并且第二列的那些数值从1减去的：

```
sm_acts = torch.softmax(acts, dim=1)
sm_acts
```

Out: $\begin{array}{r} tensor(
		&[[0.6025, 0.3975],\\
        &[0.5021, 0.4979],\\
        &[0.1332, 0.8668],\\
        &[0.9966, 0.0034],\\
        &[0.5959, 0.4041],\\
        &[0.3661, 0.6339]]&)\end{array}$

`softmax` is the multi-category equivalent of `sigmoid`—we have to use it any time we have more than two categories and the probabilities of the categories must add to 1, and we often use it even when there are just two categories, just to make things a bit more consistent. We could create other functions that have the properties that all activations are between 0 and 1, and sum to 1; however, no other function has the same relationship to the sigmoid function, which we've seen is smooth and symmetric. Also, we'll see shortly that the softmax function works well hand-in-hand with the loss function we will look at in the next section.

If we have three output activations, such as in our bear classifier, calculating softmax for a single bear image would then look like something like <bear_softmax>.

`softmax`是`sigmoid`的多分类等效式，我们有超过两个分类且分类的概率必须合计为时，我们必须无一例外的使用它。及时只有两个分类我们也经常使用它，只是为了保证事物更加的一致。我们能够创建其它函数，它们具有所有激活在从0到1之间，且合计为1的属性。然而，其它函数都没有与S函数相同的逻辑关系，我们已经看过的平滑与对称。另外，在下一小节，我们将会很快看到softmax函数与损失函数非常好的协同工作。

如果我们有三个激活输出，正如在我们的熊分类中那样，对于单个熊的图像计算softmax，然后会看到如<熊分类softmax例子>中的内容。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00062.png" alt="Bear softmax example" width="280" id="bear_softmax" caption="Example of softmax on the bear classifier" />
  </p>
  <p align="center">图：熊分类softmax例子</p>
</div>

What does this function do in practice? Taking the exponential ensures all our numbers are positive, and then dividing by the sum ensures we are going to have a bunch of numbers that add up to 1. The exponential also has a nice property: if one of the numbers in our activations `x` is slightly bigger than the others, the exponential will amplify this (since it grows, well... exponentially), which means that in the softmax, that number will be closer to 1.

在实践中这个函数做了什么事情呢？使用指数可确保我们所有的数值是正的，然后除以合计数，以确保我们会有一组总计为1的数值。指数也有一个很好的属性：如果在我们激活`x`中的其中一个数值稍微比其它数据更大，指数会对些放大（由于它称指数级的增长），这意味在softmax中数值会接近1 。

Intuitively, the softmax function *really* wants to pick one class among the others, so it's ideal for training a classifier when we know each picture has a definite label. (Note that it may be less ideal during inference, as you might want your model to sometimes tell you it doesn't recognize any of the classes that it has seen during training, and not pick a class because it has a slightly bigger activation score. In this case, it might be better to train a model using multiple binary output columns, each using a sigmoid activation.)

直观的说，softmax函数*实际* 想从其它类型中取出一个类，所以当我们知道每一张图片有明确的标签，训练一个分类器它是理想的。（注解，在推理期间它可能不是太理想，你可能希望你的模型有时候告诉你它不能识别训练期间看到的任何分类，没有选出分类是因为它有一个稍微大一点的激活分数。在这个例子中，可能最好训练一个使用多个二值输出列的模型，每一列使用一个S激活。）

Softmax is the first part of the cross-entropy loss—the second part is log likeklihood.

Softmax是交叉熵损失函数的第一部分，第二部分是对数似然。

### Log Likelihood

### 对数似然

When we calculated the loss for our MNIST example in the last chapter we used:

在上一章节中，在我们计算MNIST损失的例子中，我们使用了：

```python
def mnist_loss(inputs, targets):
    inputs = inputs.sigmoid()
    return torch.where(targets==1, 1-inputs, inputs).mean()
```

Just as we moved from sigmoid to softmax, we need to extend the loss function to work with more than just binary classification—it needs to be able to classify any number of categories (in this case, we have 37 categories). Our activations, after softmax, are between 0 and 1, and sum to 1 for each row in the batch of predictions. Our targets are integers between 0 and 36.

正如我们从sigmoid转移到softmax，我们需要扩展损失函数以处理不仅仅二值分类，它需要能够分类任意数量种类（在例中，我们有37个种类）。在softmax之后，我们的激活处于从0到1之间，且批处理预测的全部行合计为1。我们的目标是在从0到36之间的整型数值。

In the binary case, we used `torch.where` to select between `inputs` and `1-inputs`. When we treat a binary classification as a general classification problem with two categories, it actually becomes even easier, because (as we saw in the previous section) we now have two columns, containing the equivalent of `inputs` and `1-inputs`. So, all we need to do is select from the appropriate column. Let's try to implement this in PyTorch. For our synthetic 3s and 7s example, let's say these are our labels:

在二值事例中，我们用了`torch.where`来选择从`inputs`到`1-inputs`之间的值。当我们处理一个有二个类型普通分类的二值分类问题时，实际上它甚至变的更容易，因为（正好我们在之前小节看到的）现在我们有二列，包含了`inputs`和`1-inputs`的等价物。所以，所有我们需要做的是从合适的列中做选择。让我们在PyTorch中尝试执行。对于我们人造的数字3和7的例子，假设这些是我们的标签：

```
targ = tensor([0,1,0,1,1,0])
```

and these are the softmax activations:

这些是softmax激活：

```
sm_acts
```

Out: $\begin{array}{r} tensor(
		&[[0.6025, 0.3975],\\
        &[0.5021, 0.4979],\\
        &[0.1332, 0.8668],\\
        &[0.9966, 0.0034],\\
        &[0.5959, 0.4041],\\
        &[0.3661, 0.6339]]&)\end{array}$

Then for each item of `targ` we can use that to select the appropriate column of `sm_acts` using tensor indexing, like so:

然后对于`targ`的每个数据项，使用张量索引我们能够用其来选择`sm_acts`的合适列，像这样：

```
idx = range(6)
sm_acts[idx, targ]
```

Out: tensor([0.6025, 0.4979, 0.1332, 0.0034, 0.4041, 0.3661])

To see exactly what's happening here, let's put all the columns together in a table. Here, the first two columns are our activations, then we have the targets, the row index, and finally the result shown immediately above:

来看一下到底这里发生了什么，我们把所有列放在一个表里。这里的头两列是我们的激活，然后是我们的目标值，行索引，以及最后立即展示上面的结果：

```
#hide_input
from IPython.display import HTML
df = pd.DataFrame(sm_acts, columns=["3","7"])
df['targ'] = targ
df['idx'] = idx
df['loss'] = sm_acts[range(6), targ]
t = df.style.hide_index()
#To have html code compatible with our script
html = t._repr_html_().split('</style>')[1]
html = re.sub(r'<table id="([^"]+)"\s*>', r'<table >', html)
display(HTML(html))
```

|        3 |          7 | targ |  idx |       loss |
| -------: | ---------: | ---: | ---: | ---------: |
| 0.602469 |   0.397531 |    0 |    0 |   0.602469 |
| 0.502065 |   0.497935 |    1 |    1 |   0.497935 |
| 0.133188 |   0.866811 |    0 |    2 |   0.133188 |
|  0.99664 | 0.00336017 |    1 |    3 | 0.00336017 |
| 0.595949 |   0.404051 |    1 |    4 |   0.404051 |
| 0.366118 |   0.633882 |    0 |    5 |   0.366118 |

Looking at this table, you can see that the final column can be calculated by taking the `targ` and `idx` columns as indices into the two-column matrix containing the `3` and `7` columns. That's what `sm_acts[idx, targ]` is actually doing.

看这个表，你能够看到最后一行能够通过`targ`和`idx`列作为包含`3`和`7`两列矩阵的索引而计算得出。 这就是`sm_acts[idx, targ]`实际做的事情。

The really interesting thing here is that this actually works just as well with more than two columns. To see this, consider what would happen if we added an activation column for every digit (0 through 9), and then `targ` contained a number from 0 to 9. As long as the activation columns sum to 1 (as they will, if we use softmax), then we'll have a loss function that shows how well we're predicting each digit.

真正有有趣的事情是，这个功能实际上对于更多列（大于两列）也一样有效。为此，考虑一下如果我们对每个数字（从0到9）都增加一个激活会发生什么，且`targ`包含的数值从0到9。只要所有激活列的合计为1（如果我们使用softmax它们就会有这样的结果），然后我们会用一个损失函数展示我们预测每个数字的良好情况。

We're only picking the loss from the column containing the correct label. We don't need to consider the other columns, because by the definition of softmax, they add up to 1 minus the activation corresponding to the correct label. Therefore, making the activation for the correct label as high as possible must mean we're also decreasing the activations of the remaining columns.

我们只从包含正确标签的列选取损失。我们不需要考虑其它列，因为依据softmax的定义，他们合计为1，减去正确标签的相应激活。因为，对于正确标签使其激活尽可能的高，这就一定意味着我们已经减小了剩余列的激活。

PyTorch provides a function that does exactly the same thing as `sm_acts[range(n), targ]` (except it takes the negative, because when applying the log afterward, we will have negative numbers), called `nll_loss` (*NLL* stands for *negative log likelihood*):

Pytorch提供了一个函数叫做`nll_loss`（NLL为负的对于似然），做了与`sm_acts[range(n), targ]`完全相同的事情（除了它取负，因为在之后的应用对数时，我们会有一个负数）：

```
-sm_acts[idx, targ]
```

Out: tensor([-0.6025, -0.4979, -0.1332, -0.0034, -0.4041, -0.3661])

```
F.nll_loss(sm_acts, targ, reduction='none')
```

Out: tensor([-0.6025, -0.4979, -0.1332, -0.0034, -0.4041, -0.3661])

Despite its name, this PyTorch function does not take the log. We'll see why in the next section, but first, let's see why taking the logarithm can be useful.

尽管它名字了，但这个PyTorch函数并没有采用对数。在之后的小节里我们会看到为什么这样，但首先，让我们看一下为什么采纳对数能够有用。

### Taking the Log

### 采纳对数 

The function we saw in the previous section works quite well as a loss function, but we can make it a bit better. The problem is that we are using probabilities, and probabilities cannot be smaller than 0 or greater than 1. That means that our model will not care whether it predicts 0.99 or 0.999. Indeed, those numbers are so close together—but in another sense, 0.999 is 10 times more confident than 0.99. So, we want to transform our numbers between 0 and 1 to instead be between negative infinity and infinity. There is a mathematical function that does exactly this: the *logarithm* (available as `torch.log`). It is not defined for numbers less than 0, and looks like this:

在之前的小节我们看到这个函数作为损失函数工作的十分好，但我们能够使它更好一点。问题是我们使用了概率，且概率不能比0小或比1大。意味我们模型将不考虑它测试是0.99或0.999。确实，那些数字是如何的接近，但其它视角看，0.999比0.99的确信度多10倍。所以，作为替代我们想转换0和1之间的数值到负无穷和正无穷。这里有一个数学函数可准确的做这个事情：对数（以`torch.log`获得）。对于比0小的数值它没有定义，看起来是下面这个样子：

```
plot_function(torch.log, min=0,max=4)
```

Out: ![img](data:image/png;base64,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)

Does "logarithm" ring a bell? The logarithm function has this identity:

“对数”有印象吗？对数函数有这个特点：

```
y = b**a
a = log(y,b)
```

In this case, we're assuming that `log(y,b)` returns *log y base b*. However, PyTorch actually doesn't define `log` this way: `log` in Python uses the special number `e` (2.718...) as the base.

在这个例子中，我们假设`log(y,b)`返回`b为底的对数y`，PyTorch实际上没有定义`log`这个方法：在Python中`对数`使用特定数值`e`(2.718...)为底。

Perhaps a logarithm is something that you have not thought about for the last 20 years or so. But it's a mathematical idea that is going to be really critical for many things in deep learning, so now would be a great time to refresh your memory. The key thing to know about logarithms is this relationship:

也许对数是你大约近20年没有想过的事情。但是它将是一个在深度学习中对于很多事情是真正关键的数学概念，所以现在也许是恢复你的记忆的最好时刻吧。关于对数关键要知道的事情是这个逻辑关系：

```
log(a*b) = log(a)+log(b)
```

When we see it in that format, it looks a bit boring; but think about what this really means. It means that logarithms increase linearly when the underlying signal increases exponentially or multiplicatively. This is used, for instance, in the Richter scale of earthquake severity, and the dB scale of noise levels. It's also often used on financial charts, where we want to show compound growth rates more clearly. Computer scientists love using logarithms, because it means that modification, which can create really really large and really really small numbers, can be replaced by addition, which is much less likely to result in scales that are difficult for our computers to handle.

当我们在这样的格式中看到它时，看起来有点无聊。但想一下这真正表达的意思是什么。它表达的意思是当呈极大的几何级数或乘法倍数级增长时对数呈线性增长。例如，这被用于地震严重程度的里氏等级和噪声水平的分贝等级。它也经常被用于金融图表，我们想来展示更加清晰的组合增长率。计算机科学家喜欢使用对数，因为它意味能创建的真正十分十分巨大和十分十分小的数值，通过加法进行修改替换，这极不可能导致在规模计算上我们计算机会处理困难。

> s: It's not just computer scientists that love logs! Until computers came along, engineers and scientists used a special ruler called a "slide rule" that did multiplication by adding logarithms. Logarithms are widely used in physics, for multiplying very big or very small numbers, and many other fields.
>
> 西：不只是计算机科学家喜爱对数！在计算机出现之前，工程师和科学家使用一种称为“滑尺”的特殊尺子，通过加对数做乘法计算。对数在物理学中，非常大或非常小的数值乘法和很多其它领域都被广泛使用。

Taking the mean of the positive or negative log of our probabilities (depending on whether it's the correct or incorrect class) gives us the *negative log likelihood* loss. In PyTorch, `nll_loss` assumes that you already took the log of the softmax, so it doesn't actually do the logarithm for you.

对于我们的概率取正或负对数的平均值（依据它是正确或错误的分类）提供给我们*负对数似然*损失。在PyTorch中，`nll_loss`假定我们已经取了softmax的对数，所以它不实际上不会对为计算对数。

> warning: Confusing Name, Beware: The nll in `nll_loss` stands for "negative log likelihood," but it doesn't actually take the log at all! It assumes you have *already* taken the log. PyTorch has a function called `log_softmax` that combines `log` and `softmax` in a fast and accurate way. `nll_loss` is deigned to be used after `log_softmax`.
>
> 提醒：当心混淆命名：在`nll_loss`中的nll代表的是“负对数似然”，但它实际上根本没有取对数！它假设你*已经*取了对数。PyTorch有一个名为`log_softmax`的函数，它以快速且准确的方法把`log`和`softmax`组合在一起。`nll_loss`被设计用于`log_softmax`之后。

When we first take the softmax, and then the log likelihood of that, that combination is called *cross-entropy loss*. In PyTorch, this is available as `nn.CrossEntropyLoss` (which, in practice, actually does `log_softmax` and then `nll_loss`):

当我们首次采用softmax，且随后对它做对数似然时，这个组合被称为*交叉熵损失*。在PyTorch中以`nn.CrossEntropyLoss`获得（事实上，它实际做了`log_softmax`且随后做了`nll_loss`）：

```
loss_func = nn.CrossEntropyLoss()
```

As you see, this is a class. Instantiating it gives you an object which behaves like a function:

正如你看到的，这是一个类。实例化后它提供给你一个操作操函数的对象：

```
loss_func(acts, targ)
```

Out: tensor(1.8045)

All PyTorch loss functions are provided in two forms, the class just shown above, and also a plain functional form, available in the `F` namespace:

所有的PyTorch损失函数以两种形式被提供，如上所示的类，用另外一种纯函数的形式，在`F`命名空间中可获取：

```
F.cross_entropy(acts, targ)
```

Out: tensor(1.8045)

Either one works fine and can be used in any situation. We've noticed that most people tend to use the class version, and that's more often used in PyTorch's official docs and examples, so we'll tend to use that too.

By default PyTorch loss functions take the mean of the loss of all items. You can use `reduction='none'` to disable that:

每一个都运行极好且能够被用于任何情况下。我们已经提到绝大多数人倾向使用类版本，且在PyTorch的官方文档和例子中它更经常使用，所以我们也会倾向使用它。

PyTorch损失函数在默认情况下会求所有数据项的损失平均值。你能够用`reduction='none'`来禁用它：

```
nn.CrossEntropyLoss(reduction='none')(acts, targ)
```

Out: tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])

> s: An interesting feature about cross-entropy loss appears when we consider its gradient. The gradient of `cross_entropy(a,b)` is just `softmax(a)-b`. Since `softmax(a)` is just the final activation of the model, that means that the gradient is proportional to the difference between the prediction and the target. This is the same as mean squared error in regression (assuming there's no final activation function such as that added by `y_range`), since the gradient of `(a-b)**2` is `2*(a-b)`. Because the gradient is linear, that means we won't see sudden jumps or exponential increases in gradients, which should lead to smoother training of models.
>
> 西：当我们思考交叉熵的梯度时，对于交叉熵损失就显示出一个有趣的特征。`cross_entropy(a,b)`的梯度就是`softmax(a)-b`。因为`softmax(a)`就是最终的模型激活，这表达的意思是梯度对于预测和目标间的差异是成比例的。在回归中它与均方误差是相同的（假定没有最终的激活函数，例如通过`y_range`添加的函数），因为`(a-b)**2`的梯度是`2*(a-b)`。因为梯度是线性的，这意味着在梯度中我们不会看到突然的跳跃或指数级增长，它应该会导致模型的训练更加平滑。

We have now seen all the pieces hidden behind our loss function. But while this puts a number on how well (or badly) our model is doing, it does nothing to help us know if it's actually any good. Let's now see some ways to interpret our model's predictions.

我们现在已经看到了损失函数背后的所有隐藏部分。但当为我们模型做的是如何好（或坏）提供一个数值时，它无助于让我们知道是否它有实际的什么好处。现在让我们看一些方法来解释我们模型的预测。