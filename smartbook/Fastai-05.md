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