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