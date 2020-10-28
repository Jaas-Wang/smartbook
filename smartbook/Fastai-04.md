

# Under the Hood: Training a Digit Classifier

# 追根溯源：训练一个数字分类

Having seen what it looks like to actually train a variety of models in Chapter 2, let’s now look under the hood and see exactly what is going on. We’ll start by using computer vision to introduce fundamental tools and concepts for deep learning.

在第二章已经发它看起来好像可以训练各种模型，现在让我们追根溯源并看实际上到底是怎么一回事。我们将通过使用计算机视觉开始介绍一些基础工具和深度学习的概念。

To be exact, we'll discuss the roles of arrays and tensors and of broadcasting, a powerful technique for using them expressively. We'll explain stochastic gradient descent (SGD), the mechanism for learning by updating weights automatically. We'll discuss the choice of a loss function for our basic classification task, and the role of mini-batches. We'll also describe the math that a basic neural network is actually doing. Finally, we'll put all these pieces together.

为了准确，我们会讨论数组、张量和传播的作用，一个强大的技术，使用他们具有深远意义。我们会解释随机剃度下降（SGD），通过自动更新权重来学习的机制。我们会讨论对于基础分类任务的损失函数选择，及最小批次的作用。我们也会描述一个基础神经网络实际在做的数学知识。最后，我们会把所有部分汇总在一起。

In future chapters we’ll do deep dives into other applications as well, and see how these concepts and tools generalize. But this chapter is about laying foundation stones. To be frank, that also makes this one of the hardest chapters, because of how these concepts all depend on each other. Like an arch, all the stones need to be in place for the structure to stay up. Also like an arch, once that happens, it's a powerful structure that can support other things. But it requires some patience to assemble.

在后续章节我们也会深度研究其它应用，并看这些概念和工具如何泛化。但是这一章是基石。坦率的说，这也使得这是难的章节之一。因为这些概念彼此都是依赖的。就像一个拱形，所以的石头需要在架构保持稳定的位置。也像一个拱形，一旦问题发生，这个强大的架构能够支持其它事情。因而它需要一些耐心去装配。

Let's begin. The first step is to consider how images are represented in a computer.

让我们开始吧。第一步来考虑在计算机里图片是怎样表达的。

## Pixels: The Foundations of Computer Vision

## 像素：计算机视觉的基础

In order to understand what happens in a computer vision model, we first have to understand how computers handle images. We'll use one of the most famous datasets in computer vision, [MNIST](https://en.wikipedia.org/wiki/MNIST_database), for our experiments. MNIST contains images of handwritten digits, collected by the National Institute of Standards and Technology and collated into a machine learning dataset by Yann Lecun and his colleagues. Lecun used MNIST in 1998 in [Lenet-5](http://yann.lecun.com/exdb/lenet/), the first computer system to demonstrate practically useful recognition of handwritten digit sequences. This was one of the most important breakthroughs in the history of AI.

为了理解在计算机视觉模型里到底到生了什么，我们先要理解计算机怎样处理图像。作为我们的尝试，我们会使用在计算机视觉最著名的数据集之一：[MNIST](https://en.wikipedia.org/wiki/MNIST_database)。MNIST包含了手写体数字图像，由美国国家标准技术研究所搜集，并被杨·立昆和他的同事纳入到一个机器学习数据集中。立昆1998年在[Lenet-5](http://yann.lecun.com/exdb/lenet/)卷积神经网络中使用了MNIST数据集，这是第一个计算机系统实际证明了可用的手写数字系列的识别。这是人工智能历史上最重要的突破之一。

## Sidebar: Tenacity and Deep Learning

## 侧边栏：坚韧和深度学习

The story of deep learning is one of tenacity and grit by a handful of dedicated researchers. After early hopes (and hype!) neural networks went out of favor in the 1990's and 2000's, and just a handful of researchers kept trying to make them work well. Three of them, Yann Lecun, Yoshua Bengio, and Geoffrey Hinton, were awarded the highest honor in computer science, the Turing Award (generally considered the "Nobel Prize of computer science"), in 2018 after triumphing despite the deep skepticism and disinterest of the wider machine learning and statistics community.

深度学习的故事是少数具有献身精神的研究人员坚韧和坚持之一。在1990年代和2000年代早期的希望（和过度宣传！）神经网络令人失去兴趣，只有少数的研究人员持续尝试使优化他们的工作。尽管更广泛的机器学习和统计团队深度质疑和漠不关心，在巨大成功后，2018年他们中的三人，杨·立昆、约书亚·本吉奥和杰弗里·辛顿被授予计算机科学界的最高荣誉：图灵奖（普通认为这是“计算机科学的诺贝尔奖”）。

Geoff Hinton has told of how even academic papers showing dramatically better results than anything previously published would be rejected by top journals and conferences, just because they used a neural network. Yann Lecun's work on convolutional neural networks, which we will study in the next section, showed that these models could read handwritten text—something that had never been achieved before. However, his breakthrough was ignored by most researchers, even as it was used commercially to read 10% of the checks in the US!

杰弗里·辛顿说过，即使学术论文怎样显示出比之前发布文章的更戏剧化的结果，也许会被顶级期刊和会议拒绝，只是因为他们用了一个神经网络。杨·立昆在卷积神经网络的工作，显示了这些模型能够阅读手写文本（有些内容被），一些工作以前从来没有完成过，在下一部分我们会进行研究。即便它被用于商业用途读取了美国10%的支票信息，然而，他的突破却被绝大多数研究人员所忽略。

In addition to these three Turing Award winners, there are many other researchers who have battled to get us to where we are today. For instance, Jurgen Schmidhuber (who many believe should have shared in the Turing Award) pioneered many important ideas, including working with his student Sepp Hochreiter on the long short-term memory (LSTM) architecture (widely used for speech recognition and other text modeling tasks, and used in the IMDb example in <chapter_intro>). Perhaps most important of all, Paul Werbos in 1974 invented back-propagation for neural networks, the technique shown in this chapter and used universally for training neural networks ([Werbos 1994](https://books.google.com/books/about/The_Roots_of_Backpropagation.html?id=WdR3OOM2gBwC)). His development was almost entirely ignored for decades, but today it is considered the most important foundation of modern AI.

除了这三名图灵获奖者，还有很多其它奋斗的研究人员为我们争取到了今天的成果。例如于尔根·施密德胡伯（很多人认为应该获得图灵奖）是很多重要想法的先驱，包括与他的学生塞普·霍克赖特在长短期记忆（LSTM）架构（广泛被用于语音识别和其它文本建模任务，及被用于在<概述>章节中的IMDb例子）上的工作。可能所有工作中重要的是在1974年保罗·韦伯斯发明的后向传播神经网络，这一技术在本章节会出现并被广泛用于训练神经网络[（韦伯斯 1994）](https://books.google.com/books/about/The_Roots_of_Backpropagation.html?id=WdR3OOM2gBwC))。他的发展几乎被忽略了整整十几年，但如今它被认为是现代人工智能最重要的基础。

There is a lesson here for all of us! On your deep learning journey you will face many obstacles, both technical, and (even more difficult) posed by people around you who don't believe you'll be successful. There's one *guaranteed* way to fail, and that's to stop trying. We've seen that the only consistent trait amongst every fast.ai student that's gone on to be a world-class practitioner is that they are all very tenacious.

这是一个对我们所有人的教训！在你的深度学习之旅，你将会面临很多障碍，技术上的和态度两者，后者甚至更艰难，你周边的人不相信你会取得成功。没有*肯定*失败的方法并停止尝试。我们已经看到了，在每一个fast.ai励志做一个世界一流水平行业者的学生中有个唯一一致特点是他们都非常有韧性。

## End sidebar

## 侧边栏结束

For this initial tutorial we are just going to try to create a model that can classify any image as a 3 or a 7. So let's download a sample of MNIST that contains images of just these digits:

我们将尝试创建一个能够分类任何3和7图片的模型做为本次初始教程。所以让我们下载一个只包含这两个数字图片的MNIST样本：

```python
path = untar_data(URLs.MNIST_SAMPLE)
#hide
Path.BASE_PATH = path
```

We can see what's in this directory by using `ls`, a method added by fastai. This method returns an object of a special fastai class called `L`, which has all the same functionality of Python's built-in `list`, plus a lot more. One of its handy features is that, when printed, it displays the count of items, before listing the items themselves (if there are more than 10 items, it just shows the first few):

你能够使用`ls`查看这个目录里的内容，这个方法是fastai增加的。本方法返回一个叫做`L`的特定fastai类对象，这与Python内置的`list`的实用性完全类似，只是增加了一些内容。它好用功能之一是，当输出时它会先显示条目数，随后列出条目自身（如果有超过10个条目，它只显示头几个）：

```python
path.ls()
```

out：(#3) [Path('valid'),Path('labels.csv'),Path('train')]

The MNIST dataset follows a common layout for machine learning datasets: separate folders for the training set and the validation set (and/or test set). Let's see what's inside the training set:

MNIST数据集遵从了机器学习数据集的通用框架：把文件夹分割为训练集和验证集（和/或测试集）。让我们看一下训练集的内部是什么：

```python
(path/'train').ls()
```

out: (#2) [Path('train/7'),Path('train/3')]

There's a folder of 3s, and a folder of 7s. In machine learning parlance, we say that "3" and "7" are the *labels* (or targets) in this dataset. Let's take a look in one of these folders (using `sorted` to ensure we all get the same order of files):

有一个3的文件夹和一个7的文件夹。在机器学习中的用语，我们说“3”和“7”是本数据集中的*标签*（或靶）。让我们看一下这些文件夹中的一个（使用*sorted*以确保我们完全取得相同顺序的文件）：

```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
```

out: (#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]

As we might expect, it's full of image files. Let’s take a look at one now. Here’s an image of a handwritten number 3, taken from the famous MNIST dataset of handwritten numbers:

正如我们可能期望的，它全是图像文件。让我们现在看一张。这是一张来自著名MNIST数据集的手写数字3：

```python
im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```

out: ![three_number](./_v_images/three_number.png)

Here we are using the `Image` class from the *Python Imaging Library* (PIL), which is the most widely used Python package for opening, manipulating, and viewing images. Jupyter knows about PIL images, so it displays the image for us automatically.

这里我们用了来自*python图像库*（PIL）的`Image`类，这是使用最为广泛的Python包用于打开、操作和查看图像。Jupyter知道PIL图片，所以它会为我们自动显示图片。

In a computer, everything is represented as a number. To view the numbers that make up this image, we have to convert it to a *NumPy array* or a *PyTorch tensor*. For instance, here's what a section of the image looks like, converted to a NumPy array:

在一个计算机里，所有的事情被表示为一个数值。查看组成这一图片的数值，我们必须把它转换为一个*NumPy数组*或一个*PyTorch张量*。例如，这是一张选中的图像看起来已转换为一个NumPy数组：

```python
array(im3)[4:10,4:10]
```

$
\begin{matrix} out:array([&[& 0, & 0, & 0, & 0, & 0, & 0&],\\ 
	& [&0,& 0,& 0,& 0,& 0,& 29&], \\ 
	& [&0,& 0,& 0,& 48,& 166,& 224&], \\ 
	& [&0,& 93,& 244,& 249,& 253,& 187&], \\
	& [&0,& 107,& 253,& 253,& 230,& 48&], \\
	& [&0,& 3,& 20,& 20,& 15,& 0&]]&dtype=uint8)
\end{matrix}
$

The `4:10` indicates we requested the rows from index 4 (included) to 10 (not included) and the same for the columns. NumPy indexes from top to bottom and left to right, so this section is located in the top-left corner of the image. Here's the same thing as a PyTorch tensor:

4:10指示的是我们需要索引号从4（包含）到10（不包含）的行和相同的列。NumPy索引号从上到下和从左到右，所以这一部分位置在图像的左上部。对于PyTorch张量是同样的事情：

```python
tensor(im3)[4:10,4:10]
```

$
\begin{matrix} out:array([&[& 0, & 0, & 0, & 0, & 0, & 0&],\\ 
	& [&0,& 0,& 0,& 0,& 0,& 29&], \\ 
	& [&0,& 0,& 0,& 48,& 166,& 224&], \\ 
	& [&0,& 93,& 244,& 249,& 253,& 187&], \\
	& [&0,& 107,& 253,& 253,& 230,& 48&], \\
	& [&0,& 3,& 20,& 20,& 15,& 0&]]&dtype=torch.uint8)
\end{matrix}
$

We can slice the array to pick just the part with the top of the digit in it, and then use a Pandas DataFrame to color-code the values using a gradient, which shows us clearly how the image is created from the pixel values:

我们能够切片数组挑出其中数值顶部的部分，然后用Pandas的数据结构利用一个梯度对数值颜色编码，就能够清晰的展示给我们图片被创建的像数值：

```python
#hide_output
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

out: <img src="./_v_images/att_00058.png" alt="att_00058" style="zoom:50%;"  />

You can see that the background white pixels are stored as the number 0, black is the number 255, and shades of gray are between the two. The entire image contains 28 pixels across and 28 pixels down, for a total of 784 pixels. (This is much smaller than an image that you would get from a phone camera, which has millions of pixels, but is a convenient size for our initial learning and experiments. We will build up to bigger, full-color images soon.)

你能够看到白色像素背景储存的数值是0，黑色的是数值255，及灰色阴影是在两者之间的数值。整张图像包含28个横向像素和28个竖向像素，一共是784个像素。（它与你从一个手机镜头拍摄的上百万像素图片极为相似，但对于我们开始学习和试验小像素是一个合适的尺寸。稍后我们会创建更大和彩色的图像。）

So, now you've seen what an image looks like to a computer, let's recall our goal: create a model that can recognize 3s and 7s. How might you go about getting a computer to do that?

所以，现在你已经看到了对于计算机来说一张图像的样子，来回想一下我们的目标：创建能够识别3和7的模型。你如何能够让计算机做这个事情呢？

> Warning: Stop and Think!: Before you read on, take a moment to think about how a computer might be able to recognize these two different digits. What kinds of features might it be able to look at? How might it be able to identify these features? How could it combine them together? Learning works best when you try to solve problems yourself, rather than just reading somebody else's answers; so step away from this book for a few minutes, grab a piece of paper and pen, and jot some ideas down…
>
> 警示：停下来并想一下：在你阅读之前，花点时间想一下计算机如何能够具备识别这两个不同数字的能力。它能够看什么类型的特征？它怎样能够识别这些特征？它怎么能够把他们结合在一起？当你尝试解决你自己的问题，而不只是阅读别人的答案的时候，学习效果最好。所以从本书移步离开几分钟，拿张纸和笔，并快速记下一些想法...

## First Try: Pixel Similarity

## 第一步尝试：像素相似处

So, here is a first idea: how about we find the average pixel value for every pixel of the 3s, then do the same for the 7s. This will give us two group averages, defining what we might call the "ideal" 3 and 7. Then, to classify an image as one digit or the other, we see which of these two ideal digits the image is most similar to. This certainly seems like it should be better than nothing, so it will make a good baseline.

所以，这是第一个想法：我们如何发现３图像每个像素的平均像素值，然后对７的图像做同样的事情。这会给我们两组平均值，定义我们认为“理想”的３和７。然后分类一张图像作为一个数字或另一个，我们看这两个理想的数字图片更像哪一个。这确实好像应该比无法识别任何东西更好，所以它会做作为一个好的基线。

> jargon: Baseline: A simple model which you are confident should perform reasonably well. It should be very simple to implement, and very easy to test, so that you can then test each of your improved ideas, and make sure they are always better than your baseline. Without starting with a sensible baseline, it is very difficult to know whether your super-fancy models are actually any good. One good approach to creating a baseline is doing what we have done here: think of a simple, easy-to-implement model. Another good approach is to search around to find other people that have solved similar problems to yours, and download and run their code on your dataset. Ideally, try both of these!
>
> 术语：基线：你相信有理由应该表现更好的一个简单模型。它应该很简单实施，并很容易测试，所以你能够稍后测试每一个你改进的想法，并确认他们总是比你的基线更好。不从一个实用基线开始，就很困难知晓你超级热爱的模型是不是真的好。创建一个基线的方法是做我们这里已经在做的事情：一个简单的想法，易于实施的模型。另一个好方法是查找周围去寻找其它人与你类似问题的现成解决方案，下载并在你的数据集上运行他们的代码。最合适的方法是，尝试这两个方法！

