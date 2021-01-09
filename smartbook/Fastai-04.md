

# Under the Hood: Training a Digit Classifier

# 追根溯源：训练一个数字分类器

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

Out：(#3) [Path('valid'),Path('labels.csv'),Path('train')]

The MNIST dataset follows a common layout for machine learning datasets: separate folders for the training set and the validation set (and/or test set). Let's see what's inside the training set:

MNIST数据集遵从了机器学习数据集的通用框架：把文件夹分割为训练集和验证集（和/或测试集）。让我们看一下训练集的内部是什么：

```python
(path/'train').ls()
```

Out: (#2) [Path('train/7'),Path('train/3')]

There's a folder of 3s, and a folder of 7s. In machine learning parlance, we say that "3" and "7" are the *labels* (or targets) in this dataset. Let's take a look in one of these folders (using `sorted` to ensure we all get the same order of files):

有一个3的文件夹和一个7的文件夹。在机器学习中的用语，我们说“3”和“7”是本数据集中的*标签*（或靶）。让我们看一下这些文件夹中的一个（使用*sorted*以确保我们完全取得相同顺序的文件）：

```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
```

Out: (#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]

As we might expect, it's full of image files. Let’s take a look at one now. Here’s an image of a handwritten number 3, taken from the famous MNIST dataset of handwritten numbers:

正如我们可能期望的，它全是图像文件。让我们现在看一张。这是一张来自著名MNIST数据集的手写数字3：

```python
im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```

Out: ![three_number](./_v_images/three_number.png)

Here we are using the `Image` class from the *Python Imaging Library* (PIL), which is the most widely used Python package for opening, manipulating, and viewing images. Jupyter knows about PIL images, so it displays the image for us automatically.

这里我们用了来自*python图像库*（PIL）的`Image`类，这是使用最为广泛的Python包用于打开、操作和查看图像。Jupyter知道PIL图片，所以它会为我们自动显示图片。

In a computer, everything is represented as a number. To view the numbers that make up this image, we have to convert it to a *NumPy array* or a *PyTorch tensor*. For instance, here's what a section of the image looks like, converted to a NumPy array:

在一个计算机里，所有的事情被表示为一个数值。查看组成这一图片的数值，我们必须把它转换为一个*NumPy数组*或一个*PyTorch张量*。例如，这是一张选中的图像看起来已转换为一个NumPy数组：

```python
array(im3)[4:10,4:10]
```

Out: $
\begin{matrix} array([&[& 0, & 0, & 0, & 0, & 0, & 0&],\\ 
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

Out: $
\begin{matrix} array([&[& 0, & 0, & 0, & 0, & 0, & 0&],\\ 
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

Out: <img src="./_v_images/att_00058.png" alt="att_00058" style="zoom:30%;"  />

You can see that the background white pixels are stored as the number 0, black is the number 255, and shades of gray are between the two. The entire image contains 28 pixels across and 28 pixels down, for a total of 784 pixels. (This is much smaller than an image that you would get from a phone camera, which has millions of pixels, but is a convenient size for our initial learning and experiments. We will build up to bigger, full-color images soon.)

你能够看到白色像素背景储存的数值是0，黑色的是数值255，及灰色阴影是在两者之间的数值。整张图像包含28个横向像素和28个竖向像素，一共是784个像素。（它与你从一个手机镜头拍摄的上百万像素图片极为相似，但对于我们开始学习和试验小像素是一个合适的尺寸。稍后我们会创建更大和彩色的图像。）

So, now you've seen what an image looks like to a computer, let's recall our goal: create a model that can recognize 3s and 7s. How might you go about getting a computer to do that?

所以，现在你已经看到了对于计算机来说一张图像的样子，来回想一下我们的目标：创建能够识别3和7的模型。你如何能够让计算机做这个事情呢？

> Warning: Stop and Think!: Before you read on, take a moment to think about how a computer might be able to recognize these two different digits. What kinds of features might it be able to look at? How might it be able to identify these features? How could it combine them together? Learning works best when you try to solve problems yourself, rather than just reading somebody else's answers; so step away from this book for a few minutes, grab a piece of paper and pen, and jot some ideas down…
>
> 警示：停下来并想一下：在你阅读之前，花点时间想一下计算机如何能够具备识别这两个不同数字的能力。它能够看什么类型的特征？它怎样能够识别这些特征？它怎么能够把他们结合在一起？当你尝试解决你自己的问题，而不只是阅读别人的答案的时候，学习效果最好。所以从本书移步离开几分钟，拿张纸和笔，并快速记下一些想法...

## First Try: Pixel Similarity

## 首要尝试：像素相似处

So, here is a first idea: how about we find the average pixel value for every pixel of the 3s, then do the same for the 7s. This will give us two group averages, defining what we might call the "ideal" 3 and 7. Then, to classify an image as one digit or the other, we see which of these two ideal digits the image is most similar to. This certainly seems like it should be better than nothing, so it will make a good baseline.

所以，这是第一个想法：我们如何发现３图像每个像素的平均像素值，然后对７的图像做同样的事情。这会给我们两组平均值，定义我们认为“理想中”的３和７。然后分类一张图像作为一个数字或另一个，我们看这两个理想的数字图片更像哪一个。这确实好像应该比无法识别任何东西更好，所以它会做作为一个好的基线。

> jargon: Baseline: A simple model which you are confident should perform reasonably well. It should be very simple to implement, and very easy to test, so that you can then test each of your improved ideas, and make sure they are always better than your baseline. Without starting with a sensible baseline, it is very difficult to know whether your super-fancy models are actually any good. One good approach to creating a baseline is doing what we have done here: think of a simple, easy-to-implement model. Another good approach is to search around to find other people that have solved similar problems to yours, and download and run their code on your dataset. Ideally, try both of these!
>
> 术语：基线：你相信有理由应该表现更好的一个简单模型。它应该很简单实施，并很容易测试，所以你能够稍后测试每一个你改进的想法，并确认他们总是比你的基线更好。不从一个实用基线开始，就很困难知晓你超级热爱的模型是不是真的好。创建一个基线的方法是做我们这里已经在做的事情：一个简单的想法，易于实施的模型。另一个好方法是查找周围去寻找其它人与你类似问题的现成解决方案，下载并在你的数据集上运行他们的代码。最合适的方法是，尝试这两个方法！

Step one for our simple model is to get the average of pixel values for each of our two groups. In the process of doing this, we will learn a lot of neat Python numeric programming tricks!

我们简易模型的第一步是获取我们两组图像中每一组的平均像素值。处理这个过程中，我们会学到一些灵巧的Python数据规划技巧！

Let's create a tensor containing all of our 3s stacked together. We already know how to create a tensor containing a single image. To create a tensor containing all the images in a directory, we will first use a Python list comprehension to create a plain list of the single image tensors.

让我们创建一个张量包含我们所有堆叠在一起的３图像。我们已经知道怎样创建一个包含单张图像的张量。创建一个包含所有在同一目录的图像张量，我们会第一次使用一个Python列表生成器创建一个单张图像张量的纯列表。

We will use Jupyter to do some little checks of our work along the way—in this case, making sure that the number of returned items seems reasonable:

运用这一方法，我们会使用Jupyter做一些工作上的小检查：在这个例子中，确保返回的数值项看起来是可理解的：

```python
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)
```

Out: (6131, 6265)

> note: List Comprehensions: List and dictionary comprehensions are a wonderful feature of Python. Many Python programmers use them every day, including the authors of this book—they are part of "idiomatic Python." But programmers coming from other languages may have never seen them before. There are a lot of great tutorials just a web search away, so we won't spend a long time discussing them now. Here is a quick explanation and example to get you started. A list comprehension looks like this: `new_list = [f(o) for o in a_list if o>0]`. This will return every element of `a_list` that is greater than 0, after passing it to the function `f`. There are three parts here: the collection you are iterating over (`a_list`), an optional filter (`if o>0`), and something to do to each element (`f(o)`). It's not only shorter to write but way faster than the alternative ways of creating the same list with a loop.
>
> 注解：列表生成器：列表和目录生成器是Python一个非常好的功能。许多Python程序员每天都会用它们，也包括本书的作者，他们是“Python惯用语”的一部分。但是来自其它语言的程序员之前可能多来没有看过他们。这里有很多只用网页所搜的极好的指引，所以现在我们不会花费太长时间讨论他们。这有一个快速解释和让我们开始的例子。一个列表生成器看起来像这样：`new_list = [f(o) for o in a_list if o>0]`。这会返回每一个`a_list`大于0的元素，之后把它传递给函数`f`。这里有三部分：收集你在（`a_list`）之上的迭代，一个操作过滤器（`if o>0`），和对每个元素进行处理的（`f(o)`）。它不仅仅编写短小，而且此方法相比使用循环创建相同列表的替代方法要更快。

We'll also check that one of the images looks okay. Since we now have tensors (which Jupyter by default will print as values), rather than PIL images (which Jupyter by default will display as images), we need to use fastai's `show_image` function to display it:

我们也会检查其中一张是否是好的。因为我们现在有了张量（Jupyter默认会输出为数值）而不是PIL图像（Jupyter默认会输出为一张图像），我们需要使用fastai的`show_image`函数去显示它：

```python
show_image(three_tensors[1]);
```

Out: <img src="./_v_images/three_2.png" alt="three_2" style="zoom:30%;" />

For every pixel position, we want to compute the average over all the images of the intensity of that pixel. To do this we first combine all the images in this list into a single three-dimensional tensor. The most common way to describe such a tensor is to call it a *rank-3 tensor*. We often need to stack up individual tensors in a collection into a single tensor. Unsurprisingly, PyTorch comes with a function called `stack` that we can use for this purpose.

对于每个像素位置，我们要去计算所以图像上的像素强度平均值。做本工作，我们首先要组合在这个列表中的所有图像进入一个单一三维张量。最常用的方法来描述这样一个张量是名为*三阶张量*。我们经常需要堆砌在一个集合里的各个独立张量进入一个单一张量。不用太惊讶，PyTorch提供了一个名为`stack`的函数，我们能用它来实现这一目的。

Some operations in PyTorch, such as taking a mean, require us to *cast* our integer types to float types. Since we'll be needing this later, we'll also cast our stacked tensor to `float` now. Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method.

在PyTorch中的一些操作，例如要取一个平均值，需要我们把整型*转换*为浮点型。因而我们稍后会需要这个，现在我们也将会转换堆叠后的张量为`float`。在PyTorch中转换是简单的输入你希望转换后的类型名，然后就做为一个方法处理它。

Generally when images are floats, the pixel values are expected to be between 0 and 1, so we will also divide by 255 here:

通常当一个图像是浮点型时，就希望像素值处于0到1之间，所以在这里我们也会除以255：

```python
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
```

Out: torch.Size([6131, 28, 28])

Perhaps the most important attribute of a tensor is its *shape*. This tells you the length of each axis. In this case, we can see that we have 6,131 images, each of size 28×28 pixels. There is nothing specifically about this tensor that says that the first axis is the number of images, the second is the height, and the third is the width—the semantics of a tensor are entirely up to us, and how we construct it. As far as PyTorch is concerned, it is just a bunch of numbers in memory.

也许张量最重要的属性是它的*shape*。这会告诉你每一个坐标轴的长度。在这个例子中，我们能够看到我们有6131张图像，每张图像的尺寸28×28像素。关于这个张量没有什么特别的只是说第一个坐标轴是图像的数值，第二个是图像的高，第三个是图像的宽：这样一个张量的含义就完全呈现给我们，及我们怎么构造他。正如PyTorch所考虑的，在内存中它只是一堆数。

The *length* of a tensor's shape is its rank:

张量形状的*长度*是它的阶：

```python
len(stacked_threes.shape)
```

Out: 3

It is really important for you to commit to memory and practice these bits of tensor jargon: *rank* is the number of axes or dimensions in a tensor; *shape* is the size of each axis of a tensor.

它对你是很重要的去记住和实践这些张量术语：*阶*是张量中的坐标数量或维度；*形状*是一个张量每个坐标轴的大小。

> A: Watch out because the term "dimension" is sometimes used in two ways. Consider that we live in "three-dimensonal space" where a physical position can be described by a 3-vector `v`. But according to PyTorch, the attribute `v.ndim` (which sure looks like the "number of dimensions" of `v`) equals one, not three! Why? Because `v` is a vector, which is a tensor of rank one, meaning that it has only one *axis* (even if that axis has a length of three). In other words, sometimes dimension is used for the size of an axis ("space is three-dimensional"); other times, it is used for the rank, or the number of axes ("a matrix has two dimensions"). When confused, I find it helpful to translate all statements into terms of rank, axis, and length, which are unambiguous terms.
>
> 亚：小心，因为术语“维度”有时候会用于两种方式。思考你所生活的“三维空间”，这是一个通过3维向量`v`能够描述的物理位置。但根据PyTorch，`v.ndim`的属性（确实看起来像`v`的"维度数"）等于1，而不是3！为什么？因为`v`是一个向量。即，是一个1阶的张量，意味它只有一个*坐标轴*（即使那个坐标轴有一个3的长度）。换句话说，有时候维度被用于坐标轴的大小（“空间是三维”）；其它时候，它被用于阶或坐标轴的数量（“一个矩阵有二维”）。当混淆的时候，我发现把所有的声明转换为阶、坐标轴和长度这些不模糊的术语是有帮助的。

We can also get a tensor's rank directly with `ndim`:

我们也能用`ndim`直接取得一个张量的阶：

```python
stacked_threes.ndim
```

Out: 3

Finally, we can compute what the ideal 3 looks like. We calculate the mean of all the image tensors by taking the mean along dimension 0 of our stacked, rank-3 tensor. This is the dimension that indexes over all the images.

最后，我们能够计算理想中的３像什么。我们计算全部图像张量的平均值，这个值是沿着我们所堆积张量的维度0取的平均值。这是在所有图像之上索引的维度。

In other words, for every pixel position, this will compute the average of that pixel over all images. The result will be one value for every pixel position, or a single image. Here it is:

换句话说，这将会计算所有图像像素之上的每一个像素位置的平均值。其结果是每个像素位置一个数值，或一张图像。它是这样的：

```python
mean3 = stacked_threes.mean(0)
show_image(mean3);
```

Out: <img src="./_v_images/three_3.png" alt="three_3" style="zoom:33%;" />

According to this dataset, this is the ideal number 3! (You may not like it, but this is what peak number 3 performance looks like.) You can see how it's very dark where all the images agree it should be dark, but it becomes wispy and blurry where the images disagree.

根据这个数据集，这是一个理想的数字3！（你可能不喜欢它，但是这看起来是最优的数字3的表现。）你能看到所有的图片一致认为它应该是暗的地方非常暗的，但对于图像不一致的地方就变成小束壮和模糊不清。

Let's do the same thing for the 7s, but put all the steps together at once to save some time:

让我们对7做同样的事情，但同时把所有的步骤合并起来以节省时间：

```
mean7 = stacked_sevens.mean(0)
show_image(mean7);
```

Out: <img src="./_v_images/seven.png" alt="seven" style="zoom:33%;" />

Let's now pick an arbitrary 3 and measure its *distance* from our "ideal digits."

让我们现在随意取一个3并测量我们“理想中数字”间的*差距*。

> stop: Stop and Think!: How would you calculate how similar a particular image is to each of our ideal digits? Remember to step away from this book and jot down some ideas before you move on! Research shows that recall and understanding improves dramatically when you are engaged with the learning process by solving problems, experimenting, and trying new ideas yourself
>
> 暂停：停下来并想一想！：你将如何计算每一个我们理想数字是如何相似一个特定图片？记住从本书离开一会并在你继续前写一下一些想法！研究显示当你通过解决问题、试验和尝试你自己的新想法来应对你的学习过程的时候，回忆能力和理解力会戏剧化的改善。

Here's a sample 3:

这是一个样本3：

```python
a_3 = stacked_threes[1]
show_image(a_3);
```

Out: <img src="./_v_images/three_4.png" alt="three_4" style="zoom:33%;" />

How can we determine its distance from our ideal 3? We can't just add up the differences between the pixels of this image and the ideal digit. Some differences will be positive while others will be negative, and these differences will cancel out, resulting in a situation where an image that is too dark in some places and too light in others might be shown as having zero total differences from the ideal. That would be misleading!

我们怎样能够确定它与我们理想中3的差距？我们不能只是加这张图片和理想中数字像素间的差异。一些差异是正向的同时其它一些会是负向的，这些差异将会抵消，在图像中一些地方太暗，在别一些地方太亮会被显示与理想数字总差异为零的一种结果情况。 这会被误导！

To avoid this, there are two main ways data scientists measure distance in this context:

为避免发现此事， 在这种情况下这里有两个数据科学家测量差距的主要方法：

- Take the mean of the *absolute value* of differences (absolute value is the function that replaces negative values with positive values). This is called the *mean absolute difference* or *L1 norm*
- Take the mean of the *square* of differences (which makes everything positive) and then take the *square root* (which undoes the squaring). This is called the *root mean squared error* (RMSE) or *L2 norm*.
- 取差异的*绝对值*的平均值（绝对值是替换负值为正值的一个方法）这称之为*平均绝对差*或*L1 正则*。
- 到差异的*平方*的平均值（使得每个数为正），然后求*平均根*（撤销平方）。这称之为*均方根误差*（RMSE）或*L2正则*

> important: It's Okay to Have Forgotten Your Math: In this book we generally assume that you have completed high school math, and remember at least some of it... But everybody forgets some things! It all depends on what you happen to have had reason to practice in the meantime. Perhaps you have forgotten what a *square root* is, or exactly how they work. No problem! Any time you come across a maths concept that is not explained fully in this book, don't just keep moving on; instead, stop and look it up. Make sure you understand the basic idea, how it works, and why we might be using it. One of the best places to refresh your understanding is Khan Academy. For instance, Khan Academy has a great [introduction to square roots](https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:rational-exponents-radicals/x2f8bb11595b61c86:radicals/v/understanding-square-roots).
>
> 重要：已经忘记你的数学知识这没什么关系：在本书我们通常假设你已经完成高中数学，并至少还记得一些... 但每个人都忘记了很多！在此期间它完全依赖于你碰巧有机会用到。也许你已经忘记什么是*平均根*或究竟他们怎么计算。没有问题！在本书任何时候你遇到了一个没有完全解释的数学概念，不要只是向下看，而是停下来并查找它。确保你理解基础概念，它的计算原理，及我们为什么可能会用到它。恢复你的理解最好的地方之一便是可汗学院。例如，可汗学院有一个很好的对[平均根的介绍](https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:rational-exponents-radicals/x2f8bb11595b61c86:radicals/v/understanding-square-roots)。

Let's try both of these now:

让我们现在尝试一下这两种方法：

```python
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr
```

Out: (tensor(0.1114), tensor(0.2021))

```python
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr
```

Out: (tensor(0.1586), tensor(0.3021))

In both cases, the distance between our 3 and the "ideal" 3 is less than the distance to the ideal 7. So our simple model will give the right prediction in this case.

在两个事例中，我们的3和“理想的”3之间的差距要小于它与理想的7之间的差距。所以在本例中我们的简单模型会给出正确的预测。

PyTorch already provides both of these as *loss functions*. You'll find these inside `torch.nn.functional`, which the PyTorch team recommends importing as `F` (and is available by default under that name in fastai):

PyTorch已经以*损失函数*的方式提供了这两个方法。你会在`torch.nn.functional`内部发现他们，PyTorch团队推荐以`F`作为引入（并且在fastai中默认该名可用）

```python
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```

Out: (tensor(0.1586), tensor(0.3021))

Here `mse` stands for *mean squared error*, and `l1` refers to the standard mathematical jargon for *mean absolute value* (in math it's called the *L1 norm*).

这里的`mse`代表*均方误差*，和`l1`参照的标准的数学术语*平均绝对值*（在数学里它称为*L1正则*）。

> S: Intuitively, the difference between L1 norm and mean squared error (MSE) is that the latter will penalize bigger mistakes more heavily than the former (and be more lenient with small mistakes).
>
> 西：直观地的来说，L1正则和均方差（MSE）之间的差异是后者相对前者会对更大的错误惩罚的更重（且对小错误惩罚更宽松）。

> J: When I first came across this "L1" thingie, I looked it up to see what on earth it meant. I found on Google that it is a *vector norm* using *absolute value*, so looked up *vector norm* and started reading: *Given a vector space V over a field F of the real or complex numbers, a norm on V is a nonnegative-valued any function p: V → [0,+∞) with the following properties: For all a ∈ F and all u, v ∈ V, p(u + v) ≤ p(u) + p(v)...* Then I stopped reading. "Ugh, I'll never understand math!" I thought, for the thousandth time. Since then I've learned that every time these complex mathy bits of jargon come up in practice, it turns out I can replace them with a tiny bit of code! Like, the *L1 loss* is just equal to `(a-b).abs().mean()`, where `a` and `b` are tensors. I guess mathy folks just think differently than me... I'll make sure in this book that every time some mathy jargon comes up, I'll give you the little bit of code it's equal to as well, and explain in common-sense terms what's going on.
>
> 杰：在我第一次碰到这个“L1”的时候，我查找看它到底代表什么意思。在谷歌上我发现它是一个使用*绝对值*的*向量范数*，所以查看*向量范数*并开始读：*在真实或复杂数据域 F 之上给出一个向量空间 V，在 V 上的一个范数是具有以下属性的非负数据任何函数 p: V → [0,+∞) 所有 a ∈ F 并且所有的 u, v ∈ V, p(u + v) ≤ p(u) + p(v)...* 然后我停止了阅读。我想了无数次，“嗯...，我永远都无法理解数学！” 自那以后我学会了，每当在实践中遇到这种复杂数学术语，证明我都能用一小段代码替换他们，就像L1损失只是等于 `(a-b).abs().mean()`，其中`a`和`b`是张量。我猜想搞数学的这些家伙只是想法上与我不同...在本书我保证每当遇到一些数学术语，我会给你与它相等的一小段代码，并用常识性语言解释到底是怎么回事。

We just completed various mathematical operations on PyTorch tensors. If you've done some numeric programming in PyTorch before, you may recognize these as being similar to NumPy arrays. Let's have a look at those two very important data structures.

我们刚刚完成了在PyTorch张量上各种数学操作。在PyTorch之前如果你已经完成一些数值编程，你应该可以看出这些与NumPy数组是相似的。让我们看一下这两者非常重要的数据结构。

### NumPy Arrays and PyTorch Tensors

### NumPy数组和PyTorch张量

[NumPy](https://numpy.org/) is the most widely used library for scientific and numeric programming in Python. It provides very similar functionality and a very similar API to that provided by PyTorch; however, it does not support using the GPU or calculating gradients, which are both critical for deep learning. Therefore, in this book we will generally use PyTorch tensors instead of NumPy arrays, where possible.

[NumPy](https://numpy.org/)是使用作为广泛的科学与数值编程库。它提供的功能与API与PyTorch所提供的非常相似。然而，它不支持对深度学习很重要的两个事情：GPU或计算剃度。因而，在本书，我们会广泛使用PyTorch张量尽可能的来替代NumPy数组。

(Note that fastai adds some features to NumPy and PyTorch to make them a bit more similar to each other. If any code in this book doesn't work on your computer, it's possible that you forgot to include a line like this at the start of your notebook: `from fastai.vision.all import *`.)

（注解：fastai增加了一些NumPy和PyTorch特性，以使得它们彼此间有点类似。如果在本收中的任何代码无法在你本地运行，可能你在笔记开头忘记包含像这样的一行代码：`from fastai.vision.all import *`。）

But what are arrays and tensors, and why should you care?

但数组和张量是什么，为什么你应该关注？

Python is slow compared to many languages. Anything fast in Python, NumPy, or PyTorch is likely to be a wrapper for a compiled object written (and optimized) in another language—specifically C. In fact, **NumPy arrays and PyTorch tensors can finish computations many thousands of times faster than using pure Python.**

相比项目语言Python是慢的。在Python、NumPy、或PyTorch中任何运行速度快的可能在其它语言（特别是C）中编写（和优化），对编译对象做了包装。事实上，**Numpy数组和PyTorch张量能够完成计算比单纯使用Python快数千倍**。

A NumPy array is a multidimensional table of data, with all items of the same type. Since that can be any type at all, they can even be arrays of arrays, with the innermost arrays potentially being different sizes—this is called a "jagged array." By "multidimensional table" we mean, for instance, a list (dimension of one), a table or matrix (dimension of two), a "table of tables" or "cube" (dimension of three), and so forth. If the items are all of some simple type such as integer or float, then NumPy will store them as a compact C data structure in memory. This is where NumPy shines. NumPy has a wide variety of operators and methods that can run computations on these compact structures at the same speed as optimized C, because they are written in optimized C.

一个NumPy数组是所有条目类型相同的多维数据表。因为能完全做为任何类型，他们甚至能做为数组的数组，数组最里面可能是不同的尺寸，这被称为“不规则数组”。我们表达的“多维的表”，例如是一列（一维），一个表或矩阵（二维），一个“表中的表”或“立方体”（三维）等等。如果所有的条目是一些整型或浮点型之类简单的类型，NumPy会在内存中以简洁的C数据结构存贮他们。这是NumPy的闪光点。NumPy有多种运算符和方法在这些简洁的结构上运行计算，运行速度与优化的C相同，因为他们的优化是用C写的。

A PyTorch tensor is nearly the same thing as a NumPy array, but with an additional restriction that unlocks some additional capabilities. It's the same in that it, too, is a multidimensional table of data, with all items of the same type. However, the restriction is that a tensor cannot use just any old type—it has to use a single basic numeric type for all components. For example, a PyTorch tensor cannot be jagged. It is always a regularly shaped multidimensional rectangular structure.

一个PyTorch张量做的事情与NumPy数组几乎相同，但一些附加限制打开了一些额外的能力。它同样是一个所有条目类型相同的多维数据表。然而，限制是一个张量只是不能使用任何老的类型（对所有的组件它必须用一个单一基础数值类型）。例如，一个PyTorch张量不能是不规则的。它总是一个规则多维长方形结构形状。

The vast majority of methods and operators supported by NumPy on these structures are also supported by PyTorch, but PyTorch tensors have additional capabilities. One major capability is that these structures can live on the GPU, in which case their computation will be optimized for the GPU and can run much faster (given lots of values to work on). In addition, PyTorch can automatically calculate derivatives of these operations, including combinations of operations. As you'll see, it would be impossible to do deep learning in practice without this capability.

在这些结构上由NumPy提供的的绝大多数方法和运算符也被PyTorch所提供，但PyTorch张量有额外的能力。一个主要能力是这些结构能够在GPU上运行，由此他们的计算能够针对GPU做优化并运行的更快（给出很多的数据去处理）。另外，PyTorch能够自动计算这些运算符的派生，包括运算符的混合体。你将会看到，它没有个能力它也许不可能在实践中做深度学习。

> S: If you don't know what C is, don't worry as you won't need it at all. In a nutshell, it's a low-level (low-level means more similar to the language that computers use internally) language that is very fast compared to Python. To take advantage of its speed while programming in Python, try to avoid as much as possible writing loops, and replace them by commands that work directly on arrays or tensors.
>
> 西：如果你不知道C是什么不要着急，因为你压根不需要它。简单的说，相对Python它是运行的非常快的底层语言（低层的意思是更像是计算机内部使用的语言）。当在Python中编程时它会取得运行速度优势，尝试避免尽可能多的写循环，而是通过直接工作在数组或张量上的命令来替代他们。

Perhaps the most important new coding skill for a Python programmer to learn is how to effectively use the array/tensor APIs. We will be showing lots more tricks later in this book, but here's a summary of the key things you need to know for now.

可能对于一名Python程序员最重要的新编码技能是去学习如何有效使用数组/张量API接口。稍后在本书我们会展示一些技巧，但这里是你现在需要知道的一些关键事项的总结。

To create an array or tensor, pass a list (or list of lists, or list of lists of lists, etc.) to `array()` or `tensor()`:

创建一个数组和张量，传递一个列表（或列表的列表，或列表的列表的列表等。）给`array()` 或 `tensor()`：

```python
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```

```python
arr  # numpy
```

Out: $
\begin{matrix} array([&[& 1, & 2, & 3&],\\ 
	& [&4,& 5,& 6&]]&)
\end{matrix}
$

```python
tns  # pytorch
```

Out: $
\begin{matrix} tensor([&[& 1, & 2, & 3&],\\ 
	& [&4,& 5,& 6&]]&)
\end{matrix}
$

All the operations that follow are shown on tensors, but the syntax and results for NumPy arrays is identical.

随后所有的操作在张量上展示，但是句法和结果对于NumPy数组来说是相同的。

You can select a row (note that, like lists in Python, tensors are 0-indexed so 1 refers to the second row/column):

你能选择一行（请注意，就像Python中的列表，张量是从0开始索引的，所以1指的是第二行/列）：

```python
tns[1]
```

Out: tensor([4, 5, 6])

or a column, by using `:` to indicate *all of the first axis* (we sometimes refer to the dimensions of tensors/arrays as *axes*):

或一列，通过使用`:` 来指示*第一坐标轴的全部*（我们有时候参照张量/数组的维度为*坐标轴*）：

```python
tns[:,1]
```

Out: tensor([2, 5])

You can combine these with Python slice syntax (`[start:end]` with `end` being excluded) to select part of a row or column:

你能够把这些内容与Python切片句法（`[start:end]`其中`end`是被排除在外的）结合使用去选择一行或列的部分内容：

```python
tns[1,1:3]
```

Out: tensor([5, 6])

And you can use the standard operators such as `+`, `-`, `*`, `/`:

并且你能用例如`+`,`-`,`*`,`/`这些标准运算符：

```python
tns+1
```
Out: $
\begin{matrix} tensor([&[& 2, & 3, & 4&],\\ 
	& [&5,& 6,& 7&]]&)
\end{matrix}
$

Tensors have a type:

张量有一个类型：

```python
tns.type()
```

Out: 'torch.LongTensor'

And will automatically change type as needed, for example from `int` to `float`:

并且会自动改变为所需要的类型，例如从`整型`变为`浮点型`：

```python
tns*1.5
```
Out: $
\begin{matrix} tensor([&[& 1.5000, & 3.0000, & 4.5000&],\\ 
	& [& 6.0000,& 7.5000 ,& 9.0000&]]&)
\end{matrix}
$

So, is our baseline model any good? To quantify this, we must define a metric.

所以，我们的基线模型好吗？要量化，我们必须定义一个指标。

## Computing Metrics Using Broadcasting

## 使用广播计算指标

Recall that a metric is a number that is calculated based on the predictions of our model, and the correct labels in our dataset, in order to tell us how good our model is. For instance, we could use either of the functions we saw in the previous section, mean squared error, or mean absolute error, and take the average of them over the whole dataset. However, neither of these are numbers that are very understandable to most people; in practice, we normally use *accuracy* as the metric for classification models.

回忆一下，一个指标是基于我们模型预测和在我们数据集中的正确标注计算出的一个数字，为了能够告诉我们的模型如何的好。例如，我们能够用在之前小节看到的两种函数：均方误差或平均绝对误差。并取整个数据集的平均数。然而，对大多数人来说这些数字都不是非常不易懂。在实践中，我们通常会用*精度*作为分类模型的指标。

As we've discussed, we want to calculate our metric over a *validation set*. This is so that we don't inadvertently overfit—that is, train a model to work well only on our training data. This is not really a risk with the pixel similarity model we're using here as a first try, since it has no trained components, but we'll use a validation set anyway to follow normal practices and to be ready for our second try later.

正如我们讨论过的，我们希望在一个*验证集*上计算我们的指标。这是因为我们不要无意中过拟（它是训练一个模型只在我们的训练数据上工作的好）。作为第一次尝试，我们在这里使用的像素相似模型，这不是一个真实风险，因为它没有被训练的要素，但不论如何，我们会用一个验证集遵循正常的实践，并为我们稍后的第二次尝试做好准备。

To get a validation set we need to remove some of the data from training entirely, so it is not seen by the model at all. As it turns out, the creators of the MNIST dataset have already done this for us. Do you remember how there was a whole separate directory called *valid*? That's what this directory is for!

我们需要从整个训练集中移除一些数据来得到一个验证集，所以它对模型是完全不可见的。事实证明，MNIST数据集的创建者已经为我们做了这个事情。你还记得有一个完全分割的目录称为*验证*吗？那就是这个目录的使用！

So to start with, let's create tensors for our 3s and 7s from that directory. These are the tensors we will use to calculate a metric measuring the quality of our first-try model, which measures distance from an ideal image:

所以让我们为来自那个目录的3和7创建张量开始。这些张量我们会用于计算一个指标来测量我们第一步尝试模型的质量，即测量与理想中图片的差距：

```python
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape
```

Out: (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))

It's good to get in the habit of checking shapes as you go. Here we see two tensors, one representing the 3s validation set of 1,010 images of size 28×28, and one representing the 7s validation set of 1,028 images of size 28×28.

养成随时检查形状的习惯这很好。这里我们看到两个张量，一个代表3的验证集（1010张图像，尺寸28×28），一个代表7的验证集（1028张图像，尺寸28×28）。

We ultimately want to write a function, `is_3`, that will decide if an arbitrary image is a 3 or a 7. It will do this by deciding which of our two "ideal digits" this arbitrary image is closer to. For that we need to define a notion of distance—that is, a function that calculates the distance between two images.

我们最终会写一个`is_3`的函数，它将会判断一个随机图像是3或7。它将决策我们两个“理想中的数字”那一个与这张随机图像是更接近。为了这个，我们需要定义一个差距概念：那是一个计算两张图像差距的函数。

We can write a simple function that calculates the mean absolute error using an experssion very similar to the one we wrote in the last section:

我们能写一个计算平均绝对误差的简单函数，使用的表达与我们在最后小节写的那个非常相似：

```python
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
```

Out: tensor(0.1114)

This is the same value we previously calculated for the distance between these two images, the ideal 3 `mean3` and the arbitrary sample 3 `a_3`, which are both single-image tensors with a shape of `[28,28]`.

这与我们之前计算的两张图像间的差距值是相同的。这两张图像是理想中的3 `mean3`和随机样本3 `a_3`,这两者都是`[28,28]`形状的单图像张量。

But in order to calculate a metric for overall accuracy, we will need to calculate the distance to the ideal 3 for *every* image in the validation set. How do we do that calculation? We could write a loop over all of the single-image tensors that are stacked within our validation set tensor, `valid_3_tens`, which has a shape of `[1010,28,28]` representing 1,010 images. But there is a better way.

但为了计算一个全局精度指标，我将需要来计算理想中的3对验证集中*每张*图像的差距。我们怎么做这个计算呢？我们能够写一个循环，把所有单张图像张量堆叠在我们的验证集张量，`valid_3_tens`，这个张量的形状为`[1010,28,28]`代表1010张图像。但这里有一个更好的方法。

Something very interesting happens when we take this exact same distance function, designed for comparing two single images, but pass in as an argument `valid_3_tens`, the tensor that represents the 3s validation set:

当你们采纳这个完全一样的距离函数时，会发生一些有趣的事情，这个函数的设计用于比较两张单张图像，`valid_3_lens`做为一个参数传递进来后，张量就相当于3的验证集：

```python
valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```

Out: (tensor([0.1634, 0.1145, 0.1363,  ..., 0.1105, 0.1111, 0.1640]),torch.Size([1010]))

Instead of complaining about shapes not matching, it returned the distance for every single image as a vector (i.e., a rank-1 tensor) of length 1,010 (the number of 3s in our validation set). How did that happen?

替代吐槽形状不匹配，它返回了长度1010（在我们验证集中3图像的数量）每个单张图像的差距的向量（即，一阶张量）。这是怎么发生的？

Take another look at our function `mnist_distance`, and you'll see we have there the subtraction `(a-b)`. The magic trick is that PyTorch, when it tries to perform a simple subtraction operation between two tensors of different ranks, will use *broadcasting*. That is, it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. Broadcasting is an important capability that makes tensor code much easier to write.

再看一下我们的`mnist_distance`函数，你会看在这里我们有减法`(a-b)`。那是PyTorch奇妙的技巧，当它尝试在两个不同阶张量间执行减法操作时，将用到*广播*。这是自动扩展张量到与最大阶张量相同的阶，从而他们间具有了相同尺寸。广播是使得张量代码更容易编写的一个很重要的能力。

After broadcasting so the two argument tensors have the same rank, PyTorch applies its usual logic for two tensors of the same rank: it performs the operation on each corresponding element of the two tensors, and returns the tensor result. For instance:

广播后两个张量参数有了相同的阶，PyTorch对两个相同阶的张量应用通常逻辑：它在两个张量的每个相符元素上执行操作，并返回张量结果。例如：

```python
tensor([1,2,3]) + tensor([1,1,1])
```

Out: tensor([2, 3, 4])

So in this case, PyTorch treats `mean3`, a rank-2 tensor representing a single image, as if it were 1,010 copies of the same image, and then subtracts each of those copies from each 3 in our validation set. What shape would you expect this tensor to have? Try to figure it out yourself before you look at the answer below:

所以在这个例子中，PyTorch处理`mean3`（一个两阶张量等同于一个单张图像）好像它是相同图像的1010次拷贝，然后这些拷贝与我们验证集中每一张图像3做减法操作。你期望这个张量有什么样的形状？在看到下面的答案之前尝试自己想像出它的结果：

```python
(valid_3_tens-mean3).shape
```

Out: torch.Size([1010, 28, 28])

We are calculating the difference between our "ideal 3" and each of the 1,010 3s in the validation set, for each of 28×28 images, resulting in the shape `[1010,28,28]`.

我们正在计算我们“理想中的3”和在验证集中1010张图像每张3之间的差异，每张图像的尺寸是 28×28 ，得到结果形状为`[1010,28,28]`。

There are a couple of important points about how broadcasting is implemented, which make it valuable not just for expressivity but also for performance:

关于广播的实施这里有两个重要的点，每个产生的价值不仅仅在其表达性上，而且有其执行：

- PyTorch doesn't *actually* copy `mean3` 1,010 times. It *pretends* it were a tensor of that shape, but doesn't actually allocate any additional memory
- It does the whole calculation in C (or, if you're using a GPU, in CUDA, the equivalent of C on the GPU), tens of thousands of times faster than pure Python (up to millions of times faster on a GPU!).
- PyTorch不*实际* 拷贝`mean3`1010次。它*假装*是那个形状的一个张量，但是并没有实际分配任何额外内存
- 它在C中做整个计算（或如果你使用一个GPU，在CUDA中等同于在GPU上的C），比纯Python快几万倍（在GPU上要快百万倍以上）。

This is true of all broadcasting and elementwise operations and functions done in PyTorch. *It's the most important technique for you to know to create efficient PyTorch code.*

在PyTorch中广播和逐元素操作及函数处理都是如此。*对于你知道如何创建高效的PyTorch代码它是最重要的*。

Next in `mnist_distance` we see `abs`. You might be able to guess now what this does when applied to a tensor. It applies the method to each individual element in the tensor, and returns a tensor of the results (that is, it applies the method "elementwise"). So in this case, we'll get back 1,010 absolute values.

接下来我们看`mnist_distance`中的`abs`。当它被应用于一个张量时，你现在也许能够猜这是做什么的。它应用本方法到张量中的每个独立元素，并返回结果的一个张量（也就是说，它按“逐元素”应用本方法）。所以在这个例子中，我们会返回1010个绝对值。

Finally, our function calls `mean((-1,-2))`. The tuple `(-1,-2)` represents a range of axes. In Python, `-1` refers to the last element, and `-2` refers to the second-to-last. So in this case, this tells PyTorch that we want to take the mean ranging over the values indexed by the last two axes of the tensor. The last two axes are the horizontal and vertical dimensions of an image. After taking the mean over the last two axes, we are left with just the first tensor axis, which indexes over our images, which is why our final size was `(1010)`. In other words, for every image, we averaged the intensity of all the pixels in that image.

最后，我们的函数名叫`mean((-1,-2))`。这个`(-1,-2)`元组代表坐标轴的一个范围。在Python中，-1参照的是最后一个元素，-2参照的是倒数第二个元素。所以在这个例子中，这是告诉PyTorch我们想在张量的最后两个坐标轴索引的数值范围上求平均。最后两个坐标轴是一个图像的横和纵两个维度。在最后两个坐标轴上求平均值后，我们就剩下覆盖我们所有图像索引的第一个张量坐标轴，这主是为什么我们最终的尺寸是`(1010)`。换句话说，对每一张图片，我们对在那张图像上的所有像素强度就了平均。

We'll be learning lots more about broadcasting throughout this book, especially in <<chapter_foundations>, and will be practicing it regularly too.

通过本书，我们将会学习到一些关于广播的内容，特别在<章节：基础>中会也会经常的对它进行练习。

We can use `mnist_distance` to figure out whether an image is a 3 or not by using the following logic: if the distance between the digit in question and the ideal 3 is less than the distance to the ideal 7, then it's a 3. This function will automatically do broadcasting and be applied elementwise, just like all PyTorch functions and operators:

我们能够用`mnist_distance`通过使用下述逻辑想像出一个图像是否是3：在问题图片和理想中3之间的差距小于理想中7的差距，它就是3.这个函数将会自动做广播并应用到逐个元素，就像所有的PyTorch函数和操作一样：

```python
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
```

Let's test it on our example case:

让我们在我们的事例上测试一下它：

```python
is_3(a_3), is_3(a_3).float()
```

Out: (tensor(True), tensor(1.))

Note that when we convert the Boolean response to a float, we get `1.0` for `True` and `0.0` for `False`. Thanks to broadcasting, we can also test it on the full validation set of 3s:

注意，当我们转换布尔型为一个浮点型，我们取`1.0`为`真`和`0.0`为`假`。感谢广播机制，我们也能够在全是3的验证集上测试它：

```python
is_3(valid_3_tens)
```

Out: tensor([True, True, True,  ..., True, True, True])

Now we can calculate the accuracy for each of the 3s and 7s by taking the average of that function for all 3s and its inverse for all 7s:

现在我们能够通过对所有3的函数求平均值和对所有7的反函数求平均值，来计算每个3和7的精度：

```python
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
```

Out: (tensor(0.9168), tensor(0.9854), tensor(0.9511))

This looks like a pretty good start! We're getting over 90% accuracy on both 3s and 7s, and we've seen how to define a metric conveniently using broadcasting.

这看起来是一个相当好的开始！我们对3和7取得了超过90%的精度，并且我们已经看到如何方便的利用传广播来定义一个指标。

But let's be honest: 3s and 7s are very different-looking digits. And we're only classifying 2 out of the 10 possible digits so far. So we're going to need to do better!

但说实话，3和7看起来是差异非常大的数字。并且到目前为止我们只分类出10个可能数字中的2个。所以我们将需要做的更好！

To do better, perhaps it is time to try a system that does some real learning—that is, that can automatically modify itself to improve its performance. In other words, it's time to talk about the training process, and SGD.

为了做的更好，也许是时候尝试一个做真正学习的系统了。也就是说，能够自己改变它自己来改善它的表现。换句话话，是时候讨论关于训练过程和随机梯度下降（SGD）的内容了

## Stochastic Gradient Descent (SGD)

## 随机梯度下降（SGD）

Do you remember the way that Arthur Samuel described machine learning, which we quoted in <chapter_intro>?

你还记得亚瑟·塞缪尔描述的机器学习方法吗？我们在<章节：概述>中引用的话。

> : Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.
>
> ：假设我们安排了一些基于实际的表现测试当前权重分配有效性的方法，并提供一种改变权重分配的机制以使其表现最优。我们不需要进入这个程序的细节就能到看它能够实现完全自动化，并看到这样程序的机器会从它的经验中“学习”。

As we discussed, this is the key to allowing us to have a model that can get better and better—that can learn. But our pixel similarity approach does not really do this. We do not have any kind of weight assignment, or any way of improving based on testing the effectiveness of a weight assignment. In other words, we can't really improve our pixel similarity approach by modifying a set of parameters. In order to take advantage of the power of deep learning, we will first have to represent our task in the way that Arthur Samuel described it.

正如我们讨论过的，这是一个允许我们有一个越来越好模型的关键：它能学习。但我们的像素相似法不能真正的做这个事情。我们没有任何种类的权重分配，或任何基于测试权重分配的有效性的改善方法。换种说法，我们不能通过改变一组参数，真正的改善我们的像素相似法。为了充分利用深度深度的力量，我们将首先必须用亚瑟·塞缪尔描述的方法来体现我们的任务。

Instead of trying to find the similarity between an image and an "ideal image," we could instead look at each individual pixel and come up with a set of weights for each one, such that the highest weights are associated with those pixels most likely to be black for a particular category. For instance, pixels toward the bottom right are not very likely to be activated for a 7, so they should have a low weight for a 7, but they are likely to be activated for an 8, so they should have a high weight for an 8. This can be represented as a function and set of weight values for each possible category—for instance the probability of being the number 8:

替代尝试寻找一个张图像与一个“理想中的图像”之间相似之处，相反，我们能够查看每个独立的像素并为对每个像素提供一组权重，这样对于一个特定分类那些最有可能是黑色的像素被分配了最高的权重。例如对于7，右下角像素是极不可能被激活的。所以对于7来说他们应该有一个低权重，但是对于8，他们可能被激活，所以对8来说他们应该有一个高权重。对于每一个可能分类这能够表示为一个函数和权重值集，例如做为数字8的概率：

​			def pr_eight(x,w) = (x*w).sum()

Here we are assuming that `x` is the image, represented as a vector—in other words, with all of the rows stacked up end to end into a single long line. And we are assuming that the weights are a vector `w`. If we have this function, then we just need some way to update the weights to make them a little bit better. With such an approach, we can repeat that step a number of times, making the weights better and better, until they are as good as we can make them.

这是我们正在假设`x`是图像，表示为一个向量：换句话说，把所有的行端对端的堆叠为一个很长的单行。并且我们正在假设权重是一个向量`w`。如果我们有这样的函数，然后我们只需要一些方法来更新权重，以使得它们稍好一点。用这样的方法，我们能够重复这个步骤很多次，使得权重越来越好，直到他们与我们所能够做到的一样好。

We want to find the specific values for the vector `w` that causes the result of our function to be high for those images that are actually 8s, and low for those images that are not. Searching for the best vector `w` is a way to search for the best function for recognising 8s. (Because we are not yet using a deep neural network, we are limited by what our function can actually do—we are going to fix that constraint later in this chapter.)

对于向量`w`我们希望找到特点值，使得我们的函数对于那些真实为8的图像是高的，且对于那些不是8的图像是低的。搜索最好的向量`w`是一个去搜索识别8的最好函数的方法。（因为我们还没使用深度神经网络，我们被我们函数所能实际做的事情限制了，我们将在本章修复这个限制。）

To be more specific, here are the steps that we are going to require, to turn this function into a machine learning classifier:

为了更加具体，这里是我们将需要的一些步骤，把这个函数变为一个机器学习分类器：

1. *Initialize* the weights.
2. For each image, use these weights to *predict* whether it appears to be a 3 or a 7.
3. Based on these predictions, calculate how good the model is (its *loss*).
4. Calculate the *gradient*, which measures for each weight, how changing that weight would change the loss
5. *Step* (that is, change) all the weights based on that calculation.
6. Go back to the step 2, and *repeat* the process.
7. Iterate until you decide to *stop* the training process (for instance, because the model is good enough or you don't want to wait any longer).



1. *初始化*权重。
2. 对每张图像使用权重来*预测* 它像是3或7。
3. 基于这些预测，计算模型是如何的好（它的损失）。
4. 计算用于测量每个权重*梯度*，改变权重可能会改变损失。
5. 根据这个计算*步进*（即改变）所有权重。
6. 返回到步骤2，并重复这一过程。
7. 重复，直到你决定*停止*这个训练过程（例如，因为这个模型足够的好了或你不想再等了）。

These seven steps, illustrated in <gradient_descent>, are the key to the training of all deep learning models. That deep learning turns out to rely entirely on these steps is extremely surprising and counterintuitive. It's amazing that this process can solve such complex problems. But, as you'll see, it really does!

这七步（在<梯度下降>图中做了插图说明）是训练所有深度模型的关键。深度学习产生完全的依赖这些步骤是极度令人惊讶和反直觉的。这个过程能够解决如此复杂的问题是奇妙的。但，正好你将看到的，它确实如此！

```mermaid
graph LR
init((init))
predict((predict))
loss((loss))
gradient((gradient))
step((step))
stop((stop))
subgraph the_gradient_descent_process
init --> predict --> loss --> gradient  --> step --> stop 
end
subgraph the_gradient_descent_process
step --repeat--> predict
end
```



```mermaid
graph LR
init((初始化))
predict((预测))
loss((损失))
gradient((梯度))
step((步进))
stop((停止))
subgraph 梯度下降过程
init --> predict --> loss --> gradient  --> step --> stop 
end
subgraph 梯度下降过程
step --重复--> predict
end
```

There are many different ways to do each of these seven steps, and we will be learning about them throughout the rest of this book. These are the details that make a big difference for deep learning practitioners, but it turns out that the general approach to each one generally follows some basic principles. Here are a few guidelines:

有很多不同方法来做这七步中的第一步，我将在本书的其它部分学习他们。对于深度学习从业人员来说有些会带来很大差别的细节，但事实证明任何一些通用方法通常会遵循一些基本原则。这是一些指导方针：

- Initialize:: We initialize the parameters to random values. This may sound surprising. There are certainly other choices we could make, such as initializing them to the percentage of times that pixel is activated for that category—but since we already know that we have a routine to improve these weights, it turns out that just starting with random weights works perfectly well.
- Loss:: This is what Samuel referred to when he spoke of *testing the effectiveness of any current weight assignment in terms of actual performance*. We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).
- Step:: A simple way to figure out whether a weight should be increased a bit, or decreased a bit, would be just to try it: increase the weight by a small amount, and see if the loss goes up or down. Once you find the correct direction, you could then change that amount by a bit more, and a bit less, until you find an amount that works well. However, this is slow! As we will see, the magic of calculus allows us to directly figure out in which direction, and by roughly how much, to change each weight, without having to try all these small changes. The way to do this is by calculating *gradients*. This is just a performance optimization, we would get exactly the same results by using the slower manual process as well.
- Stop:: Once we've decided how many epochs to train the model for (a few suggestions for this were given in the earlier list), we apply that decision. This is where that decision is applied. For our digit classifier, we would keep training until the accuracy of the model started getting worse, or we ran out of time.
- 初始化：我们用随机值来初始化参数。这可能听起来让人吃惊。我们确实可做其它选择，例如用这个分类像素被激活次数的百分比来初始化他们。但是因为我们已经知道，我们有一个例行程序来改善这些权重，事实证明，只从随机权重开始工作就十分有效。
- 损失：这是塞缪尔提出的，他谈到*依据实际表现测试当前权重分配的有效性*，我们需要一些函数，如果模型是好的会返回一个小数值（标准方法是认为一个小的损失认为是好，而一个大的损失认为是坏，虽然这仅仅是一个惯例）。
- 步进：一个简单的方法是算出一个权重应该被增加一点，还是被减少一点，也许只需要尝试一下：通过一个小数值增加权重，并看损失增加还是减小。一旦你发现正确的方向，然后你能够通过增加一点和减小一点来改变这个数值，直到你发现一个数值效果很好。然而，这太慢了！正如我们将要看不对劲的，计算的魔力允许我们直接计算出方向，及大致是多少，来改变每个权重，不用尝试这些小的改变。这个方法是通过计算*梯度*来做这个事情。通过使用更慢的手工过程也能取得相同精确的结果，这只是一个表现优化。
- 停止：一旦我们已经决定用多少周期来训练模型（对于此事在很早的列表中已经出给出了一些小建议），我们就应用这个决策。这就是决策被应用的地方。对于我们的数字分类，模型的精度开始变的糟糕或我们没有运行时间了，我们将会停止训练。

Before applying these steps to our image classification problem, let's illustrate what they look like in a simpler case. First we will define a very simple function, the quadratic—let's pretend that this is our loss function, and `x` is a weight parameter of the function:

应用这些步骤到我们图像分类问题之前，让我们在一个很简单的例子中插图说明他们什么样子。首先我们要定义一个非常简单的函数：二次方程。让我们假定这是我们的损失函数，并且`x`是这个函数的权重：

```python
def f(x): return x**2
```

Here is a graph of that function:

这是该函数的图形：

```python
plot_function(f, 'x', 'x**2')
```

Out: <img src="./_v_images/quadratic_graph.png" alt="quadratic_graph" style="zoom:100%;" />

The sequence of steps we described earlier starts by picking some random value for a parameter, and calculating the value of the loss:

在早先我们描述的一系列步骤从为参数取随机值开始，并计算损失值：

```python
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');
```

Out: <img src="./_v_images/quadraticplot.png" alt="quadraticplot" style="zoom:100%;" />

Now we look to see what would happen if we increased or decreased our parameter by a little bit—the *adjustment*. This is simply the slope at a particular point:

现在我们如果通过小小的*调整*，来增加或减小我们参数看看会发生什么。这是一个在特定点上简单的斜坡：

<img src="./_v_images/grad_illustration.svg" alt="grad_illustration" style="zoom:110%;" />

We can change our weight by a little in the direction of the slope, calculate our loss and adjustment again, and repeat this a few times. Eventually, we will get to the lowest point on our curve:

我们能够在斜坡上的方向通过少许调整改变我们的权重，计算我们的损失和再次调整，并重复这一过程几次，我们会在曲线上达到最低点：

<img src="/Users/Y.H/Documents/GitHub/smartbook/smartbook/_v_images/chapter2_perfect.svg" alt="chapter2_perfect" style="zoom:100%;" />

This basic idea goes all the way back to Isaac Newton, who pointed out that we can optimize arbitrary functions in this way. Regardless of how complicated our functions become, this basic approach of gradient descent will not significantly change. The only minor changes we will see later in this book are some handy ways we can make it faster, by finding better steps.

这个基本想法完全来自艾萨克·牛顿，它指出我们能够用这种方法优化任意函数。不管我们的函数变得多么复杂，这个梯度下降的基本方法都不会有明显改变。在本书中稍晚我们会看到仅有的一点改变，是一些通过寻找更优的步骤能让它更快的实用方法，

### Calculating Gradients

### 计算梯度

The one magic step is the bit where we calculate the gradients. As we mentioned, we use calculus as a performance optimization; it allows us to more quickly calculate whether our loss will go up or down when we adjust our parameters up or down. In other words, the gradients will tell us how much we have to change each weight to make our model better.

一个神奇的步骤是我们计算梯度。正如我们提到的，我们使用微积分进行性能优化。当我们上调或上调我们的参数时，它允许我们更快的计算我们损失会上升还是下降。换句话说，梯度会告诉我们我们必须对每个权重调整多少以使得我们的模型更好。

You may remember from your high school calculus class that the *derivative* of a function tells you how much a change in its parameterss will change its result. If not, don't worry, lots of us forget calculus once high school is behind us! But you will have to have some intuitive understanding of what a derivative is before you continue, so if this is all very fuzzy in your head, head over to Khan Academy and complete the [lessons on basic derivatives](https://www.khanacademy.org/math/differential-calculus/dc-diff-intro). You won't have to know how to calculate them yourselves, you just have to know what a derivative is.

你可能记得你的高中微积分课程，一个函数的*导数*告诉你在它们的参数上调整多少会改变它的结果。如果不记的了，不要着急。我们大多数人一旦高中毕业就忘记微积分了！但在你继续之前，你将必须对导数是什么有一些直觉上的理解，所以如果在你的脑子里它非常模糊不清，前往可汗学院并完成[基础导数课程](https://www.khanacademy.org/math/differential-calculus/dc-diff-intro)。你不必知道如何自己计算他们，你只须知道一个导数是什么。

The key point about a derivative is this: for any function, such as the quadratic function we saw in the previous section, we can calculate its derivative. The derivative is another function. It calculates the change, rather than the value. For instance, the derivative of the quadratic function at the value 3 tells us how rapidly the function changes at the value 3. More specifically, you may recall that gradient is defined as *rise/run*, that is, the change in the value of the function, divided by the change in the value of the parameter. When we know how our function will change, then we know what we need to do to make it smaller. This is the key to machine learning: having a way to change the parameters of a function to make it smaller. Calculus provides us with a computational shortcut, the derivative, which lets us directly calculate the gradients of our functions.

关于导数的关键点是：对于任何函数，例如在之前部分我们看到的二次方程函数，我们能够计算它的导数。导数是另一个函数。它计算变化而不是数值。例如，在数值3上二次方程函数的导数告诉我们，在数值3上这个函数改变如何的快。更具体的说，你可能记得梯度被定义为*上升/运行*。也就是说，函数值的变化除以参数值的变化。当我们知道我们的函数将如何变化时，然后我们就知道我们需要做什么以使它更小。这是机器学习的关键：有个改变一个函数参数的方法，使函数更小。微积分提供给我们一个计算的捷径：导数。这让我们可直接的计算我们函数的梯度。

One important thing to be aware of is that our function has lots of weights that we need to adjust, so when we calculate the derivative we won't get back one number, but lots of them—a gradient for every weight. But there is nothing mathematically tricky here; you can calculate the derivative with respect to one weight, and treat all the other ones as constant, then repeat that for each other weight. This is how all of the gradients are calculated, for every weight.

一个需要注意的重要事情是我们的函数有很多权重需要调整，所以当我们计算导数时，不能只返回一个数，而是很多：对每个权重的梯度。但这里没有任何数学技巧，你能够计算一个权重的导数，并处理所有的其它权重为常数，然后对每个其它权重重复求导。

We mentioned just now that you won't have to calculate any gradients yourself. How can that be? Amazingly enough, PyTorch is able to automatically compute the derivative of nearly any function! What's more, it does it very fast. Most of the time, it will be at least as fast as any derivative function that you can create by hand. Let's see an example.

我们刚刚已经提到我不必自己计算任何梯度。那如何做到呢？非常奇妙，PyTorch能够自动计算几乎任何函数的导数！更重要的是，它做的非常快。大多数时间，至少它与你能手工创建的任何求导函数一样快。让我们看一个例子。

First, let's pick a tensor value which we want gradients at:

首先，让我们选择一个张量值，我们想在上面求梯度的：

```python
xt = tensor(3.).requires_grad_()
```

Notice the special method `requires_grad_`? That's the magical incantation we use to tell PyTorch that we want to calculate gradients with respect to that variable at that value. It is essentially tagging the variable, so PyTorch will remember to keep track of how to compute gradients of the other, direct calculations on it that you will ask for.

注意这个特定方法`requires_grad_`？这是我们用于告诉PyTorch，在这个值上我们想要计算关于这个变量梯度的方法。

> a: This API might throw you off if you're coming from math or physics. In those contexts the "gradient" of a function is just another function (i.e., its derivative), so you might expect gradient-related APIs to give you a new function. But in deep learning, "gradients" usually means the *value* of a function's derivative at a particular argument value. The PyTorch API also puts the focus on the argument, not the function you're actually computing the gradients of. It may feel backwards at first, but it's just a different perspective.
>
> 亚：如果你来自数学或物理专业 这个API可能会让你失望。在那些情况下一个函数的“梯度”只是另一个函数（即它的导数），所以你可以预想到这些梯度相关的API给了你一个新函数。但在深度学习里，“梯度”通常表达的意思是在特定参数值上一个函数导数的*值*。PyTorch的API也注意力放在参数上，而不是我们正在实际计算梯度的函数上。在一开始可能感觉它倒退了，但是这只是不同的观点。

Now we calculate our function with that value. Notice how PyTorch prints not just the value calculated, but also a note that it has a gradient function it'll be using to calculate our gradients when needed:

现在我们用这个值来计算我们的函数。注意PyTorch的输出不仅是计算后的值，而且是有一个梯度函数的注解，当需要的时候，它将被用于计算我们的梯度：

```python
yt = f(xt)
yt
```

Out: tensor(9., grad_fn=<PowBackward0>)

Finally, we tell PyTorch to calculate the gradients for us:

最后，我们告诉PyTorch为我们计算梯度：

```python
yt.backward()
```

The "backward" here refers to *backpropagation*, which is the name given to the process of calculating the derivative of each layer. We'll see how this is done exactly in chapter <chapter_foundations>, when we calculate the gradients of a deep neural net from scratch. This is called the "backward pass" of the network, as opposed to the "forward pass," which is where the activations are calculated. Life would probably be easier if `backward` was just called `calculate_grad`, but deep learning folks really do like to add jargon everywhere they can!

这里的“backward”指的是*反向传播*，这个名字给出了每导计算导数的过程。当我们从头开始计算一个深度神经网络的梯度时，在<章节：基础>里我们会看到具体如何做的。这被称为网络的“反向传递”，与之相对的是“前向传递”，后者是计算激活的位置。如果`backward`只是被叫做`计算梯度`工作可能会更容易，但做深度学习的这些人真喜欢在任何能加的地方的增加术语！

We can now view the gradients by checking the `grad` attribute of our tensor:

我们现在能够通过检查张量的`grad`属性查看这些梯度：

```python
xt.grad
```

Out: tensor(6.)

If you remember your high school calculus rules, the derivative of `x**2` is `2*x`, and we have `x=3`, so the gradients should be `2*3=6`, which is what PyTorch calculated for us!

如果你还记得高中微积分规则，`x**2`的导数是`2*x`，当`x=3`，梯度应该是`2*3=6`，这就是PyTorchy为我们计算的结果！

Now we'll repeat the preceding steps, but with a vector argument for our function:

现在我们会重复向前的这些步骤，只是为我们的函数使用一个向量参数：

```python
xt = tensor([3.,4.,10.]).requires_grad_()
xt
```

Out: tensor([ 3.,  4., 10.], requires_grad=True)

And we'll add `sum` to our function so it can take a vector (i.e., a rank-1 tensor), and return a scalar (i.e., a rank-0 tensor):

我们会增加`sum`到我们的函数，所以它能取一个向量（即1阶张量）并后加一个标量（即，一个0阶张量）：

```python
def f(x): return (x**2).sum()

yt = f(xt)
yt
```

Out: tensor(125., grad_fn=<SumBackward0>)

Our gradients are `2*xt`, as we'd expect!

正如我们所预期的，我们的梯度是`2*xt`！

```python
yt.backward()
xt.grad
```

Out: tensor([ 6.,  8., 20.])

The gradients only tell us the slope of our function, they don't actually tell us exactly how far to adjust the parameters. But it gives us some idea of how far; if the slope is very large, then that may suggest that we have more adjustments to do, whereas if the slope is very small, that may suggest that we are close to the optimal value.

这些梯度只告诉我们函数的斜率，他们没有真正的告诉我们具体调整参数多少。但它给了我们调整多少的一些想法。如果斜率非常大，其后可能建议我们多调整一些。而如果斜率非常小，那可能的建议是我们接近最优值了。

### Stepping With a Learning Rate

### 用学习率步进

Deciding how to change our parameters based on the values of the gradients is an important part of the deep learning process. Nearly all approaches start with the basic idea of multiplying the gradient by some small number, called the *learning rate* (LR). The learning rate is often a number between 0.001 and 0.1, although it could be anything. Often, people select a learning rate just by trying a few, and finding which results in the best model after training (we'll show you a better approach later in this book, called the *learning rate finder*). Once you've picked a learning rate, you can adjust your parameters using this simple function:

基于梯度值决定如何改变我们的参数是深度学习过程一个重要部分。几乎所有方法都是从一个小数乘以梯度的基本思想开始的，这个小数称为*学习率*（LR）。学习率通常是在0.001到0.1之间的一个数字，当然它能够是任何数字。通常，人们只是通过一点点尝试来选择一个学习率，并寻找那个结果在训练后是最好的模型（我们稍后在本书会给你展示一个更好的方法，称为*学习率查找器*）。一旦你选中了一个学习率，利用下面这个简单函数你就能够调整参数了：

​	w -= gradient(w) * lr

This is known as *stepping* your parameters, using an *optimizer step*.

这被称为利用一个*优化步骤*，*步进* 你的参数。

If you pick a learning rate that's too low, it can mean having to do a lot of steps. <descent_small> illustrates that.

如果你选中的学习率太小，这意味着必须要做很多步。如下图<低学习率梯度下降>的插图说明。

<div>
  <p align="center">
    <img src="./_v_images/chapter2_small.svg" alt="An illustration of gradient descent with a LR too low" width="400" caption="Gradient descent with low LR" id="descent_small" />
  </p>
  <p align="center">
    低学习率梯度下降
  </p>
</div>

But picking a learning rate that's too high is even worse—it can actually result in the loss getting *worse*, as we see in <descent_div>!

如果选择的学习率太大是很糟糕的，在损失方向它实际的结果会更*糟*，正如下图所求<高学习率梯度下降>！

<div>
  <p align="center">
    <img alt="An illustration of gradient descent with a LR too high" width="400" caption="Gradient descent with high LR" src="./_v_images/chapter2_div.svg" id="descent_div" />
  </p>
  <p align="center">
    高学习率梯度下降
  </p>
</div>

If the learning rate is too high, it may also "bounce" around, rather than actually diverging; <<descent_bouncy>> shows how this has the result of taking many steps to train successfully.

如果学习率太高，它可能会围着曲线*弹跳*，而不是发散。下图<弹跳学习率梯度下降>展示了如何花费了很多步取得这样的结果使训练成功。

<div>
  <p align="center">
    <img alt="An illustation of gradient descent with a bouncy LR" width="400" caption="Gradient descent with bouncy LR" src="./_v_images/chapter2_bouncy.svg" id="descent_bouncy"/>
  </p>
  <p align="center">
    弹跳学习率梯度下降
  </p>
</div>

Now let's apply all of this in an end-to-end example.

现在让我们在一个端到端的例子中应用上述所有步骤。

### An End-to-End SGD Example

### 一个端到端随机梯度下降的例子

We've seen how to use gradients to find a minimum. Now it's time to look at an SGD example and see how finding a minimum can be used to train a model to fit data better.

我们已经看了如何利用梯度来寻找一个最小数。现在是时候来看一个随机梯度下降的例子，并看如何寻找一个最小值，这个值能够被用于训练一个更好拟合数的模型。

Let's start with a simple, synthetic, example model. Imagine you were measuring the speed of a roller coaster as it went over the top of a hump. It would start fast, and then get slower as it went up the hill; it would be slowest at the top, and it would then speed up again as it went downhill. You want to build a model of how the speed changes over time. If you were measuring the speed manually every second for 20 seconds, it might look something like this:

让我们从一个简单的合成实例模型开始！想像你正在测试一个过山车越过一个驼峰顶部的速度。它也许开始很快，然后在上山的时候速度变慢。它也会在顶部是最慢的，随后它下山的速度会再次加快。我想创建一个随着时间推移速度如何变化的模型。如果你正在手工测量20秒内每秒的速度，一些内容看起来可能像这样：

```python
time = torch.arange(0,20).float(); time
```

Out: tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])

```python
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);
```

Out: <img src="./_v_images/rollerspeed.png" alt="rollerspeed" style="zoom:100%;" />

We've added a bit of random noise, since measuring things manually isn't precise. This means it's not that easy to answer the question: what was the roller coaster's speed? Using SGD we can try to find a function that matches our observations. We can't consider every possible function, so let's use a guess that it will be quadratic; i.e., a function of the form `a*(time**2)+(b*time)+c`.

我们增加了一点随机噪声，因为手动测量并不精准。这意味着它不容易回答问题：过山车的速度是什么？利用随机梯度下降我们能够尝试找到一个函数用于比对我们的观测数据。我们不可能考虑到每个可能的函数，所以让我们用猜想的方式，它会是一个二次方程。即，函数形式是``a*(time**2)+(b*time)+c`。

We want to distinguish clearly between the function's input (the time when we are measuring the coaster's speed) and its parameters (the values that define *which* quadratic we're trying). So, let's collect the parameters in one argument and thus separate the input, `t`, and the parameters, `params`, in the function's signature:

我们想要函数的输入（当我们正在测量的过山车速度时间的时候）和它的参数（我们正在尝试定义的二次方程的值）之间有清晰区分。所以，让我们在一个传参中收集参数，从而在函数签名中拆分输入`t`和参数`params`：

```python
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
```

In other words, we've restricted the problem of finding the best imaginable function that fits the data, to finding the best *quadratic* function. This greatly simplifies the problem, since every quadratic function is fully defined by the three parameters `a`, `b`, and `c`. Thus, to find the best quadratic function, we only need to find the best values for `a`, `b`, and `c`.

换句话说，我们已经限定了拟合数据的可想像的最好的函数问题，来查找最好的*二次*函数。这是非常简单的问题，因为每个二次函数完全通过三个参数`a`、`b`和`c`来定义。因为，查找最好的二次函数，我们只需要找最好的`a`、`b`和`c`的值就行。

If we can solve this problem for the three parameters of a quadratic function, we'll be able to apply the same approach for other, more complex functions with more parameters—such as a neural net. Let's find the parameters for `f` first, and then we'll come back and do the same thing for the MNIST dataset with a neural net.

如果我们能够解决二次函数三个参数的问题，我们就能够对其它函数应用两样的方法，更复杂的函数拥有更多的参数，例如一个神经网络。让我们首先找`f`参数，然后我们会返回并对MNIST数据集用一个神经网络做同样的事情。

We need to define first what we mean by "best." We define this precisely by choosing a *loss function*, which will return a value based on a prediction and a target, where lower values of the function correspond to "better" predictions. For continuous data, it's common to use *mean squared error*:

我们需要首先定义我们“最好”的含义。我们通过选择一个*损失函数*来严谨的定义，这个函数会基于一个预测和一个目标返回一个值，函数最小值与“最好”预测相符。对于连续的数据，它通常用*均方误差*：

```python
def mse(preds, targets): return ((preds-targets)**2).mean()
```

Now, let's work through our 7 step process.

现在，让我们通过7个步骤过程来处理。

####  Step 1: Initialize the parameters

#### 步骤一：初始化参数

First, we initialize the parameters to random values, and tell PyTorch that we want to track their gradients, using `requires_grad_`:

首先，我们初始化参数为随机值，并告诉PyTorch我们利用`requires_grad_`方法来跟踪他们的梯度：

```python
params = torch.randn(3).requires_grad_()
```

```python
#hide
orig_params = params.clone()
```

#### Step 2: Calculate the predictions

#### 步骤二：计算预测

Next, we calculate the predictions:

下一步，我们计算预测：

```python
preds = f(time, params)
```

Let's create a little function to see how close our predictions are to our targets, and take a look:

让我们创建一个小函数，来看我们的预测是如何接近我们的目标的，并输出图形看一下：

```python
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)
```

```python
show_preds(preds)
```

Out: <img src="./_v_images/show_preds.png" alt="show_preds" style="zoom:100%;" />

This doesn't look very close—our random parameters suggest that the roller coaster will end up going backwards, since we have negative speeds!

看起来不是非常接近，我们随机参数代表的是过山车最终会倒退，因为我们的速度是负数！

#### Step 3: Calculate the loss

#### 步骤三：计算损失

We calculate the loss as follows:

接下来我们计算损失：

```python
loss = mse(preds, speed)
loss
```

Out: tensor(25823.8086, grad_fn=<MeanBackward0>)

Our goal is now to improve this. To do that, we'll need to know the gradients.

我们的目标是现在来改善它。做这个事，我们需要知道梯度。

#### Step 4: Calculate the gradients

#### 步骤四：计算梯度

The next step is to calculate the gradients. In other words, calculate an approximation of how the parameters need to change:

下一步骤是来计算梯度。换名话说，计算一个参数如果改变的进似值：

```python
loss.backward()
params.grad
```

Out: tensor([-53195.8594,  -3419.7146,   -253.8908])

```python
params.grad * 1e-5
```

Out: tensor([-0.5320, -0.0342, -0.0025])

We can use these gradients to improve our parameters. We'll need to pick a learning rate (we'll discuss how to do that in practice in the next chapter; for now we'll just use 1e-5, or 0.00001):

我们能够用这些梯度来改善我们的参数。我们将需要选择一个学习率（我们会在下一章节讨论在实践中如果来做。现在我们只用 1e-5或0.00001）

```python
params
```

Out: tensor([-0.7658, -0.7506,  1.3525], requires_grad=True)

#### Step 5: Step the weights

#### 步骤五：步进权重

Now we need to update the parameters based on the gradients we just calculated:

现在我们需要基于我们刚刚计算过的梯度来重新参数：

```python
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
```

> a: Understanding this bit depends on remembering recent history. To calculate the gradients we call `backward` on the `loss`. But this `loss` was itself calculated by `mse`, which in turn took `preds` as an input, which was calculated using `f` taking as an input `params`, which was the object on which we originally called `required_grads_`—which is the original call that now allows us to call `backward` on `loss`. This chain of function calls represents the mathematical composition of functions, which enables PyTorch to use calculus's chain rule under the hood to calculate these gradients.
>
> 亚：依赖所记住的最近过程来理解这一点。计算梯度我们调用`loss`上的`backward`。但这个`loss`是通过`mse`它自己计算的，mse反过来取`preds`做为输入，preds利用`params`做为输入的`f`计算得出的，params是我们最初称为`requires_grad_`上的对象，这一原始调用允许我们调用`loss`上的`backward`。这一函数链调用相当于数学的函数组装，这使得PyTorch能够在后台用微积分链规则来计算这些梯度。

Let's see if the loss has improved:

让我们看看损失是否已经改善：

```python
preds = f(time,params)
mse(preds, speed)
```

Out: tensor(5435.5366, grad_fn=<MeanBackward0>)

And take a look at the plot:

并输出图形看一下：

```python
show_preds(preds)
```

Out: <img src="./_v_images/show_preds1.png" alt="show_preds1" style="zoom:100%;" />

We need to repeat this a few times, so we'll create a function to apply one step:

我们需要重复这一过程几次，所以我们要创建一个函数来应用这一步骤：

```python
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds
```

#### Step 6: Repeat the process

#### 步骤六：重复这一过程

Now we iterate. By looping and performing many improvements, we hope to reach a good result:
现在我们迭代。能够循环和做一些改善，我们希望找到一个好的结果：

```python
for i in range(10): apply_step(params)
```

Out:  $
\begin{array} {}
	5435.53662109375\\
	1577.4495849609375\\
	847.3780517578125\\
	709.22265625\\
	683.0757446289062\\
	678.12451171875\\
	677.1839599609375\\
	677.0025024414062\\
	676.96435546875\\
	676.9537353515625
\end{array}
$

```python
#hide
params = orig_params.detach().requires_grad_()
```

The loss is going down, just as we hoped! But looking only at these loss numbers disguises the fact that each iteration represents an entirely different quadratic function being tried, on the way to finding the best possible quadratic function. We can see this process visually if, instead of printing out the loss function, we plot the function at every step. Then we can see how the shape is approaching the best possible quadratic function for our data:

正如我们所希望的，损失正在下降！但这些损失数值看起来掩盖了每次迭代都代表尝试的是一个完全不同的二次函数，这是寻找最好可能性的二次函数方法。如果我们能够标定每一步的函数，来替代损失函数的输出，我们能够看到把这一可视化过程。然后我们能够看到相关步骤对我们的数据是如何接近最好可能性的二次函数：

```python
_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()
```

Out: <img src="/Users/Y.H/Documents/GitHub/smartbook/smartbook/_v_images/tigh_plot.png" alt="tigh_plot" style="zoom:100%;" />

#### Step 7: stop

#### 步骤七：停止

We just decided to stop after 10 epochs arbitrarily. In practice, we would watch the training and validation losses and our metrics to decide when to stop, as we've discussed.

我们只是武断的决定10个周期后就停止。正如我们已经讨论过的，在实践中，我们也许要看训练和验证损失及我们的指标来决定什么时候停止。

### Summarizing Gradient Descent

### 总结梯度下降

```python
#hide_input
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')
```

```mermaid
graph LR
init((init))
predict((predict))
loss((loss))
gradient((gradient))
step((step))
stop((stop))
subgraph the_gradient_descent_process
init --> predict --> loss --> gradient  --> step --> stop 
end
subgraph the_gradient_descent_process
step --repeat--> predict
end
```
```mermaid
graph LR
init((初始化))
predict((预测))
loss((损失))
gradient((梯度))
step((步进))
stop((停止))
subgraph 梯度下降过程
init --> predict --> loss --> gradient  --> step --> stop 
end
subgraph 梯度下降过程
step --重复--> predict
end
```

To summarize, at the beginning, the weights of our model can be random (training *from scratch*) or come from a pretrained model (*transfer learning*). In the first case, the output we will get from our inputs won't have anything to do with what we want, and even in the second case, it's very likely the pretrained model won't be very good at the specific task we are targeting. So the model will need to *learn* better weights.

总结一下，在一开始，我们模型的权重能够被随机化（*从零*开始训练）或来自一个预训练模型（*迁移学习*）。在第一个例子中，从我们的输入获得输出与我们我们想要做的事情没有任何关系，即使在第二个例子中，预训练模型非常可能在我们特定目标任务上不是非常的好。所以这个模型需要*学习* 更好的权重。

We begin by comparing the outputs the model gives us with our targets (we have labeled data, so we know what result the model should give) using a *loss function*, which returns a number that we want to make as low as possible by improving our weights. To do this, we take a few data items (such as images) from the training set and feed them to our model. We compare the corresponding targets using our loss function, and the score we get tells us how wrong our predictions were. We then change the weights a little bit to make it slightly better.

我们利用一个*损失函数*，通过对比模型给我们的输出与我们的目标（我们已经有了标注数据，所以我们知道模型应该给我们提供什么结果）开始，损失函数返回的数值，我们希望通过改善我们的权重使其尽可能的低。我们从训练集取一些数据（如图像）并把这些数据喂给我们的模型，来做这个事情。我们利用损失函数对比相符的目标，我们得到的分数会告诉我们的预测糟糕程度。然后我们稍微变更权重来让它稍稍好一些。

To find how to change the weights to make the loss a bit better, we use calculus to calculate the *gradients*. (Actually, we let PyTorch do it for us!) Let's consider an analogy. Imagine you are lost in the mountains with your car parked at the lowest point. To find your way back to it, you might wander in a random direction, but that probably wouldn't help much. Since you know your vehicle is at the lowest point, you would be better off going downhill. By always taking a step in the direction of the steepest downward slope, you should eventually arrive at your destination. We use the magnitude of the gradient (i.e., the steepness of the slope) to tell us how big a step to take; specifically, we multiply the gradient by a number we choose called the *learning rate* to decide on the step size. We then *iterate* until we have reached the lowest point, which will be our parking lot, then we can *stop*.

我们用微积分计算*梯度*来寻找如何改变权重以使损失稍微好点。（实际上，我们让PyTorch为我们做这个工作！）让我们思考一个推演。想像你在山里迷路了，你的车停放在最低点。你可能会感到奇怪，用一个随机的方向来寻找你返回停车场的路，这可能不会有太多帮助。因为你知道你的车在最低点，你最终应该抵达你的目的地。我们用梯度的大小（即，斜坡的陡峭度）来告诉我们一步是多大。通过我们选择的*学习率* 乘以梯度来决定一步的大小。然后我们*重复* 这些步骤直到我们搜索到最低点，这就是我们的汽车停靠点，这时我们就可以*停止*了。

All of that we just saw can be transposed directly to the MNIST dataset, except for the loss function. Let's now see how we can define a good training objective.

所有我们刚刚看到的这些内容，除了损失函数，都能够直接转移到MNIST数据集上去。现在让我们看看我们能够如果定义一个好的训练目标。

## The MNIST Loss Function

## MNIST损失函数

We already have our dependent variables `x`—these are the images themselves. We'll concatenate them all into a single tensor, and also change them from a list of matrices (a rank-3 tensor) to a list of vectors (a rank-2 tensor). We can do this using `view`, which is a PyTorch method that changes the shape of a tensor without changing its contents. `-1` is a special parameter to `view` that means "make this axis as big as necessary to fit all the data":

我们已经有了依赖的变量`x`，这些变量值就是图像自身。我们会把所有图像串联一个单张量中，然后把他们从一个矩阵（三阶张量）列表改变为一个向量（2阶张量）列。我们通用用`view`方法来做这个工作，这是一个PyTorch方法，它能在不改变张量内容的情况下改变张量的形状。对于`view`来说`-1`是一个特殊的参数，意思是“使得这个坐标轴与必须适合的所有数据一样大”：

```python
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```

We need a label for each image. We'll use `1` for 3s and `0` for 7s:

对每张图像我们都需要一个标签。我们会对图像3用`1`来表示，图像7用`0`来表示：

```python
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
```

Out: (torch.Size([12396, 784]), torch.Size([12396, 1]))

A `Dataset` in PyTorch is required to return a tuple of `(x,y)` when indexed. Python provides a `zip` function which, when combined with `list`, provides a simple way to get this functionality:

在PyTorch中一个`数据集`当被索引后需要后加一个`(x,y)`元组。Python提供了一个`zip`函数，当与`list`组合时，提供了一个简单方法来获取这一功能：

```python
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
```

Out: (torch.Size([784]), tensor([1]))

```python
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

Now we need an (initially random) weight for every pixel (this is the *initialize* step in our seven-step process):

现在我们需要一个对每个像素的（随机初始）权重（在我们七步骤过程中，这是`初始化`步骤）：

```python
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```

```PYTHON
weights = init_params((28*28,1))
```

The function `weights*pixels` won't be flexible enough—it is always equal to 0 when the pixels are equal to 0 (i.e., its *intercept* is 0). You might remember from high school math that the formula for a line is `y=w*x+b`; we still need the `b`. We'll initialize it to a random number too:

函数`weights*pixels`不是足够的灵活，当像素等于零时它会一直等于零（即它*截距*是零）。我们可能记得高中数学的一个线性方式是`y=w*x+b`，我们一直需要`b`，我们也会初始化它为一个随机数：

```python
bias = init_params(1)
```

In neural networks, the `w` in the equation `y=w*x+b` is called the *weights*, and the `b` is called the *bias*. Together, the weights and bias make up the *parameters*.

在神经网络中，在等式`y=w*x+b`中`w`被称为*权重*，`b`被称为*偏差*。权重和偏差在一起组成了*参数*。

> jargon: Parameters: The *weights* and *biases* of a model. The weights are the `w` in the equation `w*x+b`, and the biases are the `b` in that equation.
>
> 术语：参数：一个模型的*权重*和*偏差*。在等式`w*x+b`中权重是`w`，偏差是`b`。

We can now calculate a prediction for one image:

对一个图像，现在我们能够计算预测值：

```python
(train_x[0]*weights.T).sum() + bias
```

Out: tensor([20.2336], grad_fn=<AddBackward0>)

While we could use a Python `for` loop to calculate the prediction for each image, that would be very slow. Because Python loops don't run on the GPU, and because Python is a slow language for loops in general, we need to represent as much of the computation in a model as possible using higher-level functions.

当我们使用Python的`for`循环来计算每一张图像时，那也许会非常慢。因为Python的循环不能在GPU上运行，并且因为通常对于循环Python是一个很慢的语言，我们需要在一个模型里尽可能使用高级函数来表述更多的计算。

In this case, there's an extremely convenient mathematical operation that calculates `w*x` for every row of a matrix—it's called *matrix multiplication*. <matmul> shows what matrix multiplication looks like.

在这个例子中，计算`w*x`为矩阵的每个行这是一个极为方便的数学操作。这称为*矩阵乘法*。下图为矩阵乘法的示意图。

<div>
  <p align="center">
    <img src="./_v_images/matmul2.svg" alt="Matrix multiplication" width="400" caption="Matrix multiplication"  id="matmul"/>
  </p>
  <p align="center">
    矩阵乘法示意图
  </p>
</div>

This image shows two matrices, `A` and `B`, being multiplied together. Each item of the result, which we'll call `AB`, contains each item of its corresponding row of `A` multiplied by each item of its corresponding column of `B`, added together. For instance, row 1, column 2 (the orange dot with a red border) is calculated as a1,1∗b1,2+a1,2∗b2,2a1,1∗b1,2+a1,2∗b2,2. If you need a refresher on matrix multiplication, we suggest you take a look at the [Intro to Matrix Multiplication](https://youtu.be/kT4Mp9EdVqs) on *Khan Academy*, since this is the most important mathematical operation in deep learning.

这个图显示了`A`和`B`两个相乘的矩阵。每个结果项我们称为`AB`，它包含了`A`相应行上的每项乘以`B`相应列上的每项，然后加总。例如，行1和列2（红色边线的橘黄色点）是a1,1∗b1,2+a1,2∗b2,2a1,1∗b1,2+a1,2∗b2,2计算后的结果。如果你需要复习矩阵乘法，我们建议你看一下*可汗学院*上的[矩阵乘法概述](https://youtu.be/kT4Mp9EdVqs)，因为这是深度学习中最重要的数学运算。

In Python, matrix multiplication is represented with the `@` operator. Let's try it:

在Python中，矩阵乘法用`@`运算符代表。让我们尝试一下：

```python
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
```

Out: $
\begin{matrix} tensor([&[20.2336],\\
									&[17.0644],\\
									&[15.2384],\\
									&...,\\
									&[18.3804],\\
									&[23.8567],\\
									&[28.6816]]&, grad\_fn=<AddBackward0>)
\end{matrix}
$

The first element is the same as we calculated before, as we'd expect. This equation, `batch@weights + bias`, is one of the two fundamental equations of any neural network (the other one is the *activation function*, which we'll see in a moment).

正如我们所期望的，第一个元素与我们以前的计算是一致的。`batch@weights + bias`这个等式，是神经网络两个基础等式中的一个（另一个是*激活函数*，稍后我们会看到它）。

Let's check our accuracy. To decide if an output represents a 3 or a 7, we can just check whether it's greater than 0, so our accuracy for each item can be calculated (using broadcasting, so no loops!) with:

让我们检查一下精度。决定一个输出代表3或7，我们只能够检查是否它比零大，所以我们对每项的精度能够被计算（用广播，而不是循环）：

```python
corrects = (preds>0.0).float() == train_y
corrects
```

Out:  $
\begin{array} {l}tensor([&[True],\\
									&[True],\\
									&[True],\\
									&...,\\
									&[False],\\
									&[False],\\
									&[False]]&)
\end{array}
$

```python
corrects.float().mean().item()
```

Out: 0.4912068545818329

Now let's see what the change in accuracy is for a small change in one of the weights:

现在让我们看一下，对于其中一个权重做个小的改变，精度方面会有什么样的变化：

```python
weights[0] *= 1.0001
```

```python
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()
```

Out: 0.4912068545818329

As we've seen, we need gradients in order to improve our model using SGD, and in order to calculate gradients we need some *loss function* that represents how good our model is. That is because the gradients are a measure of how that loss function changes with small tweaks to the weights.

正如我们看到的，为了利用随机梯度下降来改善我们的模型我们需要梯度，并且为了计算梯度，我们需要代表我们模型是如何好的*损失函数*。因为梯度是损失函数如何微小调整权重的度量。

So, we need to choose a loss function. The obvious approach would be to use accuracy, which is our metric, as our loss function as well. In this case, we would calculate our prediction for each image, collect these values to calculate an overall accuracy, and then calculate the gradients of each weight with respect to that overall accuracy.

所以，我们需要来挑选一个损失函数。明显的方法应该是用我们的指标：精度，还有我们的损失函数。在这个例子中，我们对每张图像计算预测，收集这些值来计算一个整体精度，然后根据整体精度来计算每个权重的梯度。

Unfortunately, we have a significant technical problem here. The gradient of a function is its *slope*, or its steepness, which can be defined as *rise over run*—that is, how much the value of the function goes up or down, divided by how much we changed the input. We can write this in mathematically as: `(y_new - y_old) / (x_new - x_old)`. This gives us a good approximation of the gradient when `x_new` is very similar to `x_old`, meaning that their difference is very small. But accuracy only changes at all when a prediction changes from a 3 to a 7, or vice versa. The problem is that a small change in weights from `x_old` to `x_new` isn't likely to cause any prediction to change, so `(y_new - y_old)` will almost always be 0. In other words, the gradient is 0 almost everywhere.

很不幸，我们有一个重大的技术问题。一个函数的梯度是它的*斜率*，或它的陡度（能被定义为*随着运行上升*，即，函数值上升或下降多少），除以我们输入改变的大小。我们能够用数学方程式来表达：`(y_new - y_old) / (x_new - x_old)`。当`x_new`和`x_old`非常相似的时候，这就给了我们好的近似梯度，这意味着他们差异非常小。当预测改变从3到7的时候，精度才会完全改变，反之亦然。有个问题，在权重中从 `x_old` 到 `x_new`一个很小的改变不太可能引发任何预测的改变，所以`(y_new - y_old)`几乎一直是零。换句话说，梯度几乎在任何地方都是零。

A very small change in the value of a weight will often not actually change the accuracy at all. This means it is not useful to use accuracy as a loss function—if we do, most of the time our gradients will actually be 0, and the model will not be able to learn from that number.

在一个权重的数值中非常小的改变通常根本不会引发精度的实际改变。这意味着做为损失函数，它使用精度是没有帮助的，如果我们这样做了，绝大多数时间我们的梯度的实际上会是零，并且模型也无法从那些数字中学习。

> S: In mathematical terms, accuracy is a function that is constant almost everywhere (except at the threshold, 0.5), so its derivative is nil almost everywhere (and infinity at the threshold). This then gives gradients that are 0 or infinite, which are useless for updating the model.
>
> 西：在数学的术语中，精度是一个几乎在任何地方都恒定不变的函数（除阈值是0.5），所以它的导数几乎在任何地方都是零（阈值是无限大）。然后给出的梯度要么是零，要么是无限大，这对更新模型没有任何用处。

Instead, we need a loss function which, when our weights result in slightly better predictions, gives us a slightly better loss. So what does a "slightly better prediction" look like, exactly? Well, in this case, it means that if the correct answer is a 3 the score is a little higher, or if the correct answer is a 7 the score is a little lower.

相替代的是，当我们的权重结果是稍微更好的预测的时候，我们需要给我们稍微更好损失的损失函数。那么一个“稍微更好的预测”看起来到底是什么样子呢？所以，在这个例子中，意味着如果正确的答案是3，其分数会稍微更高些，或者如果正确答案是7，其分数会稍微更低些。

Let's write such a function now. What form does it take?

让我们现在写一个这样的函数。它采取什么形式呢？

The loss function receives not the images themseles, but the predictions from the model. Let's make one argument, `prds`, of values between 0 and 1, where each value is the prediction that an image is a 3. It is a vector (i.e., a rank-1 tensor), indexed over the images.

损失函数所接收的没有图像自身，而是来自模型的预测。让我们指定一个取值范围在0和1之间的参数`prds`，每个都是图像3的预测值。它是一个向量（即1阶张量），依据图像进行的索引。

The purpose of the loss function is to measure the difference between predicted values and the true values — that is, the targets (aka labels). Let's make another argument, `trgts`, with values of 0 or 1 which tells whether an image actually is a 3 or not. It is also a vector (i.e., another rank-1 tensor), indexed over the images.

损失函数的目的是计量预测值和真实值之间的差异，真实值是目标（又称标签）。让我们指定另一个取值为0或1的参数`trgts`，这会告诉一个图像实际上是3或不是3.它也是一个依据图像索引的向量（即另一个1阶张量）。

So, for instance, suppose we had three images which we knew were a 3, a 7, and a 3. And suppose our model predicted with high confidence (`0.9`) that the first was a 3, with slight confidence (`0.4`) that the second was a 7, and with fair confidence (`0.2`), but incorrectly, that the last was a 7. This would mean our loss function would receive these values as its inputs:

所以，例如，假设我们有3张图像，我们知道分别为3、7、3。然后假设我们的模型预测非常确信（0.9）第一张是3，不太确信（0.4）第二张是7，然后不敢确信（0.2）且不正确的认为最后一张是7.也就代表我们的损失函数接收了这些数值做为输入：

```python
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])
```

Here's a first try at a loss function that measures the distance between `predictions` and `targets`:

这是在损失函数上计量`预测` 和 `目标`之间差距的第一个尝试：

```python
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

We're using a new function, `torch.where(a,b,c)`. This is the same as running the list comprehension `[b[i] if a[i] else c[i] for i in range(len(a))]`, except it works on tensors, at C/CUDA speed. In plain English, this function will measure how distant each prediction is from 1 if it should be 1, and how distant it is from 0 if it should be 0, and then it will take the mean of all those distances.

我们正在使用一个新的函数：`torch.where(a,b,c)`。这与运行的列表生成器 `[b[i] if a[i] else c[i] for i in range(len(a))]`类似，只是这个新的函数是以C/CUDA的速度运行在张量上。简单的来说，这个函数会测量每个预测与1的差距（如果它应该是1的话），或它与0的差距是多少（如果它应该是0的话），然后它会取所有这些差距的平均值。

> note: Read the Docs: It's important to learn about PyTorch functions like this, because looping over tensors in Python performs at Python speed, not C/CUDA speed! Try running `help(torch.where)` now to read the docs for this function, or, better still, look it up on the PyTorch documentation site.
>
> 注解：阅读文档：它对于学习像这样的PyTorch函数是重要的，因为在Python中张量上的循环性能，没有C/CUDA运行速度快的！现在尝试运行`help(torch.where)`阅读这个函数的文档，或查找PyTorch的在线文档会更好。

Let's try it on our `prds` and `trgts`:

让我们在`prds` 和 `trgts`上尝试一下这个新函数：

```python
torch.where(trgts==1, 1-prds, prds)
```

Out: tensor([0.1000, 0.4000, 0.8000])

You can see that this function returns a lower number when predictions are more accurate, when accurate predictions are more confident (higher absolute values), and when inaccurate predictions are less confident. In PyTorch, we always assume that a lower value of a loss function is better. Since we need a scalar for the final loss, `mnist_loss` takes the mean of the previous tensor:

你能够看到，当预测更加精准时，即预测精度更确信（更高的绝对值），预测不准确是更低的确信，这个函数会返回更小的数值。在PyTorch中，我们一直假设一个更低的损失函数值是更好的。因为我们需要一个对最终损失的纯量，`mnist_loss`取了之前张量的平均值：

```python
mnist_loss(prds,trgts)
```

Out: tensor(0.4333)

For instance, if we change our prediction for the one "false" target from `0.2` to `0.8` the loss will go down, indicating that this is a better prediction:

例如，如果我们对一个“假”的目标预测从`0.2`进行改变到`0.8`，损失会降低，这就代表是一个好的预测结果：

```python
mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)
```

Out: tensor(0.2333)

One problem with `mnist_loss` as currently defined is that it assumes that predictions are always between 0 and 1. We need to ensure, then, that this is actually the case! As it happens, there is a function that does exactly that—let's take a look.

`mnist_loss`作为当前定义的一个问题是，它假设预测总是在0和1之间。我们需要确保实际是这个样子！正好有一个函数可以做到这个事情，让我们看一下。

### Sigmoid

### S型函数

The `sigmoid` function always outputs a number between 0 and 1. It's defined as follows:

`sigmoid`函数一些输出0和1之间的数字。它的定义如下：

```
def sigmoid(x): return 1/(1+torch.exp(-x))
```

PyTorch defines an accelerated version for us, so we don’t really need our own. This is an important function in deep learning, since we often want to ensure values are between 0 and 1. This is what it looks like:

PyTorch为我们定义了一个加速版，所以并需要我们自己来实际定义。在深度学习中这是一个重要函数，因为我们经常要确保数值在0和1之间。它看起来像如下形式：

```python
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
```

Out: <img src="./_v_images/sigmoid.png" alt="sigmoid" style="zoom:100%;" />

As you can see, it takes any input value, positive or negative, and smooshes it onto an output value between 0 and 1. It's also a smooth curve that only goes up, which makes it easier for SGD to find meaningful gradients.

正如你能看到的，它所取的任何输入值（正值或负值），并平滑输出0和1之间的值。它也是一个只向上的平滑曲线，使得它对随机梯度下降更容易找到有意义的梯度。

Let's update `mnist_loss` to first apply `sigmoid` to the inputs:

让我们更新`mnist_loss`首先用于`sigmoid`来做输入：

```
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

Now we can be confident our loss function will work, even if the predictions are not between 0 and 1. All that is required is that a higher prediction corresponds to higher confidence an image is a 3.

现在我们能够确信，即使预测不在0和1之间，我们的损失函数依然会有效。所有需要的是一个更高预测相当于更高的可信度一个图像是3.

Having defined a loss function, now is a good moment to recapitulate why we did this. After all, we already had a metric, which was overall accuracy. So why did we define a loss?

已经定义了一个损失函数，现在是时候来总结一下为什么我们做这个事情。毕竟我们已经有了一个整体精度的指标。那么为什么我们要定义一个损失那？

The key difference is that the metric is to drive human understanding and the loss is to drive automated learning. To drive automated learning, the loss must be a function that has a meaningful derivative. It can't have big flat sections and large jumps, but instead must be reasonably smooth. This is why we designed a loss function that would respond to small changes in confidence level. This requirement means that sometimes it does not really reflect exactly what we are trying to achieve, but is rather a compromise between our real goal, and a function that can be optimized using its gradient. The loss function is calculated for each item in our dataset, and then at the end of an epoch the loss values are all averaged and the overall mean is reported for the epoch.

一个关键的差异是指标驱动人类的理解而损失驱动自动学习。驱动自动学习，损失函数必须是个有意义导数的函数。它不能有很大的平坦部分和巨大的跳跃，而必须是合理的平滑。这就是为什么我们测试了一个损失函数，在可信的水平上响应小的改变。这个要求意味着有时候它不会真实准确的反应出我们努力实现的事情，而只是在我们真实目标间的一个妥协，并且利用它的梯度能够被优化的函数。损失函数是在我们的数据集中对每个项目计算的，然后在一个周期的最后损失值是所有的平均值，并且对于这个周期会被报告整体平均值。

Metrics, on the other hand, are the numbers that we really care about. These are the values that are printed at the end of each epoch that tell us how our model is really doing. It is important that we learn to focus on these metrics, rather than the loss, when judging the performance of a model.

另一方向，指标是我们真实关系的一些数字。这些数值会在每个周期的最后打印输出，告诉我们模型实际上做的怎么样。当评判一个模型表现的时候，我们学会关注这些指标而不是损失是重要的，

### SGD and Mini-Batches

### 随机梯度下降和最小批次

Now that we have a loss function that is suitable for driving SGD, we can consider some of the details involved in the next phase of the learning process, which is to change or update the weights based on the gradients. This is called an *optimization step*.

现在我们有一个损失函数，它对于随机梯度下降是合适的，我们能够思考一些涉及学习过程的下一阶段的细节，用于基于梯度来改变或更新权重。这被称为*优化步骤*。

In order to take an optimization step we need to calculate the loss over one or more data items. How many should we use? We could calculate it for the whole dataset, and take the average, or we could calculate it for a single data item. But neither of these is ideal. Calculating it for the whole dataset would take a very long time. Calculating it for a single item would not use much information, so it would result in a very imprecise and unstable gradient. That is, you'd be going to the trouble of updating the weights, but taking into account only how that would improve the model's performance on that single item.

为了获得优化步骤，我们需要计算在一个或多个数据项上的损失。我们应该用多少呢？我们能够对整个数据集计算它，并取平均值，或我们只对单一数据项计算它。但这两者都不是理想的。对整个数据集计算它需要花费非常长的时候。对单一数据项来计算它也许没有更多有用的信息，所以它也许会产生一个非常不准确及不稳定的梯度。即，你将陷入更新权重的麻烦，且只考虑了如何在单一数据项上模型表现的优化。

So instead we take a compromise between the two: we calculate the average loss for a few data items at a time. This is called a *mini-batch*. The number of data items in the mini-batch is called the *batch size*. A larger batch size means that you will get a more accurate and stable estimate of your dataset's gradients from the loss function, but it will take longer, and you will process fewer mini-batches per epoch. Choosing a good batch size is one of the decisions you need to make as a deep learning practitioner to train your model quickly and accurately. We will talk about how to make this choice throughout this book.

所以我们在两者之间采取了一个妥协：每一次我们对少量数据计算平均损失。这被称为*最小批次*。在最小批次中数据项的数目被称为*批次尺寸*。一个更大的批次尺寸意味着你获得来自损失函数对你数据集的梯度更准确和更稳定的评估。但它会花费更长时间，且每个周期你将处理更少的最小批次。为了快速和准确的训练你的模型，选择一个合适的批次尺寸是你做为一名深度学习行业人员需要做的一个决定。在这本书里我们会讨论如何做这个选择。

Another good reason for using mini-batches rather than calculating the gradient on individual data items is that, in practice, we nearly always do our training on an accelerator such as a GPU. These accelerators only perform well if they have lots of work to do at a time, so it's helpful if we can give them lots of data items to work on. Using mini-batches is one of the best ways to do this. However, if you give them too much data to work on at once, they run out of memory—making GPUs happy is also tricky!

在实践中，对于使用最小批次而不是计算单一数据项梯度的另一个合理原因是，我们几乎一直在加速器（例如GPU）上做我们的训练。这些加速器只有在我们同一时间做很多工作的时候它才会性能良好，所以如果我们能够给它们很多数据项来处理，这是有利的。使用最小批次来做这个事情是众多最优方法之一。然而，如果你同时给它们太多的处理数据，它们会内存溢出，让GPU开心也是很难的！

As we saw in our discussion of data augmentation in <chapter_production>, we get better generalization if we can vary things during training. One simple and effective thing we can vary is what data items we put in each mini-batch. Rather than simply enumerating our dataset in order for every epoch, instead what we normally do is randomly shuffle it on every epoch, before we create mini-batches. PyTorch and fastai provide a class that will do the shuffling and mini-batch collation for you, called `DataLoader`.

正如在<章节：产品>中我们看到的数据增强的讨论，训练期间如果我们能够多样化处理，我们会获得更好的泛化。一个简单且有效的处理是我们能够改变放入每个最小批次中的数据项。我们创建最小批次之前，标准的做法是对每个周期随机打乱数据项，而不是简单的为每个周期列举我们的数据集。PyTorch和fastai提供了一个名叫`DataLoader`的类，它会为你做这个洗牌和最小批次集合。

A `DataLoader` can take any Python collection and turn it into an iterator over many batches, like so:

`DataLoader`能够接受任何Python集合，并把它转化为多批次的迭代器，如下所示：

```python
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
```

Out: $
\begin{array}{rr} [tensor([ 3, &12,  &8, &10,  &2]),\\
				tensor([ 9, & 4,  &7, &14,  &5]),\\
				tensor([ 1, &13,  &0,  &6, &11])]
\end{array}
$

For training a model, we don't just want any Python collection, but a collection containing independent and dependent variables (that is, the inputs and targets of the model). A collection that contains tuples of independent and dependent variables is known in PyTorch as a `Dataset`. Here's an example of an extremely simple `Dataset`:

为了训练一个模型，我们不仅仅只想要Python集合，而是一个包含独立变量和因变量的集合（即，模型的输入和目标）。在PyTorch中，包含独立变量和因变量元组的集合被叫做`数据集`。下面是一个异常简单的`数据集`例子：

```
ds = L(enumerate(string.ascii_lowercase))
ds
```

Out: (#26) [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, 'h'),(8, 'i'),(9, 'j')...]

When we pass a `Dataset` to a `DataLoader` we will get back many batches which are themselves tuples of tensors representing batches of independent and dependent variables:

当我们传递一个`数据集`给`DataLoader`，我们会获得许多批次，它们自己的张量元组代表独立变量和因变量的批次：

```
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
```

Out:$
{\begin{array} {l}
	\begin{array}{rrrrrrrl}
		[(tensor([&17, &18, &10, &22, &8, &14]), &('r', 's', 'k', 'w', 'i', 'o')),\\
		(tensor([&20, &15, & 9, &13, &21, &12]), &('u', 'p', 'j', 'n', 'v', 'm')),\\
		(tensor([& 7, &25, & 6, & 5, &11, &23]), &('h', 'z', 'g', 'f', 'l', 'x')),\\
		(tensor([& 1, & 3, & 0, &24, &19, &16]), &('b', 'd', 'a', 'y', 't', 'q')),
	\end{array}		
	\\		
	\begin{array}{lrr} 
		\ (tensor([&2,& 4]), ('c', 'e'))]
	\end{array}
\end{array}}
$

We are now ready to write our first training loop for a model using SGD!

我们现在准备利用随机梯度下降编写我们的第一个模型训练循环了！

## Putting It All Together

## 合并所有过程

It's time to implement the process we saw in <gradient_descent>. In code, our process will be implemented something like this for each epoch:

现在来实施我们在<梯度下降>小节中看到的处理过程。在代码中，我们的过程对每个批次处理的内容如下所求：

```python
for x,y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
```

First, let's re-initialize our parameters:

首先让我们重新初始化我们的参数：

```
weights = init_params((28*28,1))
bias = init_params(1)
```

A `DataLoader` can be created from a `Dataset`:

从一个`数据集`中能够创建一个`DataLoader`：

```
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape
```

Out: (torch.Size([256, 784]), torch.Size([256, 1]))

We'll do the same for the validation set:

我们会做一个相同的验证集：

```
valid_dl = DataLoader(valid_dset, batch_size=256)
```

Let's create a mini-batch of size 4 for testing:

让我们创建一个尺寸为4的最小批次用于测试：

```
batch = train_x[:4]
batch.shape
```

Out: torch.Size([4, 784])

```python
preds = linear1(batch)
preds
```

Out: $
\begin{array}{llrll} tensor([&[& -8.7744&],&\\
				&[ &-8.0637&],&\\
				&[ &-8.1532&],&\\
				& [&-16.9030&]],& grad\_fn=<AddBackward0>)\\
\end{array}
$

```python
loss = mnist_loss(preds, train_y[:4])
loss
```

Out: tensor(0.9998, grad_fn=<MeanBackward0>)

Now we can calculate the gradients:

现在我们能够计算梯度：

```
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad
```

Out: (torch.Size([784, 1]), tensor(-0.0001), tensor([-0.0008]))

Let's put that all in a function:

让我们把所有过程放在一个函数中：

```
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```

and test it:

并测试这个函数：

```
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```

Out: (tensor(-5.4066e-05), tensor([-0.0004])) 

But look what happens if we call it twice:

但如果你调用它两次看看会发生什么：

```
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```

Out: tensor(-8.1099e-05), tensor([-0.0006]))

The gradients have changed! The reason for this is that `loss.backward` actually *adds* the gradients of `loss` to any gradients that are currently stored. So, we have to set the current gradients to 0 first:

梯度已经改变了！原因是`loss.backward`实际上*加*了`loss`的梯度给当前存储的梯度。所以，我们必须在一开始设置当前权重为0：

```
weights.grad.zero_()
bias.grad.zero_();
```

> note: Inplace Operations: Methods in PyTorch whose names end in an underscore modify their objects *in place*. For instance, `bias.zero_()` sets all elements of the tensor `bias` to 0.
>
> 注解：原地操作：在PyTorch中的名字结尾是下划线的方法，在*恰当*的地方修改他们的对象。例如，`bias.zero_()`设置张量`bias`的所有元素为0。

Our only remaining step is to update the weights and biases based on the gradient and learning rate. When we do so, we have to tell PyTorch not to take the gradient of this step too—otherwise things will get very confusing when we try to compute the derivative at the next batch! If we assign to the `data` attribute of a tensor then PyTorch will not take the gradient of that step. Here's our basic training loop for an epoch:

我们保保留了基于梯度和学习率更新权重和偏差这个步骤。当我们做这个事情时，我们也要必须告诉PyTorch不要接受梯度这一步骤，否则当你尝试计算下一个批次的导数时，事情就会变的让人迷惑！如果我们给一个张量的`数据`属性赋值，PyTorch会不会采纳梯度这一步骤。这是我们对每个周期的基础训练循环：

```
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```

We also want to check how we're doing, by looking at the accuracy of the validation set. To decide if an output represents a 3 or a 7, we can just check whether it's greater than 0. So our accuracy for each item can be calculated (using broadcasting, so no loops!) with:

通过查看验证集的精度我们也想检查一下做的怎么样。为了判断一个输出代表3还是7，我们只用检查它是不是比0大就可以。所以对每一个数据项我们的精度计算（用传播而不是循环）如下：

```
(preds>0.0).float() == train_y[:4]
```

Out: $
\begin{array} /tensor([&[&False&],&\\
				&[ &True&],&\\
				&[ &True&],&\\
				&[&False&]])\\
\end{array}
$


That gives us this function to calculate our validation accuracy:

下面提供的这个函数来计算我们验证集的精度：

```
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```

We can check it works:

我们能够检查它的工作情况：

```
batch_accuracy(linear1(batch), train_y[:4])
```

Out: tensor(0.5000)

and then put the batches together:

然后把所有批次集中在一起：

```
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```

```
validate_epoch(linear1)
```

Out: 0.5219

That's our starting point. Let's train for one epoch, and see if the accuracy improves:

这是我们的开始点。让我们对一个周期进行训练，并看精度是否有所改善：

```
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
```

Out: 0.6883

Then do a few more:

然后多做几次：

```
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```

Out:  $\begin{array}{rrrrrrrrrr} 
	0.8314& 0.9017& 0.9227& 0.9349& 0.9438& 0.9501 &0.9535 &0.9564& 0.9594& 0.9618 \\
	0.9613& 0.9638& 0.9643 &0.9652& 0.9662& 0.9677& 0.9687& 0.9691 &0.9691& 0.9696 
	\end{array}$

Looking good! We're already about at the same accuracy as our "pixel similarity" approach, and we've created a general-purpose foundation we can build on. Our next step will be to create an object that will handle the SGD step for us. In PyTorch, it's called an *optimizer*.

看起来很好！在我们的“像素相似性”方法中，我们已经有了同样的精度，并且我们已创建一个所能建立的通用目的的基础训练程序。下一步我们会创建一个拥有随机梯度下降步骤的对象。在PyTorch中，它被称为*优化器*。

### Creating an Optimizer

### 创建一个优化器

Because this is such a general foundation, PyTorch provides some useful classes to make it easier to implement. The first thing we can do is replace our `linear` function with PyTorch's `nn.Linear` module. A *module* is an object of a class that inherits from the PyTorch `nn.Module` class. Objects of this class behave identically to standard Python functions, in that you can call them using parentheses and they will return the activations of a model.

因为这是那些通用基础构建，PyTorch提供了一些容易使用且有用类。首先做的事情是我们用PyTorch的`nn.Linear`模块的替换我们的`linear`。一个*模块*是继承了PyTorch对象`nn.Module`的类对象，这个类对象与标准的Python函数表现相同，因此你能用圆括号调用他们并他们会返回一个模型的激活数值。

`nn.Linear` does the same thing as our `init_params` and `linear` together. It contains both the *weights* and *biases* in a single class. Here's how we replicate our model from the previous section:

`nn.Linear`做的事情与我们的`init_params`和`linear`合并在一起做的事情相同。在单一类内它包含了*权重*和*偏差*两部分。这里是我们如何从之前的小节复制我们的模型：

```
linear_model = nn.Linear(28*28,1)
```

Every PyTorch module knows what parameters it has that can be trained; they are available through the `parameters` method:

每个PyTorch模块知道它所能训练的参数，他们通过`parameters`方法能够获得：

```
w,b = linear_model.parameters()
w.shape,b.shape
```

Out: (torch.Size([1, 784]), torch.Size([1]))

We can use this information to create an optimizer:

我们能够利用这个信息来创建一个优化器：

```
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

We can create our optimizer by passing in the model's parameters:

通过传递在模型中的参数，我们能够穿件一个优化器：

```
opt = BasicOptim(linear_model.parameters(), lr)
```

Our training loop can now be simplified to:

我们的训练训练循环现在可简化为：

```
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```

Our validation function doesn't need to change at all:

我们的验证函数完全不需要改变：

```
validate_epoch(linear_model)
```

Out: 0.4157

Let's put our little training loop in a function, to make things simpler:

让我们把小训练循环放在一个函数中，来使得事情更加简洁：

```
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```

The results are the same as in the previous section:

In [ ]:

```
train_model(linear_model, 20)
```

Out:  $\begin{array}{r}
		0.4932 &0.8618 &0.8203& 0.9102& 0.9331& 0.9468& 0.9555& 0.9629& 0.9658& 0.9673 \\
		0.9687 &0.9707 &0.9726 &0.9751 &0.9761 &0.9761 &0.9775 &0.978 &0.9785& 0.9785 
	\end{array}$

fastai provides the `SGD` class which, by default, does the same thing as our `BasicOptim`:

fastai通过默认的方式提供了`SGD`类，做的事情与我们的`BasicOptim`相同：

```
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```

Out:  $\begin{array}{r}
	0.4932 &0.852& 0.8335& 0.9116 &0.9326 &0.9473& 0.9555& 0.9624& 0.9648& 0.9668 \\
	0.9692 &0.9712& 0.9731& 0.9746& 0.9761& 0.9765& 0.9775& 0.978& 0.9785& 0.9785 
\end{array}$

fastai also provides `Learner.fit`, which we can use instead of `train_model`. To create a `Learner` we first need to create a `DataLoaders`, by passing in our training and validation `DataLoader`s:

fastai也提供了`Learner.fit`，我们能够用于替代`train_model`。创建一个`Learner`前我们首先需要通过传递训练和验证`DataLoader`来创建一个`DataLoaders`：

```
dls = DataLoaders(dl, valid_dl)
```

To create a `Learner` without using an application (such as `cnn_learner`) we need to pass in all the elements that we've created in this chapter: the `DataLoaders`, the model, the optimization function (which will be passed the parameters), the loss function, and optionally any metrics to print:

不使用任何网络应用（例如`cnn_learner`）来创建一个`learner`，我们需要给它传送所有本章创建的元素：`DataLoaders`，模型，优化函数（会被传送参数），损失函数和可选的那些指标输出：

```
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```

Now we can call `fit`:

现在我们就能够调用`fit`了：

```
learn.fit(10, lr=lr)
```

<table style="width: 200px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 50px;" align="center">epoch</td>
    <td  style="width: 200px;" align="center">train_loss</td>
    <td  style="width: 200px;" align="center">valid_loss</td>
    <td  style="width: 200px;" align="center">batch_accuracy</td>
    <td  style="width: 200px;" align="center">time</td>
  </tr>
    <td style="width: 100px;" align="center">0</td>
    <td align="right">0.636857</td>
  	<td align="right">0.503549</td>
  	<td align="right">0.495584</td>
  	<td align="right">00:00</td>
  <tr>
    <td style="width: 100px;" align="center">1</td>
    <td align="right">0.545725</td>
  	<td align="right">0.170281</td>
  	<td align="right">0.866045</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">2</td>
    <td align="right">0.199223</td>
  	<td align="right">0.184893</td>
  	<td align="right">0.831207</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">3</td>
    <td align="right">0.086580</td>
  	<td align="right">0.107836</td>
  	<td align="right">0.911187</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">4</td>
    <td align="right">0.045185</td>
  	<td align="right">0.078481</td>
  	<td align="right">0.932777</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">5</td>
    <td align="right">0.029108</td>
  	<td align="right">0.062792</td>
  	<td align="right">0.946516</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">6</td>
    <td align="right">0.022560</td>
  	<td align="right">0.053017</td>
  	<td align="right">0.955348</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">7</td>
    <td align="right">0.019687</td>
  	<td align="right">0.046500</td>
  	<td align="right">0.962218</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">8</td>
    <td align="right">0.018252</td>
  	<td align="right">0.041929</td>
  	<td align="right">0.965162</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">9</td>
    <td align="right">0.017402</td>
  	<td align="right">0.038573</td>
  	<td align="right">0.967615</td>
  	<td align="right">00:00</td>
  </tr>  
</table>		

As you can see, there's nothing magic about the PyTorch and fastai classes. They are just convenient pre-packaged pieces that make your life a bit easier! (They also provide a lot of extra functionality we'll be using in future chapters.)

With these classes, we can now replace our linear model with a neural network.

正如你能看到的，关于PyTorcht和fastai类没有任何魔力。他们只是相对方便的预包装，使得更加容易使用！（他们也提供了很多外部功能，在后续章节我们会用到他们。）

利用这些类，我们现在能够替换我们的神经网络线性模型。

## Adding a Nonlinearity

## 增加一个非线性

So far we have a general procedure for optimizing the parameters of a function, and we have tried it out on a very boring function: a simple linear classifier. A linear classifier is very constrained in terms of what it can do. To make it a bit more complex (and able to handle more tasks), we need to add something nonlinear between two linear classifiers—this is what gives us a neural network.

Here is the entire definition of a basic neural network:

到目前为止对于函数的参数优化我们有了一个通用程序，并且在一个很无聊的函数上（一个简单的线性分类器）我们尝试了一下它。一个线性分类器基于它可做的内容有很大限制。让它稍微更加复杂一些（并能够处理更多任务），我们需要在两个线性分类器之间增加非线性，这就是给我们神经网络的原因。

下面是一个基础神经网络的完整定义：

```
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```

That's it! All we have in `simple_net` is two linear classifiers with a `max` function between them.

就是它！在`simple_net`中我们所拥有的是两个线性分类器之间有一个`max`函数。

Here, `w1` and `w2` are weight tensors, and `b1` and `b2` are bias tensors; that is, parameters that are initially randomly initialized, just like we did in the previous section:

这里的`w1`和`w2`是权重张量，`b1`和`b2`是偏差张量。正如之前小节我们做的那样，这些参数在最初被随机的初始化：

```
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```

The key point about this is that `w1` has 30 output activations (which means that `w2` must have 30 input activations, so they match). That means that the first layer can construct 30 different features, each representing some different mix of pixels. You can change that `30` to anything you like, to make the model more or less complex.

关键点是`w1`有30个输出激活（这意味着`w2`必须有30个输入激活，这样他们就匹配了）。这代表第一层构建了30个不同的特征，每个代表一些不同像素的混合。你能够改变这个`30`为任何你喜欢的数字，以使得的模型更加或更少的复杂度。

That little function `res.max(tensor(0.0))` is called a *rectified linear unit*, also known as *ReLU*. We think we can all agree that *rectified linear unit* sounds pretty fancy and complicated... But actually, there's nothing more to it than `res.max(tensor(0.0))`—in other words, replace every negative number with a zero. This tiny function is also available in PyTorch as `F.relu`:

`res.max(tensor(0.0))`这个小功能被称为*线性整流函数*，也被称为*ReLU*。我们认为全部接受*线性整流函数*听起来很奇特和复杂...但事实上，相比`res.max(tensor(0.0))`这里并没有增加任何东西。换句话，它会替换每个负值为零。这个小功能在PyTorch中也可获得，名为`F.relu`：

```python
plot_function(F.relu)
```

Out: <img src="/Users/Y.H/Documents/GitHub/smartbook/smartbook/_v_images/relu.png" alt="relu" style="zoom:100%;" />

> J: There is an enormous amount of jargon in deep learning, including terms like *rectified linear unit*. The vast vast majority of this jargon is no more complicated than can be implemented in a short line of code, as we saw in this example. The reality is that for academics to get their papers published they need to make them sound as impressive and sophisticated as possible. One of the ways that they do that is to introduce jargon. Unfortunately, this has the result that the field ends up becoming far more intimidating and difficult to get into than it should be. You do have to learn the jargon, because otherwise papers and tutorials are not going to mean much to you. But that doesn't mean you have to find the jargon intimidating. Just remember, when you come across a word or phrase that you haven't seen before, it will almost certainly turn to be referring to a very simple concept.
>
> 杰：在深度学习中有巨大数量的术语，包括像*线性整流函数*这样的术语。正如我们在本例中看到的，绝大多数术语用一小段代码就能来实现，不会非常复杂。实际上对于学术派为了让他们的论文发表，需要让他们的研究成果尽可能的令人印象深刻和成熟。他们可做的众多方法之一就是引入术语。不幸的是，这产生的后果是本领域本应更容易进入，最终变的更让人恐惧和更困难进入。我必须学习术语，否则文章和指引对你来说将没有多大意义。但这并不意味你必须面对查找术语的恐惧。只要记住，当你遇到一个你以前没有看到过的词或短语时，它几乎很确定的会被转为很简单的参考概念。

The basic idea is that by using more linear layers, we can have our model do more computation, and therefore model more complex functions. But there's no point just putting one linear layout directly after another one, because when we multiply things together and then add them up multiple times, that could be replaced by multiplying different things together and adding them up just once! That is to say, a series of any number of linear layers in a row can be replaced with a single linear layer with a different set of parameters.

基本想法是通过利用更多线性层，我们能让模型做更多的计算，因此模型会有更多复杂函数。但直接在另外一层后面只放一个线性层没有意义，因为当我们把事物乘起来，然后把它们相加多次，能够用通过乘不同的事物然后加它们一次来替换！也就是说，在一行中一系列任意数字的线性层能够被拥有不同参数集合的单一线性层所替换。

But if we put a nonlinear function between them, such as `max`, then this is no longer true. Now each linear layer is actually somewhat decoupled from the other ones, and can do its own useful work. The `max` function is particularly interesting, because it operates as a simple `if` statement.

但如果我们在它们之间放一个非线性函数，例如`max`，然而这就不再成立了。现在每个线性层实际是相互之间是分离的，能够做它们自己有帮助的工作。`max`函数是特别有趣，因为它作为一个简单的`if`语句来运行。

> S: Mathematically, we say the composition of two linear functions is another linear function. So, we can stack as many linear classifiers as we want on top of each other, and without nonlinear functions between them, it will just be the same as one linear classifier.
>
> 西：理论上，我们认为两个线性函数组合是另一个线性函数。所以，我们能够在每个线性分类器上叠加任何数量的线性分类器，并在他们这间不用包含任何非线性函数，它会只会与一个线性分类器相同。

Amazingly enough, it can be mathematically proven that this little function can solve any computable problem to an arbitrarily high level of accuracy, if you can find the right parameters for `w1` and `w2` and if you make these matrices big enough. For any arbitrarily wiggly function, we can approximate it as a bunch of lines joined together; to make it closer to the wiggly function, we just have to use shorter lines. This is known as the *universal approximation theorem*. The three lines of code that we have here are known as *layers*. The first and third are known as *linear layers*, and the second line of code is known variously as a *nonlinearity*, or *activation function*.

足够令人惊讶，它能被理论证明这个小函数能够以任何高的精度水平来解决任意可计算的问题，如果你能够找到对`w1`和`w2`正确的参数且如果你使得这些矩阵足够的大。对任意波动函数，我们能够近似认为它是一束连接在一起的线。我们只需要用更短的线，以使它更接近波动函数。这被称为*通过近似定理*（或万能逼近定理）。这里我们所写的三行代码被称为*层*。第一和第三被称为*线性层*，第二行代码被称为*非线性函数*或*激活函数*。

Just like in the previous section, we can replace this code with something a bit simpler, by taking advantage of PyTorch:

正如在之前小节的内容，我们能够通过利用PyTorch，用一些更简单的代码来替换：

```
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

`nn.Sequential` creates a module that will call each of the listed layers or functions in turn.

`nn.Sequential`创建了一个依次调用所列示的每一层或函数的模块。

`nn.ReLU` is a PyTorch module that does exactly the same thing as the `F.relu` function. Most functions that can appear in a model also have identical forms that are modules. Generally, it's just a case of replacing `F` with `nn` and changing the capitalization. When using `nn.Sequential`, PyTorch requires us to use the module version. Since modules are classes, we have to instantiate them, which is why you see `nn.ReLU()` in this example.

`nn.ReLU`是一个PyTorch模块，它做的事情与`F.relu`完全相同。在一个模型中出现的大多数函数也具有与模型相同的格式。通常来说，它只是用`nn`替换`F`的一个例子，并改变了字母的大小写。当使用`nn.Sequential`时，PyTorch需要我们使用模块的版本。因为模块是一些类，我们必须实例化它们，在本例子中这就是为什么你看到了`nn.ReLU()`了。

Because `nn.Sequential` is a module, we can get its parameters, which will return a list of all the parameters of all the modules it contains. Let's try it out! As this is a deeper model, we'll use a lower learning rate and a few more epochs.

因为`nn.Sequential`是一个模块，你能够获取它的参数，包含返回的所有模块的全部参数的一个列表 。让我们试验一下！在这个深度模型上，我们会用一个更低的学习率和更多的周期。

```python
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```

```python
#hide_output
learn.fit(40, 0.1)
```

<table style="width: 200px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 50px;" align="center">epoch</td>
    <td  style="width: 200px;" align="center">train_loss</td>
    <td  style="width: 200px;" align="center">valid_loss</td>
    <td  style="width: 200px;" align="center">batch_accuracy</td>
    <td  style="width: 200px;" align="center">time</td>
  </tr>
    <td style="width: 100px;" align="center">0</td>
    <td align="right">0.305828</td>
  	<td align="right">0.399663</td>
  	<td align="right">0.508341</td>
  	<td align="right">00:00</td>
  <tr>
    <td style="width: 100px;" align="center">1</td>
    <td align="right">0.142960</td>
  	<td align="right">0.225702</td>
  	<td align="right">0.807655</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">2</td>
    <td align="right">0.079516</td>
  	<td align="right">0.113519</td>
  	<td align="right">0.919529</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">3</td>
    <td align="right">0.052391</td>
  	<td align="right">0.076792</td>
  	<td align="right">0.943081</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">4</td>
    <td align="right">0.039796</td>
  	<td align="right">0.060083</td>
  	<td align="right">0.956330</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">5</td>
    <td align="right">0.033368</td>
  	<td align="right">0.050713</td>
  	<td align="right">0.963690</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">6</td>
    <td align="right">0.029680</td>
  	<td align="right">0.044797</td>
  	<td align="right">0.965653</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">7</td>
    <td align="right">0.027290</td>
  	<td align="right">0.040729</td>
  	<td align="right">0.968106</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">8</td>
    <td align="right">0.025568</td>
  	<td align="right">0.037771</td>
  	<td align="right">0.968597</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">9</td>
    <td align="right">0.024233</td>
  	<td align="right">0.035508</td>
  	<td align="right">0.970559</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">10</td>
    <td align="right">0.023149</td>
  	<td align="right">0.033714</td>
  	<td align="right">0.972031</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">11</td>
    <td align="right">0.022242</td>
  	<td align="right">0.032243</td>
  	<td align="right">0.972522</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">12</td>
    <td align="right">0.021468</td>
  	<td align="right">0.031006</td>
  	<td align="right">0.973503</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">13</td>
    <td align="right">0.020796</td>
  	<td align="right">0.029944</td>
  	<td align="right">0.974485</td>
  	<td align="right">00:00</td>
  </tr>  
  <tr>
    <td style="width: 100px;" align="center">14</td>
    <td align="right">0.020207</td>
  	<td align="right">0.029016</td>
  	<td align="right">0.975466</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">15</td>
    <td align="right">0.019683</td>
  	<td align="right">0.028196</td>
  	<td align="right">0.976448</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">16</td>
    <td align="right">0.019215</td>
  	<td align="right">0.027463</td>
  	<td align="right">0.976448</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">17</td>
    <td align="right">0.018791</td>
  	<td align="right">0.026806</td>
  	<td align="right">0.976938</td>
  	<td align="right">00:00</td>
  </tr>
  <tr>
    <td style="width: 100px;" align="center">18</td>
    <td align="right">0.018405</td>
  	<td align="right">0.026212</td>
  	<td align="right">0.977920</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">19</td>
    <td align="right">0.018051</td>
  	<td align="right">0.025671</td>
  	<td align="right">0.977920</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">20</td>
    <td align="right">0.017725</td>
  	<td align="right">0.025179</td>
  	<td align="right">0.977920</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">21</td>
    <td align="right">0.017422</td>
  	<td align="right">0.024728</td>
  	<td align="right">0.978410</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">22</td>
    <td align="right">0.017141</td>
  	<td align="right">0.024313</td>
  	<td align="right">0.978901</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">23</td>
    <td align="right">0.016878</td>
  	<td align="right">0.023932</td>
  	<td align="right">0.979392</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">24</td>
    <td align="right">0.016632</td>
  	<td align="right">0.023580</td>
  	<td align="right">0.979882</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">25</td>
    <td align="right">0.016400</td>
  	<td align="right">0.023254</td>
  	<td align="right">0.979882</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">26</td>
    <td align="right">0.016181</td>
  	<td align="right">0.022952</td>
  	<td align="right">0.979882</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">27</td>
    <td align="right">0.015975</td>
  	<td align="right">0.022672</td>
  	<td align="right">0.980864</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">28</td>
    <td align="right">0.015779</td>
  	<td align="right">0.022411</td>
  	<td align="right">0.980864</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">29</td>
    <td align="right">0.015593</td>
  	<td align="right">0.022168</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">30</td>
    <td align="right">0.015417</td>
  	<td align="right">0.021941</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">31</td>
    <td align="right">0.015249</td>
  	<td align="right">0.021728</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">32</td>
    <td align="right">0.015088</td>
  	<td align="right">0.021529</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">33</td>
    <td align="right">0.014935</td>
  	<td align="right">0.021341</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">34</td>
    <td align="right">0.014788</td>
  	<td align="right">0.021164</td>
  	<td align="right">0.981845</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">35</td>
    <td align="right">0.014647</td>
  	<td align="right">0.020998</td>
  	<td align="right">0.982336</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">36</td>
    <td align="right">0.014512</td>
  	<td align="right">0.020840</td>
  	<td align="right">0.982826</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">37</td>
    <td align="right">0.014382</td>
  	<td align="right">0.020691</td>
  	<td align="right">0.982826</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">38</td>
    <td align="right">0.014257</td>
  	<td align="right">0.020550</td>
  	<td align="right">0.982826</td>
  	<td align="right">00:00</td>
  </tr> 
  <tr>
    <td style="width: 100px;" align="center">39</td>
    <td align="right">0.014136</td>
  	<td align="right">0.020415</td>
  	<td align="right">0.982826</td>
  	<td align="right">00:00</td>
  </tr> 
</table>	

We're not showing the 40 lines of output here to save room; the training process is recorded in `learn.recorder`, with the table of output stored in the `values` attribute, so we can plot the accuracy over training as:

为了节省空间我们没有在这里显示输出的40行信息。训练过程被记录在`learn.recorder`中，利用输出表存储在`values`属性中，我们能够描绘训练过程的精度为：

```
plt.plot(L(learn.recorder.values).itemgot(2));
```

Out: ![img](data:image/png;base64,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) 

And we can view the final accuracy:

我们能够查看最终的精度：

```
learn.recorder.values[-1][2]
```

Out: 0.982826292514801

At this point we have something that is rather magical:

1. A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters
2. A way to find the best set of parameters for any function (stochastic gradient descent)

This is why deep learning can do things which seem rather magical such fantastic things. Believing that this combination of simple techniques can really solve any problem is one of the biggest steps that we find many students have to take. It seems too good to be true—surely things should be more difficult and complicated than this? Our recommendation: try it out! We just tried it on the MNIST dataset and you have seen the results. And since we are doing everything from scratch ourselves (except for calculating the gradients) you know that there is no special magic hiding behind the scenes.

在这一点上我们有个比较奇幻的事情：

1. 一个函数在给定正确参数集的情况下，能够以任何精度水平来解决任何问题（神经网络）
2. 为任何函数寻找最优参数集的一个方法（随机梯度下降）

这就是为什么深度学习能够做的事情好像有点奇幻，如此奇妙。相信这个能够解决任何问题的简单技术组合，是我们发现许多学生不得不采纳的最大步骤之一。它似乎太好了以至于感觉不真实，事实上应该比这个更困难和更复杂吗？我们的建议：试验它！我们刚刚在MNIST数据集上尝试了它，并且我们看到了结果。自从我们从零开始自己做了每一个事情（除了计算梯度），我们知道在这个场景背后没有隐藏特殊的魔法。

### Going Deeper

### 继续深入

There is no need to stop at just two linear layers. We can add as many as we want, as long as we add a nonlinearity between each pair of linear layers. As you will learn, however, the deeper the model gets, the harder it is to optimize the parameters in practice. Later in this book you will learn about some simple but brilliantly effective techniques for training deeper models.

不需要止步于仅仅两个线性层。只要我们在每对线性层之间增加一个非线性函数，我们就能够增加任何我们想加的层数量。然而，正如我们将要学到的，在实践中，更深的模型，它的参数优化就更困难。后续在本书我们会学到关于深度模型的训练一些简单但巧妙有效的技术。

We already know that a single nonlinearity with two linear layers is enough to approximate any function. So why would we use deeper models? The reason is performance. With a deeper model (that is, one with more layers) we do not need to use as many parameters; it turns out that we can use smaller matrices with more layers, and get better results than we would get with larger matrices, and few layers.

我们已经知道一个非线性函数加两个线性层足以近似任何函数。那么我们为什么还要用更深的模型？原因是性能。一个更深的模型（有更多的层），我们不需要使用太多的参数。事实证明有更多的层我们能够使用更小的矩阵，并且取得的结果也比更少的层且更大的矩阵更好。

That means that we can train the model more quickly, and it will take up less memory. In the 1990s researchers were so focused on the universal approximation theorem that very few were experimenting with more than one nonlinearity. This theoretical but not practical foundation held back the field for years. Some researchers, however, did experiment with deep models, and eventually were able to show that these models could perform much better in practice. Eventually, theoretical results were developed which showed why this happens. Today, it is extremely unusual to find anybody using a neural network with just one nonlinearity.

这就意味着我们能够更快的训练醋，并且它也会占用更少的内存。在1990年代，研究人员如此关注通用近似原理，以至于很少有人用超过一个非线性函数做实验。这个不是特别基础的理论阻止这一领域好多年。然而，有一些研究人员在实践中利用深度模型实验并最终能够显示出这些模型能够表现更好。最终发展出的理论结果显示了为什么会发生的原因。如今，发现有人只使用一个非线性函数的神经网络是极为不寻常的。

Here what happens when we train an 18-layer model using the same approach we saw in <chapter_intro>:

在这里当我们利用在<章节：概述>中看到的相同方法训练一个18层模型的时候，看会发生什么：

```
dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```

<table style="width: 200px;border-collapse: collapse;" >
  <tr>
    <td  style="width: 50px;" align="center">epoch</td>
    <td  style="width: 200px;" align="center">train_loss</td>
    <td  style="width: 200px;" align="center">valid_loss</td>
    <td  style="width: 200px;" align="center">batch_accuracy</td>
    <td  style="width: 200px;" align="center">time</td>
  </tr>
    <td style="width: 100px;" align="center">0</td>
    <td align="right">0.082089</td>
  	<td align="right">0.009578</td>
  	<td align="right">0.997056</td>
  	<td align="right">00:11</td>
  <tr>
</table>

Nearly 100% accuracy! That's a big difference compared to our simple neural net. But as you'll learn in the remainder of this book, there are just a few little tricks you need to use to get such great results from scratch yourself. You already know the key foundational pieces. (Of course, even once you know all the tricks, you'll nearly always want to work with the pre-built classes provided by PyTorch and fastai, because they save you having to think about all the little details yourself.)

近乎100%的精度！相比我们的简单神经网络这是一个巨大的差异。但正如你在本书剩余部分学到的内容，这些只是一些小技巧。你自己在从零开始的时候，需要使用它们来获得如此好的结果。你已经学到了关键基础部分。（当然，一旦你知道了所有技巧，你就几乎会一直想用PyTorch和fastai提供的预建类来工作，因为它们确保你不必自己想所有的小细节。）

## Jargon Recap

## 术语总结

Congratulations: you now know how to create and train a deep neural network from scratch! We've gone through quite a few steps to get to this point, but you might be surprised at how simple it really is.

恭喜，你现在知道如何从零开始创建并训练一个深度神经网络了！我们经历了完整的几步到达现在这个程度，但你可能惊讶于它实际上是如此简单。

Now that we are at this point, it is a good opportunity to define, and review, some jargon and key concepts.

A neural network contains a lot of numbers, but they are only of two types: numbers that are calculated, and the parameters that these numbers are calculated from. This gives us the two most important pieces of jargon to learn:

- Activations:: Numbers that are calculated (both by linear and nonlinear layers)
- Parameters:: Numbers that are randomly initialized, and optimized (that is, the numbers that define the model)

现在我们达到了这个程度，这是一个来定义和复查一些术语和关键概念的好时机。

一个神经网络包含大量数值，但它们只有两种类型：被计算得出数值和那些被计算得出的数值所依赖的参数。这就给了我们要学习的两个最重要的术语：

- 激活：被计算得出的数值（通过线性和非线性层两者）
- 参数：被随机初始化的数值，并被优化（即，定义模型的数值）

We will often talk in this book about activations and parameters. Remember that they have very specific meanings. They are numbers. They are not abstract concepts, but they are actual specific numbers that are in your model. Part of becoming a good deep learning practitioner is getting used to the idea of actually looking at your activations and parameters, and plotting them and testing whether they are behaving correctly.

在本书我们会经常谈到激活和参数。记住它们有很特殊的含义。它们是数值。它们不是抽象的概念，在你的模型内它们实际上是具体的数值。成为一个好的深度学习从业者的必要素质是要习惯于保持实际看你的激活和参数的想法，并绘制它们及测试它们的行为是否正确。

Our activations and parameters are all contained in *tensors*. These are simply regularly shaped arrays—for example, a matrix. Matrices have rows and columns; we call these the *axes* or *dimensions*. The number of dimensions of a tensor is its *rank*. There are some special tensors:

- Rank zero: scalar
- Rank one: vector
- Rank two: matrix

我们的激活和参数都被容纳在*张量*中 。有一些简单的形成数组的规律：如一个矩阵。矩阵有行和列，我们称其为*坐标轴*和*维度*。一个张量维度数是它的*阶*。这里有一些特定的张量：

- 零阶：标量
- 一阶：向量
- 二阶：矩阵

A neural network contains a number of layers. Each layer is either *linear* or *nonlinear*. We generally alternate between these two kinds of layers in a neural network. Sometimes people refer to both a linear layer and its subsequent nonlinearity together as a single layer. Yes, this is confusing. Sometimes a nonlinearity is referred to as an *activation function*.

<dljargon1> summarizes the key concepts related to SGD.

一个神经网络包含许多层。每层要么是*线性*要么是*非线性*。在一个神经网络中我们通常在些两种层类型间轮替。有时候人们认为一个线性层和它随后非线性函数两者合并在一起视为一个层。是的，这让人感到迷惑。有时候一个非线性函数被称为*激活函数*。

| Term | Meaning|
| ----------------------| --------------------------------------- |
|ReLU | Function that returns 0 for negative numbers and doesn't change positive numbers.|
|Mini-batch | A smll group of inputs and labels gathered together in two arrays. A gradient descent step is updated on this batch (rather than a whole epoch).|
|Forward pass | Applying the model to some input and computing the predictions.|
|Loss | A value that represents how well (or badly) our model is doing.|
|Gradient | The derivative of the loss with respect to some parameter of the model.|
|Backard pass | Computing the gradients of the loss with respect to all model parameters.|
|Gradient descent | Taking a step in the directions opposite to the gradients to make the model parameters a little bit better.|
|Learning rate | The size of the step we take when applying SGD to update the parameters of the model.|


> note: *Choose Your Own Adventure* Reminder: Did you choose to skip over chapters 2 & 3, in your excitement to peek under the hood? Well, here's your reminder to head back to chapter 2 now, because you'll be needing to know that stuff very soon!

| 术语                       | 含义                                                         |
| -------------------------- | ------------------------------------------------------------ |
| 线性整流函数(ReLU)         | 对负数返回另，对正数不做任何改变。                           |
| 最小批次(Mini-batch)       | 在两个数组中输入和标签聚合在一起的小数据包。一个梯度下降步骤是在这个小批次上的更新（而不是整个周期）。 |
| 顺推法(Forward pass)       | 应用模型根据一些输入并计算预测。                             |
| 损失(Loss)                 | 代表我们正在运转的模型怎样的好（或不好）的一个值。           |
| 梯度(Gradient)             | 关于模型一些参数的损失导数。                                 |
| 逆推法(Backard pass)       | 计算关于全部模型参数的损失梯度。                             |
| 梯度下降(Gradient descent) | 梯度相反方向的步进，使得模型参数变的稍微更好一些。           |
| 学习率(Learning rate)      | 当应用随机梯度下降来更新模型的参数时，我们所采取的步进大小。 |

> 注解：提示*选择你自己的冒险*：你是否出于你兴奋瞥一眼深层次内容而选择略过第二章节和第三章节？好吧，现在在这里提示你返回第二章节，因为你需要很快就要知道这个内容！

## Questionnaire

## 练习题

1. How is a grayscale image represented on a computer? How about a color image?
2. 在计算机里一张灰度图是怎样展示的？图片的颜色是怎样的？
3. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
4. 在`MNIST_SAMPLE`数据集中文件和目录的结构是怎样的？为什么？
5. Explain how the "pixel similarity" approach to classifying digits works.
6. 解释“像素相似性”方法来分类数字是怎样工作的。
7. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
8. 列表生成器是什么？现在创建一个，从列表中选择奇数并对它们加倍。
9. What is a "rank-3 tensor"?
10. 什么是一个“三阶张向”？
11. What is the difference between tensor rank and shape? How do you get the rank from the shape?
12. 张量的阶和形状之间的区别是什么？从形状中你如何获得阶？
13. What are RMSE and L1 norm?
14. RMSE和L1正则是什么？
15. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
16. 你如何能够同时对数以千计的数值上进行计算，faster比Python循环快成千上万倍吗？
17. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
18. 穿件一个3×3的张量或数组，数值范围从1 到9并对它们进行加倍处理。选择右下部的4个数值。
19. What is broadcasting?
20. 什么是传播？
21. Are metrics generally calculated using the training set, or the validation set? Why?
22. 通常使用训练集或验证集计算指标吗？为什么？
23. What is SGD?
24. 什么是随机梯度下降？
25. Why does SGD use mini-batches?
26. 为什么用最小批次来做梯度下降？
27. What are the seven steps in SGD for machine learning?
28. 对于机器深度随机梯度下降中的7个步骤是什么？
29. How do we initialize the weights in a model?
30. 在一个模型中我们如何初始化权重？
31. What is "loss"?
32. 什么是“损失”？
33. Why can't we always use a high learning rate?
34. 为什么我们不能一直使用高学习率？
35. What is a "gradient"?
36. 什么是“梯度”？
37. Do you need to know how to calculate gradients yourself?
38. 你需要知道如何自己来计算梯度吗？
39. Why can't we use accuracy as a loss function?
40. 为什么我们不能使用精度作为损失函数？
41. Draw the sigmoid function. What is special about its shape?
42. 画出sigmoid函数曲线。它的形状具体是什么样子？
43. What is the difference between a loss function and a metric?
44. 损失函数和指标之间的差异是什么？
45. What is the function to calculate new weights using a learning rate?
46. 使用一个学习率计算新权重的函数是什么？
47. What does the `DataLoader` class do?
48. `DataLoader`类做了什么？
49. Write pseudocode showing the basic steps taken in each epoch for SGD.
50. 编写伪代码，来展示对随机梯度下降每个周期中所采纳的基础步骤。
51. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
52. 创建一个函数，如果传递两个参数`[1,2,3,4]`和`abcd`，并返回`[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`。数据的数据结构具体是什么样的？
53. What does `view` do in PyTorch?
54. 在PyTorch中`view`做了什么？
55. What are the "bias" parameters in a neural network? Why do we need them?
56. 在一个神经网络中“变差”参数是什么？为什么我们需要他们？
57. What does the `@` operator do in Python?
58. 在Python中操作符`@`做了什么工作？
59. What does the `backward` method do?
60. `backward`方法做了什么工作？
61. Why do we have to zero the gradients?
62. 为什么我们必须零化梯度？
63. What information do we have to pass to `Learner`?
64. 我们必须传递给`Learner`什么信息？
65. Show Python or pseudocode for the basic steps of a training loop.
66. 对于训练循环的基础步骤展示Python或伪代码。
67. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
68. 什么是“ReLU”？对于从`-2` 到 `+2`的值绘制一个图。
69. What is an "activation function"?
70. 什么是“激活函数”？
71. What's the difference between `F.relu` and `nn.ReLU`?
72.  `F.relu` 和 `nn.ReLU`之间的差异是什么？
73. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
74. 通用近似原理展示了只用一个非线性函数就能根据需要尽可能的近似任何函数。那么，通常情况下我们为什么会大量使用？

### Further Research

### 深入研究

1. Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.
2. 基于本章节所展示的训练循环，从零开始创建你自己的`Learner`的实践。
3. Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.
4. 利用完整的MNIST数据集完成本章节的全部步骤（即，对所有数字，而不仅仅对3和7）。这是一个重要项目，会花费你相当多的时间来完成！你将需要做一些你自己的研究，以想出如何解决你在做这个项目过程中所遇到的一些障碍。

