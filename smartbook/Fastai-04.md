

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