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