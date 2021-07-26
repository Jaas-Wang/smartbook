# Application Architectures Deep Dive

# 应用架构深入研究

We are now in the exciting position that we can fully understand the architectures that we have been using for our state-of-the-art models for computer vision, natural language processing, and tabular analysis. In this chapter, we're going to fill in all the missing details on how fastai's application models work and show you how to build the models they use.

现在我们兴奋于对于机器视觉、自然语言处理和表格分析等，我们能够全部理解所使用的先进的模型架构。在本章，我们会补齐fastai应用模型如何工作的所有细节及展示他们所使用的模型是如何创建的。

We will also go back to the custom data preprocessing pipeline we saw in <chapter_midlevel_data> for Siamese networks and show you how you can use the components in the fastai library to build custom pretrained models for new tasks.

We'll start with computer vision.

我们也会返回到在<章节：用中级API做数据整理>中我们看到的自定义数据预处理管道与展示我们如何使用fastai库中的组件为新任务创建自定义预训练模型。

我们会从计算机视觉开始。

## Computer Vision

## 计算机视觉

For computer vision application we use the functions `cnn_learner` and `unet_learner` to build our models, depending on the task. In this section we'll explore how to build the `Learner` objects we used in Parts 1 and 2 of this book.

根据任务，对于计算机视觉应用，我们使用`cnn_learner`和`unet_learner`来创建我们的模型。在本部门我们会探索使用本书的第一部分和第二部分来如何创建`Learner`对象。

### cnn_learner

### cnn_learner

Let's take a look at what happens when we use the `cnn_learner` function. We begin by passing this function an architecture to use for the *body* of the network. Most of the time we use a ResNet, which you already know how to create, so we don't need to delve into that any further. Pretrained weights are downloaded as required and loaded into the ResNet.

让我们看一下，当使用`cnn_learner`函数时会发生什么。我们首先传递这个函数，用于网络*主体*的一个架构。大多时间我们使用ResNet，这个模型我们已经知道如何创建，所以我们不需要做任何进一步研究。预训练权重需要下载并加载到ResNet中。

Then, for transfer learning, the network needs to be *cut*. This refers to slicing off the final layer, which is only responsible for ImageNet-specific categorization. In fact, we do not slice off only this layer, but everything from the adaptive average pooling layer onwards. The reason for this will become clear in just a moment. Since different architectures might use different types of pooling layers, or even completely different kinds of *heads*, we don't just search for the adaptive pooling layer to decide where to cut the pretrained model. Instead, we have a dictionary of information that is used for each model to determine where its body ends, and its head starts. We call this `model_meta`—here it is for resnet-50:

然后，为了迁移学习，网络需要被*裁剪*。这指的是却掉最后的层，它只是负责ImageNet特定的分类。实际上，我们不仅仅剪切掉这一层，而是从自适应平均池化层开始的一切内容。这样处理的原因稍等一会就会清楚了。因为不同的架构可能使用不同类型的池化层，或甚至使用完全不同类型的*模型头*，我们不仅仅搜索自适应池化层来决定从哪里裁剪预训练模型。相反，我们有一个信息字典，用于搜索模型来决定其模型体的结束位置，和它的模型头的开始位置。我们称此为`model_meta`，下面是对resnet-50操作：

代码：

```
model_meta[resnet50]
```

输出结果:

```
{'cut': -2,
 'split': <function fastai.vision.learner._resnet_split(m)>,
 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
```

> jargon: Body and Head: The "head" of a neural net is the part that is specialized for a particular task. For a CNN, it's generally the part after the adaptive average pooling layer. The "body" is everything else, and includes the "stem" (which we learned about in <chapter_resnet>).

> 术语：模型体和模型头：神经网络的“头”是专门针对一个特定任务的部分。对于卷积神经网络，它通常是自适应平均池化层后的部分。“体”是其它的所有内容，也包含“stem”（这是我们在<第十四章：残差网络>中学到的）。

