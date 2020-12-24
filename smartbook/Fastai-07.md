# Training a State-of-the-Art Model

# 训练一个最先进的模型

This chapter introduces more advanced techniques for training an image classification model and getting state-of-the-art results. You can skip it if you want to learn more about other applications of deep learning and come back to it later—knowledge of this material will not be assumed in later chapters.

这一章介绍的是一些用于训练图像分类模型和取得最新成果的更高级的技术。如果你希望学习更多有关于机器学习的其它应用你可以略过它，稍后再回来看即可，这个知识内容在后面的章节不会展示。

We will look at what normalization is, a powerful data augmentation technique called mixup, the progressive resizing approach and test time augmentation. To show all of this, we are going to train a model from scratch (not using transfer learning) using a subset of ImageNet called [Imagenette](https://github.com/fastai/imagenette). It contains a subset of 10 very different categories from the original ImageNet dataset, making for quicker training when we want to experiment.

我们会看一下什么是标准化，称为mixup的一种强大的数据增强技术，渐进式的调整大小方法和测试时间增强。我们会使用一个名为[Imagenette](https://github.com/fastai/imagenette)的ImageNet的子集，从零开始训练一个模型来展示这些技术。这个数据集是包含了来自原生ImageNet数据集非常不同的10个分类的子数据集，当我们希望跑实验时在这个子数据集上会训练的更快。

This is going to be much harder to do well than with our previous datasets because we're using full-size, full-color images, which are photos of objects of different sizes, in different orientations, in different lighting, and so forth. So, in this chapter we're going to introduce some important techniques for getting the most out of your dataset, especially when you're training from scratch, or using transfer learning to train a model on a very different kind of dataset than the pretrained model used.

相比我们之前的数据集这个会更难做好，因为我们使用了全尺寸和全彩色图像，这些是在不同的方位、不同的光线等条件下，不同尺寸对象的照片。所以在本章我们会介绍充分利用你的数据集一些重点技术，尤其在你从零开始训练模型时，或利用迁移学习来训练一个模型，所使用的数据集种类与预训练模型所使用的完全不同。

## Imagenette

## Imagenette数据集

When fast.ai first started there were three main datasets that people used for building and testing computer vision models:

- ImageNet:: 1.3 million images of various sizes around 500 pixels across, in 1,000 categories, which took a few days to train
- MNIST:: 50,000 28×28-pixel grayscale handwritten digits
- CIFAR10:: 60,000 32×32-pixel color images in 10 classes

当fast.ai首次启动时，有三个主要数据集，人们用于创建和测试计算机视觉模型：

- ImageNet：有一千个分类，一百三十万张像素大约为500的各种尺寸的图像，它会耗费几天的时间来训练
- MNIST：五万张，28×28像素的灰度手写体数字
- CIFAR10：六万张，32×32像素有十个种类的彩色图像

The problem was that the smaller datasets didn't actually generalize effectively to the large ImageNet dataset. The approaches that worked well on ImageNet generally had to be developed and trained on ImageNet. This led to many people believing that only researchers with access to giant computing resources could effectively contribute to developing image classification algorithms.

问题是，小的数据集不能真正有效的泛华到大型ImageNet数据集上。在ImageNet上工作良好的方法通常是必须在ImageNet上开发和训练。这就导致很多人相信只有有权限使用巨型计算资源的研究人员能够有效的为图像分类算法做发展贡献。

We thought that seemed very unlikely to be true. We had never actually seen a study that showed that ImageNet happen to be exactly the right size, and that other datasets could not be developed which would provide useful insights. So we thought we would try to create a new dataset that researchers could test their algorithms on quickly and cheaply, but which would also provide insights likely to work on the full ImageNet dataset.

我们认为这好像极不可能是真实的。我们从来没有实际看到过一个研究显示ImageNet是一个绝对正确的尺寸，而不能开发出可提供有益见解的其它数据集。所以我们认为我们也许要尝试创建一个让研究人员快速且廉价测试他们算法的数据集，但它也能提供可工作在全部ImageNet数据集上的见解。

About three hours later we had created Imagenette. We selected 10 classes from the full ImageNet that looked very different from one another. As we had hoped, we were able to quickly and cheaply create a classifier capable of recognizing these classes. We then tried out a few algorithmic tweaks to see how they impacted Imagenette. We found some that worked pretty well, and tested them on ImageNet as well—and we were very pleased to find that our tweaks worked well on ImageNet too!

大约3个小时后我们创建了Imagenette。我们从全尺寸ImageNet里选择了看起来完全不同的10个类别。正如我们期望的，我们能够快速且廉价的创建一个具备识别它些类别的分类器。然后我们尝试一些算法微调来看它会如何影响Imagenette。我们发现有一些运行的相当好，且在ImageNet上测试他们也相当不错，非常高兴发现我们的微调工作在ImageNet上也运行良好！

There is an important message here: the dataset you get given is not necessarily the dataset you want. It's particularly unlikely to be the dataset that you want to do your development and prototyping in. You should aim to have an iteration speed of no more than a couple of minutes—that is, when you come up with a new idea you want to try out, you should be able to train a model and see how it goes within a couple of minutes. If it's taking longer to do an experiment, think about how you could cut down your dataset, or simplify your model, to improve your experimentation speed. The more experiments you can do, the better!

Let's get started with this dataset:

这里有一个很重要的信息：你得到的数据集未必是你想要的。它尤其不可能是你想要做你的开发和原型的数据集。你的目标应该是迭代速度不能超过几分钟，也就是说，当你提出一个希望尝试一下的新想法，你应该能够训练一个模型，在几分钟内查看它的情况是怎样的。如果花费太长的时间做这个实验，思考一下你能够如何减小你的数据集或简化你的模型，来改善你的实验速度。你能够更多的实验是更好的！

让我们利用这个数据集开始吧：

```
from fastai.vision.all import *
path = untar_data(URLs.IMAGENETTE)
```

First we'll get our dataset into a `DataLoaders` object, using the *presizing* trick introduced in <chapter_pet_breeds>:

首先我们会把我们的数据集放入一个`DataLoaders`对象，这使用的是在<章节：基于宠物品种的图像分类>中介绍的*填孔处理*技巧：

```
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = dblock.dataloaders(path, bs=64)
```

and do a training run that will serve as a baseline:

做一个训练，并做为基准：

```
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   1.583403 |   2.064317 | 0.401792 | 01:03 |
|     1 |   1.208877 |   1.260106 | 0.601568 | 01:02 |
|     2 |   0.925265 |   1.036154 | 0.664302 | 01:03 |
|     3 |   0.730190 |   0.700906 | 0.777819 | 01:03 |
|     4 |   0.585707 |   0.541810 | 0.825243 | 01:03 |

That's a good baseline, since we are not using a pretrained model, but we can do better. When working with models that are being trained from scratch, or fine-tuned to a very different dataset than the one used for the pretraining, there are some additional techniques that are really important. In the rest of the chapter we'll consider some of the key approaches you'll want to be familiar with. The first one is *normalizing* your data.

这是一个好的基准模型，因为我们没有使用预训练模型，但做的更好。当使用的模型是从零训练的，或微调与预训练使用的数据集完全不同时，有一些非常重要的附加技术。在剩下的章节，我们会思考一些你希望精通的关键方法。第一个技术是*规范化*你的数据。

