# Training a State-of-the-Art Model

# 训练一个最先进的模型

This chapter introduces more advanced techniques for training an image classification model and getting state-of-the-art results. You can skip it if you want to learn more about other applications of deep learning and come back to it later—knowledge of this material will not be assumed in later chapters.

这一章介绍的是一些用于训练图像分类模型和取得最新成果的更高级的技术。如果你希望学习更多有关于机器学习的其它应用你可以略过它，稍后再回来看即可，这个知识内容在后面的章节不会展示。

We will look at what normalization is, a powerful data augmentation technique called mixup, the progressive resizing approach and test time augmentation. To show all of this, we are going to train a model from scratch (not using transfer learning) using a subset of ImageNet called [Imagenette](https://github.com/fastai/imagenette). It contains a subset of 10 very different categories from the original ImageNet dataset, making for quicker training when we want to experiment.

我们会看一下什么是标准化，称为mixup的一种强大的数据增强技术，渐进式的调整大小方法和测试时间增强。我们会使用一个名为[Imagenette](https://github.com/fastai/imagenette)的ImageNet的子集，从零开始训练一个模型来展示这些技术。这个数据集是包含了来自原生ImageNet数据集非常不同的10个分类的子数据集，当我们希望跑实验时在这个子数据集上会训练的更快。

This is going to be much harder to do well than with our previous datasets because we're using full-size, full-color images, which are photos of objects of different sizes, in different orientations, in different lighting, and so forth. So, in this chapter we're going to introduce some important techniques for getting the most out of your dataset, especially when you're training from scratch, or using transfer learning to train a model on a very different kind of dataset than the pretrained model used.

相比我们之前的数据集这个会更难做好，因为我们使用了全尺寸和全彩色图像，这些是在不同的方位、不同的光线等条件下，不同尺寸对象的照片。所以在本章我们会介绍充分利用你的数据集一些重点技术，尤其在你从零开始训练模型时，或利用迁移学习来训练一个模型，所使用的数据集种类与预训练模型所使用的完全不同。

