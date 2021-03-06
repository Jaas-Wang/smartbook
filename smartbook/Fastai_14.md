# ResNets

# 残差网络

In this chapter, we will build on top of the CNNs introduced in the previous chapter and explain to you the ResNet (residual network) architecture. It was introduced in 2015 by Kaiming He et al. in the article ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) and is by far the most used model architecture nowadays. More recent developments in image models almost always use the same trick of residual connections, and most of the time, they are just a tweak of the original ResNet.

这一章节，我们是在上一章节介绍的CNN之上建设的，并向你解释残差网络（ResNet）架构。这个架构在2015年由何凯明等人编写的文章[“面向图像识别的深度残差学习”](https://arxiv.org/abs/1512.03385)中引入的，现如今这是目前为止最常用的模型架构。图像模型最新发展总会使用相同的残差连接技巧，大多数时间他们只会对原始ResNet做微调。

We will first show you the basic ResNet as it was first designed, then explain to you what modern tweaks make it more performant. But first, we will need a problem a little bit more difficult than the MNIST dataset, since we are already close to 100% accuracy with a regular CNN on it.

我会首先给你展示最初设计的基础ResNet，然后向你解释哪些最新调整使得这个架构更加高效。但首先，我们需要一个比MNIST数据集更有难度的问题，因为我们用一个常规CNN在这些基础问题上已经接近了100%的精度。