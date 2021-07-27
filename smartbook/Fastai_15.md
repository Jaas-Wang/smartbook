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

然后，为了迁移学习，网络需要被*裁剪*。这指的是切掉最后的层，它只是负责ImageNet特定的分类。实际上，我们不仅仅剪切掉这一层，而是从自适应平均池化层开始的一切内容。这样处理的原因稍等一会就会清楚了。因为不同的架构可能使用不同类型的池化层，或甚至使用完全不同类型的*模型头*，我们不仅仅搜索自适应池化层来决定从哪里裁剪预训练模型。相反，我们有一个信息字典，用于搜索模型来决定其模型体的结束位置，和它的模型头的开始位置。我们称此为`model_meta`，下面是对resnet-50操作：

实验代码：

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

If we take all of the layers prior to the cut point of `-2`, we get the part of the model that fastai will keep for transfer learning. Now, we put on our new head. This is created using the function `create_head`:

如果我们剪切 `-2` 这个点之前的所有层，我们获取的模型部分，fastai会保留下来做迁移学习。现在我们创建了新的模型头。这是使用`create_head`函数来创建的：

实验代码:

```
#hide_output
create_head(20,2)
```

输出结果:

```
Sequential(
  (0): AdaptiveConcatPool2d(
    (ap): AdaptiveAvgPool2d(output_size=1)
    (mp): AdaptiveMaxPool2d(output_size=1)
     )
  (1): full: False
  (2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Dropout(p=0.25, inplace=False)
  (4): Linear(in_features=20, out_features=512, bias=False)
  (5): ReLU(inplace=True)
  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): Dropout(p=0.5, inplace=False)
  (8): Linear(in_features=512, out_features=2, bias=False)
  )
Sequential(
  (0): AdaptiveConcatPool2d(
    (ap): AdaptiveAvgPool2d(output_size=1)
    (mp): AdaptiveMaxPool2d(output_size=1)
     )
  (1): Flatten()
  (2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
  (3): Dropout(p=0.25, inplace=False)
  (4): Linear(in_features=20, out_features=512, bias=False)
  (5): ReLU(inplace=True)
  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
  (7): Dropout(p=0.5, inplace=False)
  (8): Linear(in_features=512, out_features=2, bias=False)
  )
```

With this function you can choose how many additional linear layers are added to the end, how much dropout to use after each one, and what kind of pooling to use. By default, fastai will apply both average pooling, and max pooling, and will concatenate the two together (this is the `AdaptiveConcatPool2d` layer). This is not a particularly common approach, but it was developed independently at fastai and other research labs in recent years, and tends to provide some small improvement over using just average pooling.

使用这个函数，我们能够选择在最后添加多少个额外的线性层，每个层后面使用多少个dropout，及使用什么类型的池化。默认的情况下，fastai会应用平均池化和最大池化，并会把两者联系起来（这就是`AdaptiveConcatPool2d`层）。这不是一个特别普通的方法，它是在fastai和其它实验室在近些年独立开发的，且往往比只使用平均池化提供一些小的改进。

fastai is a bit different from most libraries in that by default it adds two linear layers, rather than one, in the CNN head. The reason for this is that transfer learning can still be useful even, as we have seen, when transferring the pretrained model to very different domains. However, just using a single linear layer is unlikely to be enough in these cases; we have found that using two linear layers can allow transfer learning to be used more quickly and easily, in more situations.

fastai有绝大多数库有点不多，默认情况下它在CNN头中添加了两个线性层而不是一个。原因是，正如我们学过的，当迁移预训练模型到一个差异很大的领域时，这样做对迁移学习竟然总是有用的。不论以何种形式，在这些情况下只使用一个线性层是不可能足够的。我们已经发现，在更多情况下，使用两个线性层总是能够允许更快速和更容易的使用迁移学习。

> note: One Last Batchnorm?: One parameter to `create_head` that is worth looking at is `bn_final`. Setting this to `true` will cause a batchnorm layer to be added as your final layer. This can be useful in helping your model scale appropriately for your output activations. We haven't seen this approach published anywhere as yet, but we have found that it works well in practice wherever we have used it.

> 注释：一个最后批次标准化？：`create_head`的一个参数，值得关注的是`bn_final`。设置它为`true`会产生一个批次标准化层，添加到模型并作为你的最后层。这能够有帮于你的模型对于输出激活做到合适的缩放。我们还没有看到这个方法在任何地方发表，但我们已经发现在实践中无论我们在哪里使用它，它都会处理的很好，

Let's now take a look at what `unet_learner` did in the segmentation problem we showed in <chapter_intro>.

现在让我们看看`unet_learner`在<第一章：概述>中谈的分割问题中做了什么。

### unet_learner

### unet_learner

One of the most interesting architectures in deep learning is the one that we used for segmentation in <chapter_intro>. Segmentation is a challenging task, because the output required is really an image, or a pixel grid, containing the predicted label for every pixel. There are other tasks that share a similar basic design, such as increasing the resolution of an image (*super-resolution*), adding color to a black-and-white image (*colorization*), or converting a photo into a synthetic painting (*style transfer*)—these tasks are covered by an [online](https://book.fast.ai/) chapter of this book, so be sure to check it out after you've read this chapter. In each case, we are starting with an image and converting it to some other image of the same dimensions or aspect ratio, but with the pixels altered in some way. We refer to these as *generative vision models*.

深度学习中最有意思的架构之一是我们在<第一章：概述>中对于分割所使用那一种。分割是一个挑战任务，因为输出需要的是一个实际图像，或一个像素格，包含对每个像素的预测标记。还有其它任务共享了相似的基础设计，如增强图像的分辨率（*超分辨率*），对黑白图像添加颜色（*着色*），或一张照片转换为一个合成绘图（*风格迁移*），这些任务在本书的[在线](https://book.fast.ai/)章节都有所介绍，所以你阅读完本章节后要确保了解清楚。每个案例中，我们从一张图像开始，并把它转化化为一些相同尺寸或纵横比的其它图像，但以某种方法改变了像素。我们把这些称为*生成视觉模型*。

The way we do this is to start with the exact same approach to developing a CNN head as we saw in the previous problem. We start with a ResNet, for instance, and cut off the adaptive pooling layer and everything after that. Then we replace those layers with our custom head, which does the generative task.



There was a lot of handwaving in that last sentence! How on earth do we create a CNN head that generates an image? If we start with, say, a 224-pixel input image, then at the end of the ResNet body we will have a 7×7 grid of convolutional activations. How can we convert that into a 224-pixel segmentation mask?

Naturally, we do this with a neural network! So we need some kind of layer that can increase the grid size in a CNN. One very simple approach to this is to replace every pixel in the 7×7 grid with four pixels in a 2×2 square. Each of those four pixels will have the same value—this is known as *nearest neighbor interpolation*. PyTorch provides a layer that does this for us, so one option is to create a head that contains stride-1 convolutional layers (along with batchnorm and ReLU layers as usual) interspersed with 2×2 nearest neighbor interpolation layers. In fact, you can try this now! See if you can create a custom head designed like this, and try it on the CamVid segmentation task. You should find that you get some reasonable results, although they won't be as good as our <> results.

Another approach is to replace the nearest neighbor and convolution combination with a *transposed convolution*, otherwise known as a *stride half convolution*. This is identical to a regular convolution, but first zero padding is inserted between all the pixels in the input. This is easiest to see with a picture—<> shows a diagram from the excellent [convolutional arithmetic paper](https://arxiv.org/abs/1603.07285) we discussed in <>, showing a 3×3 transposed convolution applied to a 3×3 image.

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00051.png" alt="A transposed convolution" width="815" caption="A transposed convolution (courtesy of Vincent Dumoulin and Francesco Visin)" id="transp_conv">
  </p>
  <p align="center">图：转置卷积</p>
</div>

As you see, the result of this is to increase the size of the input. You can try this out now by using fastai's `ConvLayer` class; pass the parameter `transpose=True` to create a transposed convolution, instead of a regular one, in your custom head.

Neither of these approaches, however, works really well. The problem is that our 7×7 grid simply doesn't have enough information to create a 224×224-pixel output. It's asking an awful lot of the activations of each of those grid cells to have enough information to fully regenerate every pixel in the output. The solution to this problem is to use *skip connections*, like in a ResNet, but skipping from the activations in the body of the ResNet all the way over to the activations of the transposed convolution on the opposite side of the architecture. This approach, illustrated in <>, was developed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in the 2015 paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597). Although the paper focused on medical applications, the U-Net has revolutionized all kinds of generative vision models.

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00052.png" alt="The U-Net architecture" width="630" caption="The U-Net architecture (courtesy of Olaf Ronneberger, Philipp Fischer, and Thomas Brox)" id="unet">
  </p>
  <p align="center">图：U-Net架构</p>
</div>

This picture shows the CNN body on the left (in this case, it's a regular CNN, not a ResNet, and they're using 2×2 max pooling instead of stride-2 convolutions, since this paper was written before ResNets came along) and the transposed convolutional ("up-conv") layers on the right. Then extra skip connections are shown as gray arrows crossing from left to right (these are sometimes called *cross connections*). You can see why it's called a "U-Net!"

With this architecture, the input to the transposed convolutions is not just the lower-resolution grid in the preceding layer, but also the higher-resolution grid in the ResNet head. This allows the U-Net to use all of the information of the original image, as it is needed. One challenge with U-Nets is that the exact architecture depends on the image size. fastai has a unique `DynamicUnet` class that autogenerates an architecture of the right size based on the data provided.

Let's focus now on an example where we leverage the fastai library to write a custom model.
