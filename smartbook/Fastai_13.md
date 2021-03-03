# Convolutional Neural Networks

# 卷积神经网络

In <chapter_mnist_basics> we learned how to create a neural network recognizing images. We were able to achieve a bit over 98% accuracy at distinguishing 3s from 7s—but we also saw that fastai's built-in classes were able to get close to 100%. Let's start trying to close the gap.

在<章节：mnist基础>中我们学习了如何创建一个识别图像的神经网络。我们能够完成超过98%准确度来区别数字3和数字7。但我们也看到了fastai的内置类能够获得进100%的准确度。让我们开始尝试缩小这个差距。

In this chapter, we will begin by digging into what convolutions are and building a CNN from scratch. We will then study a range of techniques to improve training stability and learn all the tweaks the library usually applies for us to get great results.

在本章，我们会通过深入卷积是什么开始并从零开始创建一个卷积神经网络。然后我们会研究一些列技术来改善训练的稳定性并学习所有常用应用库的微调为我们获得更好的结果。

## The Magic of Convolutions

## 卷积的魔力

One of the most powerful tools that machine learning practitioners have at their disposal is *feature engineering*. A *feature* is a transformation of the data which is designed to make it easier to model. For instance, the `add_datepart` function that we used for our tabular dataset preprocessing in <chapter_tabular> added date features to the Bulldozers dataset. What kinds of features might we be able to create from images?

机器学习从业人员有他们最强大的工作之一来处理是*特征工程*。特征是数据的变换，它被设计来使得更容易建模。例如，在<章节：表格模型>中我们用于为我们的表格数据集做预处理的`add_datepart`函数，添加了数据特征给推土机数据集。图像的什么类型的特征我们能够创建呢？

> jargon: Feature engineering: Creating new transformations of the input data in order to make it easier to model.

> 术语：特征工程：创建输入数据新的变换，为了使得它更容易建模。

In the context of an image, a feature is a visually distinctive attribute. For example, the number 7 is characterized by a horizontal edge near the top of the digit, and a top-right to bottom-left diagonal edge underneath that. On the other hand, the number 3 is characterized by a diagonal edge in one direction at the top left and bottom right of the digit, the opposite diagonal at the bottom left and top right, horizontal edges at the middle, top, and bottom, and so forth. So what if we could extract information about where the edges occur in each image, and then use that information as our features, instead of raw pixels?

在图像环境中，特征是可见的特殊特性。例如，数字7的特征是数字顶部附近的水平边缘，和其下方从右上到左下的对角线边缘。另一方面，数字3的特性是一个直接在数字左上到右下的对角线边缘，相对立的对角线在左下和右上，水平边缘在中、上、下，诸如此类。所以如果你能够抽取在每张图像中边缘存在的位置相关信息，然后使用那些信息作为我们的特征，来替代原生像素呢？

It turns out that finding the edges in an image is a very common task in computer vision, and is surprisingly straightforward. To do it, we use something called a *convolution*. A convolution requires nothing more than multiplication, and addition—two operations that are responsible for the vast majority of work that we will see in every single deep learning model in this book!

在计算机视觉中查找图像中的边缘结果是一个非常普通的任务，且惊人的简单。做这个任务，我们使用称为*卷积* 的技术。卷积卷积只需要乘法和加法，在本书中我们将看到的每个单深度学习模型，这两个运算负责绝大多数的工作！

A convolution applies a *kernel* across an image. A kernel is a little matrix, such as the 3×3 matrix in the top right of <basic_conv>.

卷积应用一个*卷积核* 横穿一个图像。一个卷积核是一个小型矩阵，例如在<基础卷积>图的右上中是 3×3 矩阵。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/chapter9_conv_basic.png" id="basic_conv" caption="Applying a kernel to one location" alt="Applying a kernel to one location" width="700">
  </p>
  <p align="center">图：基础卷积</p>
</div>

The 7×7 grid to the left is the *image* we're going to apply the kernel to. The convolution operation multiplies each element of the kernel by each element of a 3×3 block of the image. The results of these multiplications are then added together. The diagram in <basic_conv> shows an example of applying a kernel to a single location in the image, the 3×3 block around cell 18.

左侧的 7×7 是我们将要应用卷积核的*图像*。卷积运算卷积核的每个元素乘以图像的 3×3 块的每个元素。然后这些乘积结果进行加总。在<基础卷积>图中显示了一个应用卷积核在图像中单个位置上的事例，这个  3×3 的块大约有18个单元。

Let's do this with code. First, we create a little 3×3 matrix like so:

让我们编写这个操作的代码，首先，我们创建如下的 3×3 小型矩阵：

```
top_edge = tensor([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]]).float()
```

We're going to call this our kernel (because that's what fancy computer vision researchers call these). And we'll need an image, of course:

我们称这用为卷积核（因为这是那些花哨的计算机视觉研究人员所称的）。当然我们需要一张图像：

```
path = untar_data(URLs.MNIST_SAMPLE)
```

```
#hide
Path.BASE_PATH = path
```

```
im3 = Image.open(path/'train'/'3'/'12.png')
show_image(im3);
```

Out: ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAABECAYAAAA4E5OyAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAADyElEQVR4nO2aTSg1URjHf1eIS9gQETYWPhOiFLGwkiTJzs7OXpFsWMlKsqEoRT4WFmKhlI+wsWWluCtECIVh3oX3mNd5hzvGneum51d3MzPOee7//p3zPM8Zn2maCBZRPx1ApCGCaIggGiKIhgiiER3k/m/egnx2F8UhGiKIhgiiIYJoiCAaIoiGCKIhgmiIIBrBMlVbHh8fAVhfXwcgPj4egO3tbQCur68BGBkZAaClpQWArKysD8fMzMwEoLm5GYDs7Gw3oX0bcYiGL0jHzPbm0NAQAN3d3SEPKCrq9TeqqKgAoLOzE4DW1lYAUlJSQjWV1DJOcOWQgoICAA4PD23/KC0tDYCamppPJ8/Pzwfg4OCAs7MzADY3N22f3d/fB6C0tPTTMb+AOMQJrnaZra0tAE5OToD/d4TY2FgAEhMTHY/58PAAQGFhIQBHR0fv7s/PzwMhdYgt4hANV2uIF2xsbABQV1f37npcXBzwus4A5OTkhGpKWUMcYZrmZx9PMQzDNAzD7O3tNf1+v+n3+02fz/fuEwgEzEAg4MX0tt9ZHKLhapf5Lip/mZiYAGB4ePjtXkxMDACLi4sApKenhzU2cYhGWB1yfHwMQHFxMQDPz8//PaNqGVUZ+3y2m4FniEM0wuqQ2dlZwN4ZCpWxlpWVAVBfXw9Ae3s7AE1NTQBkZGR4EmNYEzOVjvf39wOwtrYGwOnpqeMx1L/U4OAgAF1dXQAkJCR8NRxJzJzwo6m7ajXe3NxweXkJwMzMDGA1oYLE99aeXFhYAL60CItDnBAxxZ2OKvYGBgYAa735iMnJSQA6OjqcTiEOccKPpO5OqK2tBWB1dRWwmsxLS0u2z6v2wHcRh2hErEMUKu+oqqoCPnZIUVFRaOYLySi/CE8dcnt7C8D09DQAJSUlAFRXVzse4+XlBbCOIXSio1+/QmVlpes4/0UcouGJQ5QzGhoaANjb2wPg/v7e8Rh3d3cAjI2NAVYmqlNeXg5AXl6eu2A1xCEanjhEHYIrZyguLi4A66hTtQsBnp6eABgfHwegp6cHsOodhcqsk5OTAZiamgpp7OIQDU9qmZWVFQAaGxtt76tD8NTU1Ldr5+fnwMeH3YqkpCQAdnZ2AOvA3AVSyzjBE4dcXV0B0NfXB8Do6KibYQArz1Adsra2NgByc3Ndj/kXcYgTPO2HGIYBwO7uLgDLy8uAVXfMzc29PatewlGo9Uc54bMX9lwiDnFCxHbMwoA4xAkiiIYIoiGCaIggGsGq3fC+ixABiEM0RBANEURDBNEQQTREEI0/H3jyQ4wdtXsAAAAASUVORK5CYII=)

Now we're going to take the top 3×3-pixel square of our image, and multiply each of those values by each item in our kernel. Then we'll add them up, like so:

现在我们会取图像顶部 3×3 像素的正方形，并那些每个值乘以卷积核中的每个数据项。然后我们会加总他们，如下：

```
im3_t = tensor(im3)
im3_t[0:3,0:3] * top_edge
```

Out:$\begin{array}{c,l,l}
		tensor([&[-0., -0., -0.],\\
        &[0., 0., 0.],\\
        &[0., 0., 0.]]) \end{array}$

```
(im3_t[0:3,0:3] * top_edge).sum()
```

Out: tensor(0.)

Not very interesting so far—all the pixels in the top-left corner are white. But let's pick a couple of more interesting spots:

截止目前不是很有趣，左上角的所有像素是白色的。所以让我们选取一些更有意思的点：

```
#hide_output
df = pd.DataFrame(im3_t[:10,:20])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

Out: <img src="./_v_images/background_gradient.jpg" alt="background_gradient" style="zoom:50%;" />

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00059.png"alt="Top section of a digit" width="490" >
  </p>
  <p align="center">图：数字顶部部分</p>
</div>

There's a top edge at cell 5,8. Let's repeat our calculation there:

在单元坐标（5，8）上是顶部边缘。让我们重复我们的计算：

```
(im3_t[4:7,6:9] * top_edge).sum()
```

Out: tensor(762.)

There's a right edge at cell 8,18. What does that give us?:

单元坐标（8，18）是右边缘。这给了我们什么呢？：

```
(im3_t[7:10,17:20] * top_edge).sum()
```

Out: tensor(-29.)

As you can see, this little calculation is returning a high number where the 3×3-pixel square represents a top edge (i.e., where there are low values at the top of the square, and high values immediately underneath). That's because the `-1` values in our kernel have little impact in that case, but the `1` values have a lot.

如你所见，在顶部边缘 3×3 像素方格位置这个小计算返回了一个很大的值（即，在方格顶部有很小的值，在下面立刻有了很大的值）。这是因为我们卷积核中`-1`值在这种情况下有很小的影响，但是值`1`有比较大的影响。

Let's look a tiny bit at the math. The filter will take any window of size 3×3 in our images, and if we name the pixel values like this:

让我们看一点数学内容。在我们图像中过滤器会取任意尺寸 3×3 的窗口，如果我们命名像素值像下面的样子：
$$
\begin{matrix} a1 & a2 & a3 \\ a4 & a5 & a6 \\ a7 & a8 & a9 \end{matrix}
$$
it will return $a1+a2+a3-a7-a8-a9$. If we are in a part of the image where $a1$, $a2$, and $a3$ add up to the same as $a7$, $a8$, and $a9$, then the terms will cancel each other out and we will get 0. However, if $a1$ is greater than $a7$, $a2$ is greater than $a7$, $a2$ is greater than $a8$, and $a3$ is greater than $a9$,  we will get a bigger number as a result. So this filter detects horizontal edges—more precisely, edges where we go from bright parts of the image at the top to darker parts at the bottom.

它会返回 $a1+a2+a3-a7-a8-a9$。如果图像 $a1$, $a2$, 和 $a3$的部分合计与 $a7$, $a8$, and $a9$部分相等，那么数据项会相互抵消且我们会得到 0。然而，如果$a1$ 比 $a7$ 更大， $a2$ 比 $a7$ 更大， $a2$ 比 $a8$ 更大， 及 $a3$ 比 $a9$ 更大，我们会得到一个更大的结果值。所以这个过滤器检测了水平边缘，更加精确的话，从图像顶部的高亮部分到底部的暗黑部分的边缘。

Changing our filter to have the row of `1`s at the top and the `-1`s at the bottom would detect horizontal edges that go from dark to light. Putting the `1`s and `-1`s in columns versus rows would give us filters that detect vertical edges. Each set of weights will produce a different kind of outcome.

改变我们的过滤器为顶部是为`1`的行，底部是为`-1`的行，会检测从暗到亮的水平边缘。相对说行，在列中放置`1`和`-1`给我们的过滤器来检测垂直边缘。每个权重设置会产出不同种类的结果。

Let's create a function to do this for one location, and check it matches our result from before:

让我们创建一个函数来为一个位置做这个操作，并检查它是否与我们之前的结果相匹配：

```
def apply_kernel(row, col, kernel):
    return (im3_t[row-1:row+2,col-1:col+2] * kernel).sum()
```

```
apply_kernel(5,7,top_edge)
```

Out: tensor(762.)

But note that we can't apply it to the corner (e.g., location 0,0), since there isn't a complete 3×3 square there.

但要注意，我们不能在角上应用它（如，位置0,0），因为这里没有一个完整的 3×3 方格。

### Mapping a Convolution Kernel

### 映射卷积核

We can map `apply_kernel()` across the coordinate grid. That is, we'll be taking our 3×3 kernel, and applying it to each 3×3 section of our image. For instance, <nopad_conv> shows the positions a 3×3 kernel can be applied to in the first row of a 5×5 image.

我们能够在坐标格上映射`apply_kernel()`。即，我们会取 3×3 卷积核，并应用它在每个我们图像上 3×3 的区域上。例如，图<未填充的卷积>中展示了放置了 3×3 卷积核能够应用在 5×5 尺寸图像的第一行上。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/chapter9_nopadconv.svg" id="nopad_conv" caption="Applying a kernel across a grid" alt="Applying a kernel across a grid" width="400" >
  </p>
  <p align="center">图：未填充的卷积</p>
</div>

To get a grid of coordinates we can use a *nested list comprehension*, like so:

取表格的坐标，我们能够使用*嵌套列表解释器*，如下：

```
[[(i,j) for j in range(1,5)] for i in range(1,5)]
```

Out: $\begin{array}{r}[[(1, 1), (1, 2), (1, 3), (1, 4)],\\
 [(2, 1), (2, 2), (2, 3), (2, 4)],\\
 [(3, 1), (3, 2), (3, 3), (3, 4)],\\
 [(4, 1), (4, 2), (4, 3), (4, 4)]] \end{array}$

> note: Nested List Comprehensions: Nested list comprehensions are used a lot in Python, so if you haven't seen them before, take a few minutes to make sure you understand what's happening here, and experiment with writing your own nested list comprehensions.
>
> 注释：嵌套列表解释：在Python中嵌套列表解释经常被使用，如果之前你没有看过它们，花几分钟确保你能够理解这会发生什么，并编写你自己的嵌套列表解释来做实验。

Here's the result of applying our kernel over a coordinate grid:

这是在坐标表格上应用我们卷积核的结果：

```
rng = range(1,27)
top_edge3 = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])

show_image(top_edge3);
```

Out:![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAABECAYAAAA4E5OyAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAE1UlEQVR4nO2c104cSxRFF9iYYHIGk0EkiSQsXuA3+Ak+iI/hAT8ihEBgRI4i2ORsgkn3wdpT0weu1YN75KurWi893dNd01TvPnXOrhIpz8/PeBypf/sG/mv4DjH4DjH4DjH4DjG8/92Xw8PD/9shaGhoKOW1414hBt8hBt8hBt8hBt8hht+OMuLp6QmAk5MTAM7OzgDY39+PnfPjxw8AVlZWALi9vQ20UVJSAsCHDx8Cx9PS0gAoKyuLHfv06RMA5eXlAGRmZoa5zUjwCjGEUoiUMDs7C8DY2BgABwcHoX9ofX099LkZGRkAVFRUANDb2wtAU1MT4NRk1RYFXiEG3yGGUK/M1dUV4CQqKXd1dcXOKSgoAKClpQWAwsJCALKysgA4PT0FIDU1+AwUfBWMAaanpwFYXV0FYGpqKvD7+fn5gf0o8QoxhFJIf38/ALW1tQBUV1cDkJ2d7Rp6/6upd+/eBbZWERZZmEdHR7FjCwsLAIyMjACwu7sLuGB7c3MDQE5OTpjbTwivEEMohZSWlgJQWVkJuCRLKgCXvN3f3wNwd3cHwMPDQ6AtfS8UQ+IVos9VVVUA1NXVBa6RqmxbUeAVYgilEI0Ah4eHAKSnp784R6n7z58/AacQ7Qs93Z2dHcDFi/jzpIzBwcHAvtjY2ABgc3MzzO0nhFeIIZRC9LTji7lEUS4jZUxOTgbabG5ujp2r/EaxQ/mGYouUmgy8QgyhFPInXF9fAzA3NwfAxMRE4HhfXx8AHR0dsWuklr29PcDFHRWIujYZeIUYkq6Q8/NzwOUjNTU1gDOB6uvrAVf7AIyOjgJuNPn27Rvgyv6GhobAvjLYKPAKMSRdIaqMW1tbAZfdpqT8mie6uLgAgnmIMtDj42MAvn79GjhXmbKMo87OTsCpLv53EsUrxJB0hagylkKkGJnLUoOePrjqViNPd3c34DLmmZkZwI1Yqod6enpibTQ2Nr7pfr1CDL5DDEl/ZWQdyqGXpVhUVBTYjzebNATLbtDQrGFYr4YKw+/fvwPuVQPIzc0FXAAOi1eIIekKUVGnQLi4uAg4ZeTl5QFQXFwcu0blvlT08eNHwD112ZJqQwVjPNZ2CItXiCHpCrHIStBWMSbe7FGs0NCpaQchZUg5GtJfI9F1uF4hhsgVoiJOxo+KO01TqCBT7HgNpexabfBvKP2Pjz9CprfaCotXiCEyhUgZy8vLgLMINSK0t7cD4SaXHh8fAVf2yzpUuq91I7ISXmvzrSaSV4ghMoUoS9ze3gbcdKMKNE2DyszRKCPiDWzZjePj44BTwOfPnwFnEKlNrTBS3IC3G9FeIYbIFKL6Q3WGtlLMly9fAJifnwderhu7vLyMfZZ6tMRC0xIyn2UMKYZIGYo5tr1E8AoxRKaQtrY24GVWKWUo+5QRZCe9NHIADAwMAC7+yCLUGjPFDilJ1e7W1tYf/x1eIQbfIYbIXhnJVyW7jBkVXprJl7xlGOm618p/tWHXlGmlgdpSoH5rII3HK8QQmUIULPXkZeZo9ZG2SuETQWm4htW1tTXAGUMyoaLAK8QQeQxZWloCnFI0L6MkS2ayCjXFhXjLT7FA1oFihewAxZBk4BViiNwgssZMogbN38YrxJDi/xlCEK8Qg+8Qg+8Qg+8Qg+8Qg+8Qwz/aP/Y2oVu6fAAAAABJRU5ErkJggg==)

Looking good! Our top edges are black, and bottom edges are white (since they are the *opposite* of top edges). Now that our image contains negative numbers too, `matplotlib` has automatically changed our colors so that white is the smallest number in the image, black the highest, and zeros appear as gray.

We can try the same thing for left edges:

看起来很好！边缘顶部是黑色，边缘底部是白色（因为它们是边缘的*对立*面）。现在我们图像也包含负值，`matplotlib`已经制动的改变了颜色，所以这个在图像中白色是最小的值，黑色是最大的值，及另显示为灰色。

```
left_edge = tensor([[-1,1,0],
                    [-1,1,0],
                    [-1,1,0]]).float()

left_edge3 = tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng])

show_image(left_edge3);
```

Out:![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAABECAYAAAA4E5OyAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAEa0lEQVR4nO2cyUosSxCGv9brPM8TDqgLZ1EXIgouRHwZH8dXEcGFS9GFKCoqqOCEA87zrGdx+Dur8nhaqy37Xi75baqpLLOTyD8jIiOrjby/v+MwJP3bA/iv4Qxi4Qxi4Qxi4Qxi8U+sxrGxsf9tCBodHY18dN8pxMIZxMIZxMIZxMIZxCJmlLF5eXkB4PT0FICKiopo2/n5OQBZWVkAnJycANDS0gLA7e0tAI+Pjx/2nZaWFv2cn58PwP39PQBXV1dBhvktnEIsAilkY2MDMDPmVUiYRCK/U4Tc3FzAqE73pUYpKEycQiycQSwCLZmpqSkA2tra/vpMSkoKANnZ2TH7yszMBOD4+BiApCQzN5eXlwAUFRX5/kYOen19HYDt7e0vj/2rOIVYBFLIw8MDYGY/PT092paXlweY8KnZVfj9jLu7u+jnyclJAEZGRgAoLS319amyp1NIAgikkM7OTgBqampCH8jh4WH08+zsLAADAwMAHB0dAZCTkwPAxcVF6N8vnEIsAimkr68PMGv4I56engC4vr6O2dfNzQ0Ae3t7fzyvREz09vYCcHZ2BjiFJJRAClHuILyRQby9vQF/38RplpV2y3coTQcYGhoCoLu7GzDRRSn72tpakGEHwinEIpBCvoOyz7m5OQCam5t97eXl5dHPlZWVADQ2NgKQkZEBwPT0tK+vn8ApxCJhClHpQOtf+yGpwFtK0DZf7OzsACYzLisrA4wP+yyiBcEpxCJhClHuoKv8gh25wOQ5UoYUoAgmBdXV1QFGKbu7u9E+lA8FxSnEImEK0W54cHAQMDOo+ok3+1UUGR8fB0xEkqpsH6O+S0pKovdmZmbiGqdTiIUziEXClozCq9Lvr6Bt/+vrK2CWhq4Kw4WFhYBJ6AAaGhoA2NzcDDROpxCLhClEp3G6qhyp4nKskoLaFhYWAOjo6PC165yovb09es8pJCQSphAbb4HaRkVlu5QwPz8PGJUVFBQApqQg1QEUFxfHNS6nEIsfU4hmVQdRTU1NgCkUHxwcfNqHSgJSivxBV1eX7zklal7VxVsicAqxCF0hUoYigtZ7f38/ACsrK4H7kppUSlQeIqQ6XQEWFxcDjx2cQv4gdIUsLy8DsL+/DxiF1NbWAmZLr/xD/kFX7zPPz8++Z5VnVFdXA5CcnAyYvMT7vki8x5xOIRahK0SzK18xPDwMmNnWu2ba9stPqB1MQUhRQ1Gkp6cHMNt8RRv5Dn03mNwkKE4hFqErRL5AvkNHllKMVwmfkZqaChifUV9fD5i3IXXkKd8Rb2Tx4hRi4QxiEfqS0TskCpVK3ScmJgBobW0FTLKl4o73bFfVdF3lePWMHKYKR6urq77738EpxCJ0hVRVVQGmqOPdkoPZdGl2hcIw+FPwj5BTXVpa8vUZBk4hFqErRGcnKuF99S3EWCiUb21tAT/7YwCnEIsfKxApyqgcKHQO6z1lA5PAedHPUBKJU4hFxP0zBD9OIRbOIBbOIBbOIBbOIBbOIBa/AEQyr63rTKk/AAAAAElFTkSuQmCC)

As we mentioned before, a convolution is the operation of applying such a kernel over a grid in this way. In the paper ["A Guide to Convolution Arithmetic for Deep Learning"](https://arxiv.org/abs/1603.07285) there are many great diagrams showing how image kernels can be applied. Here's an example from the paper showing (at the bottom) a light blue 4×4 image, with a dark blue 3×3 kernel being applied, creating a 2×2 green output activation map at the top.

如我们之前提过，卷积是以这种方式在表格上的卷积核应用运算。在论文中[深度学习卷积运算指引](https://arxiv.org/abs/1603.07285)有很多图解，展示了图卷积核如何被应用。下面是论文中的一个例子。展示了一个浅蓝色 4×4 图像（在底部）带有被应用了深蓝色 3×3 卷积核，在顶部创建了一个 2×2 的绿色输出激活映射。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00028.png" alt="Result of applying a 3×3 kernel to a 4×4 image" width="782" caption="Result of applying a 3×3 kernel to a 4×4 image (courtesy of Vincent Dumoulin and Francesco Visin)" id="three_ex_four_conv"  >
  </p>
  <p align="center">图：在 4×4 图像应用 3×3 卷积核</p>
</div>

Look at the shape of the result. If the original image has a height of `h` and a width of `w`, how many 3×3 windows can we find? As you can see from the example, there are `h-2` by `w-2` windows, so the image we get has a result as a height of `h-2` and a width of `w-2`.

看一下结果形状。如果原始图像有高为`h`及宽为`w`，我们能够找到多少个 3×3 窗口？如你例子中所看到的，有`h-2`乘以`w-2`个窗口，所以对于这个图像，我们有一个高为`h-2`和宽为`w-2`的结果。

We won't implement this convolution function from scratch, but use PyTorch's implementation instead (it is way faster than anything we could do in Python).

我们不会从零开始实现这个卷积函数，而是使用PyTorch的实现来替代（它比我们能够在Python中所使用的任何方式都要快）。

### Convolutions in PyTorch

### PyTorch中的卷积

Convolution is such an important and widely used operation that PyTorch has it built in. It's called `F.conv2d` (recall that `F` is a fastai import from `torch.nn.functional`, as recommended by PyTorch). The PyTorch docs tell us that it includes these parameters:

- input:: input tensor of shape `(minibatch, in_channels, iH, iW)`
- weight:: filters of shape `(out_channels, in_channels, kH, kW)`

卷积是如此的重要及被广泛使用的运算，以至PyTorch已经内置它了。它被称为`F.conv2d`（回忆想一下，`F`是`torch.nn.functional`的fastai导入，这是PyTorch所推荐的）。PyTorch文档告诉我们它包含这些参数：

- 输入：输入形状张量 `(minibatch, in_channels, iH, iW)`
- 权重：形状过滤器`(out_channels, in_channels, kH, kW)`

Here `iH,iW` is the height and width of the image (i.e., `28,28`), and `kH,kW` is the height and width of our kernel (`3,3`). But apparently PyTorch is expecting rank-4 tensors for both these arguments, whereas currently we only have rank-2 tensors (i.e., matrices, or arrays with two axes).

这里的`iH,iW`是图像的高和宽（例如：`28,28`），及 `kH,kW`是我们卷积核的高和宽（`3,3`）。但很显然PyTorch希望对这两个的那些参数是4阶张量，不管怎么说当前我们只有2阶张量（例如，矩阵或两个轴的数组）。

The reason for these extra axes is that PyTorch has a few tricks up its sleeve. The first trick is that PyTorch can apply a convolution to multiple images at the same time. That means we can call it on every item in a batch at once!

由于这些扩展轴的因素，PyTorch有一些处理它的技巧。第一个小技巧是PyTorch能够同时在多张图像上应用一个卷积。这表示我们能够一次性的在一个批次中每个数据项上调用它！

The second trick is that PyTorch can apply multiple kernels at the same time. So let's create the diagonal-edge kernels too, and then stack all four of our edge kernels into a single tensor:

第二个小技巧是PyTorch能够同时应用多个内样。所以我们也能够创建对角边缘内样，然后堆积我们所有边缘的四个卷积核在一个张量中：

```
diag1_edge = tensor([[ 0,-1, 1],
                     [-1, 1, 0],
                     [ 1, 0, 0]]).float()
diag2_edge = tensor([[ 1,-1, 0],
                     [ 0, 1,-1],
                     [ 0, 0, 1]]).float()

edge_kernels = torch.stack([left_edge, top_edge, diag1_edge, diag2_edge])
edge_kernels.shape
```

Out: torch.Size([4, 3, 3])

To test this, we'll need a `DataLoader` and a sample mini-batch. Let's use the data block API:

来测试一下，我们会需要一个`DataLoader`和一个简单的小批次。让我们使用数据块API：

```
mnist = DataBlock((ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(path)
xb,yb = first(dls.valid)
xb.shape
```

Out: torch.Size([64, 1, 28, 28])

By default, fastai puts data on the GPU when using data blocks. Let's move it to the CPU for our examples:

当使用数据块时，fastai默认放置数据在GPU上。让我们把我们的例子移到CPU上：

```
xb,yb = to_cpu(xb),to_cpu(yb)
```

One batch contains 64 images, each of 1 channel, with 28×28 pixels. `F.conv2d` can handle multichannel (i.e., color) images too. A *channel* is a single basic color in an image—for regular full-color images there are three channels, red, green, and blue. PyTorch represents an image as a rank-3 tensor, with dimensions `[channels, rows, columns]`.

一个批次包含64张图像，每个都是1个通道及 28×28 个像素。`F.conv2d`也能够处理多通道（即，彩色）图像。一个*通道*是一张图像中的一个单基础色，对于通常的全色图像有三个通道：红、绿和蓝。PyTorch用一个3阶张量维度为`[通道，行，列]`描述一张图像。

We'll see how to handle more than one channel later in this chapter. Kernels passed to `F.conv2d` need to be rank-4 tensors: `[channels_in, features_out, rows, columns]`. `edge_kernels` is currently missing one of these. We need to tell PyTorch that the number of input channels in the kernel is one, which we can do by inserting an axis of size one (this is known as a *unit axis*) in the first location, where the PyTorch docs show `in_channels` is expected. To insert a unit axis into a tensor, we use the `unsqueeze` method:

稍后在本章节我们会看到如何处理超过一个通道。卷积核传递给`F.conv2d`需要是4阶张量：`[channels_in, features_out, rows, columns]`. `edge_kernels`，目前缺少其中一个。我们需要告诉PyTorch在卷积核中输入通道的数量是一个，在第一个位置我们能够通过插入一个尺寸为一的轴来实现（这被称为*单元轴*），这个位置PyTorch文档显示要求是`in_channels`。插入一个单元轴进入一个张量，我们使用`unsqueeze`方法：

```
edge_kernels.shape,edge_kernels.unsqueeze(1).shape
```

Out: (torch.Size([4, 3, 3]), torch.Size([4, 1, 3, 3]))

This is now the correct shape for `edge_kernels`. Let's pass this all to `conv2d`:

现在这就是`edge_kernels`的正确形状。让我们传递所有这些内容给`conv2d`：

```
edge_kernels = edge_kernels.unsqueeze(1)
```

```
batch_features = F.conv2d(xb, edge_kernels)
batch_features.shape
```

Out: torch.Size([64, 4, 26, 26])

The output shape shows we gave 64 images in the mini-batch, 4 kernels, and 26×26 edge maps (we started with 28×28 images, but lost one pixel from each side as discussed earlier). We can see we get the same results as when we did this manually:

输出形状显示，我们在这个小批次中有64张图像，4个卷积核，及 26×26 个边缘映射（我们从 28×28 的图像开始的，但是如之前讨论过的，每个边丢失了一个像素）。我们能够看到我们获得了与手动做这个操作相同的结果：

```
show_image(batch_features[0,0]);
```

Out:![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAABECAYAAAA4E5OyAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAADdUlEQVR4nO2cyUorQRRAT5wVFJxHVARxgDgguHbjzh9wq1s/x1/RrStXggqCExFRcZFExQEnHBeP25W+rw0x6XQ/Hvesmkp1p7x9UnXrdmPi6+sLw1ER9wD+NSwgCguIwgKisIAoqvJ9uLq6+t8uQSsrK4mgdjNEYQFRWEAUFhCFBURhAVFYQBQWEIUFRGEBUeRN3eNCilaXl5cAvL+/A9Df3w9AQ0MDACMjIwAcHx8D8Pz8XPJ3myGK2A15e3sD4Obmxmt7eXkB4PPzE4BEInAfxuDgoO/ci4uLksdjhihCM+Tu7g5wv+/q6urAfufn5wD09PQAzgKxohDk2l1dXQA0NTUVMeJgzBBFSYaIFQD7+/sATE9PAz8bUggyZ/T19QVeq66uztcvjNVFMEMUJRlye3vrHW9tbQEwOTmZ9xy5u4Lc/YmJCa9NVp7t7W0ARkdHfee0t7cDLl+5urr69dh/wgxRhLbKPDw8AC67HB4eBqC1tRVwq4nkDpIzfHx8AFBfX+9dS1acjY0NwBkic8bQ0BDgDL2/vw/rzzBDNCUZkrv+z87OAlBZWVnUtWTeANjb2wP8qxjA2NgYADU1NQCcnp4W9V35MEMUJRnS3NzsHY+PjwP+Ox2EzA/6NQzJYAHW19cBmJmZ8fWRuUSucXBwUMyw82KGKEJbZXp7ewPbr6+vCzo/lUp5x5lMBoDl5WUAOjs7AWhpaQFgc3MTcCtbmJghCguIIvYC0dPTEwA7Ozte29TUFOBS9GQyCbhCkCR/5cAMUcRuiEymMpECLC4uAi5FF1N2d3eB3xWTfosZoojNENnsyRZfSooA3d3dvraqqj/DPDo68p1bDswQRWyGnJycAC5xW1hY8D6TzdvAwAAAh4eHQHnNEMwQReSGPD4+Am7FkEJSbuovZUhZTcJ4AFUoZogickMk35Ai0NLSEgAdHR1eH9nmS9/c0kC5MUMUkRuSzWYB94qD5BptbW1eHylMr62tRTw6M+QvIjdEHipJgVpyDnk8AW4HHObjhUIxQxQWEEXkPxl5S0Cq9ILUTaHwOmw5MEMUkRtydnYGwPz8vK+9sbHRO5aNXxyYIYrIDZH3Q15fXwGora0FoKLC3RtZduPADFFEbsjc3BzgEjIhnU57x7nvrEaNGaJI2D9D8GOGKCwgCguIwgKisIAoLCCKb79WEcYbcUyrAAAAAElFTkSuQmCC)

The most important trick that PyTorch has up its sleeve is that it can use the GPU to do all this work in parallel—that is, applying multiple kernels, to multiple images, across multiple channels. Doing lots of work in parallel is critical to getting GPUs to work efficiently; if we did each of these operations one at a time, we'd often run hundreds of times slower (and if we used our manual convolution loop from the previous section, we'd be millions of times slower!). Therefore, to become a strong deep learning practitioner, one skill to practice is giving your GPU plenty of work to do at a time.

PyTorch有一个最重要的技巧，就是它能够使用GPU并行做这些所有处理。即，应用多个卷积核，给多个图像，遍历多个通道。并行做这么多的处理对于让GPU有效的工作是至关重要的。如果我们每次做每次运算中的一个，通常我们的运行速度会慢上数百倍（如果我们使用上部分的手动卷积循环，我们会慢上数百万倍）。因此一个优秀的深度学习从业人员，一个常规的技巧是每次给你的GPU大量的工作去做。

It would be nice to not lose those two pixels on each axis. The way we do that is to add *padding*, which is simply additional pixels added around the outside of our image. Most commonly, pixels of zeros are added.

在每个轴上不丢失那两个像素会更好些。因此我们通过添加*填充*来完成这个工作，它是简单的在我们图像的外围添加像素。最常用的是添加像素零。

### Strides and Padding

### 步长和填充

With appropriate padding, we can ensure that the output activation map is the same size as the original image, which can make things a lot simpler when we construct our architectures. <pad_conv> shows how adding padding allows us to apply the kernels in the image corners.

有合理的填充，我们能够确保输出激活映射与原始图像是相同的尺寸，当我们构建我们的架构时它能够使的事情简单的多。图<带有填充的卷积>显示了如何添加填充以允许我们在图像的角上应用卷积核。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/chapter9_padconv.svg" id="pad_conv" caption="A convolution with padding" alt="A convolution with padding" width="600" >
  </p>
  <p align="center">图：带有填充的卷积</p>
</div>

With a 5×5 input, 4×4 kernel, and 2 pixels of padding, we end up with a 6×6 activation map, as we can see in <four_by_five_conv>.

有一个 5×5 的卷积核，4×4 的卷积核及填充了两个像素，最终有一个 6×6 的激活映射，如下图<5×5 输入的 4×4 卷积核及填充了 2 个像素>所示。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00029.png" alt="A 4×4 kernel with 5×5 input and 2 pixels of padding" width="783" caption="A 4×4 kernel with 5×5 input and 2 pixels of padding (courtesy of Vincent Dumoulin and Francesco Visin)" id="four_by_five_conv" >
  </p>
  <p align="center">图：5×5 输入的 4×4 卷积核及填充了 2 个像素</p>
</div>

If we add a kernel of size `ks` by `ks` (with `ks` an odd number), the necessary padding on each side to keep the same shape is `ks//2`. An even number for `ks` would require a different amount of padding on the top/bottom and left/right, but in practice we almost never use an even filter size.

如果我们添加了尺寸`ks`乘以`ks`的卷积核（`ks`是一个奇数），必须每个边填充`ks//2`来保持相同的形状。如果`ks`为偶数也许需要在顶部/底部和左侧/右侧填充不同的数量，但实践中我们几乎从来都不用偶数过滤器尺寸。

So far, when we have applied the kernel to the grid, we have moved it one pixel over at a time. But we can jump further; for instance, we could move over two pixels after each kernel application, as in <three_by_five_conv>. This is known as a *stride-2* convolution. The most common kernel size in practice is 3×3, and the most common padding is 1. As you'll see, stride-2 convolutions are useful for decreasing the size of our outputs, and stride-1 convolutions are useful for adding layers without changing the output size.

截止目前，当我们已经应用卷积核到表格时，我们每次结束都移动卷积核一个像素。但我们能够跳的更远。例如，我们能够在每个卷积核应用后移两个像素，如图<有 5×5 输入的 3×3 卷积核和步长2卷积及填充1个像素>所示。这被称为*步长2*卷积。在实践中最常用的卷积核尺寸是 3×3，及最常用的填充是1.你将会看到，步长2卷积对于减小我们输出的尺寸是有帮助的，步长1卷积对于不用改变输出尺寸的添加层是有用处的。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00030.png" alt="A 3×3 kernel with 5×5 input, stride-2 convolution, and 1 pixel of padding" width="774" caption="A 3×3 kernel with 5×5 input, stride-2 convolution, and 1 pixel of padding (courtesy of Vincent Dumoulin and Francesco Visin)" id="three_by_five_conv">
  </p>
  <p align="center">图：有 5×5 输入的 3×3 卷积核和步长2卷积及填充1个像素</p>
</div>

In an image of size `h` by `w`, using a padding of 1 and a stride of 2 will give us a result of size `(h+1)//2` by `(w+1)//2`. The general formula for each dimension is `(n + 2*pad - ks)//stride + 1`, where `pad` is the padding, `ks`, the size of our kernel, and `stride` is the stride.

在一张尺寸`h`乘以`w`的图像中，使用为1的填充和步长2会提供给我们尺寸`(h+1)//2` 乘以 `(w+1)//2`的结果。对于每个维度的常用公式是 `(n + 2*pad - ks)//stride + 1`，这里的`pad`是填充数，`ks`为我们卷积核的尺寸，及`stride`是步长数。

Let's now take a look at how the pixel values of the result of our convolutions are computed.

现在让我们看一下我们卷积结果的像素值是如何计算的。

### Understanding the Convolution Equations

### 理解卷积方程

To explain the math behind convolutions, fast.ai student Matt Kleinsmith came up with the very clever idea of showing [CNNs from different viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c). In fact, it's so clever, and so helpful, we're going to show it here too!

来解释一下卷积背后的数字，fast.ai的学生马特·克莱因史密斯提出了展示[CNN不同观察视角](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)的非常聪明的想法。实践上，这人想法是那么的聪明，那么的有帮助，我们也会在这里展示它！

Here's our 3×3 pixel image, with each pixel labeled with a letter:

这是我们的 3×3 像素图像，每个像素用一个字母做了标记：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00032.png" alt="The image" width="75" >
  </p>
  <p align="center">图：图像</p>
</div>

And here's our kernel, with each weight labeled with a Greek letter:

这是我们的卷积核，每个权重用一个希腊字母做了标记：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00033.png" alt="The kernel" width="55" >
  </p>
  <p align="center">图：卷积核</p>
</div>

Since the filter fits in the image four times, we have four results:

因此在图像中过滤器拟合了四次，我们有四个结果：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00034.png" alt="The activations" width="52" >
  </p>
  <p align="center">图：激活单元</p>
</div>

<apply_kernel> shows how we applied the kernel to each section of the image to yield each result.

下图<应用卷积核>展示了我们如果应用卷积核到图像的每个部分从而产生每个结果：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00035.png" alt="Applying the kernel" width="366" caption="Applying the kernel" id="apply_kernel" >
  </p>
  <p align="center">图：应用卷积核</p>
</div>

The equation view is in <eq_view>.

在图<方程>中显示了方式计算。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00036.png" alt="The equation" width="436" caption="The equation" id="eq_view" >
  </p>
  <p align="center">图：方程</p>
</div>

Notice that the bias term, *b*, is the same for each section of the image. You can consider the bias as part of the filter, just like the weights (α, β, γ, δ) are part of the filter.

请注意偏置项*b*，在图像的每个部分都是相同的。你可以认为偏置作为过滤器的部分，就像权重（α, β, γ, δ)）是过滤器的一部分一样。

Here's an interesting insight—a convolution can be represented as a special kind of matrix multiplication, as illustrated in <conv_matmul>. The weight matrix is just like the ones from traditional neural networks. However, this weight matrix has two special properties:

这是一个有趣的见解，卷积能够被描述为一个特殊类型的矩阵乘法，如插图<卷积既矩阵乘法>所示。权重矩阵只是类似传统的神经网络。然而，权重矩阵有两个特别的属性：

1. The zeros shown in gray are untrainable. This means that they’ll stay zero throughout the optimization process.
2. Some of the weights are equal, and while they are trainable (i.e., changeable), they must remain equal. These are called *shared weights*.

1. 零所显示的灰色是不可训练的。这表示整个优化过程它们会保持为零。
2. 一些权重是相等的，且在它们可训练期间（即，可改变），它们必须保持相等。这被称为*共享权重*。

The zeros correspond to the pixels that the filter can't touch. Each row of the weight matrix corresponds to one application of the filter.

零相当于过滤器不能触碰的像素。权重矩阵的每行相当于一个过滤器的应用。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00038.png" alt="Convolution as matrix multiplication" width="683" caption="Convolution as matrix multiplication" id="conv_matmul">
  </p>
  <p align="center">图：卷积既矩阵乘法</p>
</div>

Now that we understand what a convolution is, let's use them to build a neural net.

现在我们理解什么是卷积，让我们使用它们来建立一个神经网络。

## Our First Convolutional Neural Network

## 我们的首个卷积神经网络

There is no reason to believe that some particular edge filters are the most useful kernels for image recognition. Furthermore, we've seen that in later layers convolutional kernels become complex transformations of features from lower levels, but we don't have a good idea of how to manually construct these.

没有理由相信一些特定边缘过滤器对于图像识别是最有用的卷积核。此外，我们已经看到了最后的层卷积核为了更低等级要求的复杂特征变换。

Instead, it would be best to learn the values of the kernels. We already know how to do this—SGD! In effect, the model will learn the features that are useful for classification.

相反，它也许最好来学习卷积核的值。我们已经知道如何为随机梯度下降做这个事情了！事实上，模型将要学习的特征对于特征分类是有用处的。

When we use convolutions instead of (or in addition to) regular linear layers we create a *convolutional neural network* (CNN).

当我们使用卷积而不是（或还有）常规的线性层，我们就创建了一个*卷积神经网络*（CNN）。

### Creating the CNN

### 创建卷积神经网络

Let's go back to the basic neural network we had in <chapter_mnist_basics>. It was defined like this:

我们先返回到<章节：mnist基础>中的基础神经网络。它的定义是这样的：

```
simple_net = nn.Sequential(
   nn.Linear(28*28,30),
   nn.ReLU(),
   nn.Linear(30,1)
)
```

We can view a model's definition:

我们能够查看一个模型的定义：

```
simple_net
```

```
Sequential(
    (0): Linear(in_features=784, out_features=30, bias=True)
    (1): ReLU()
    (2): Linear(in_features=30, out_features=1, bias=True)
)
```

We now want to create a similar architecture to this linear model, but using convolutional layers instead of linear. `nn.Conv2d` is the module equivalent of `F.conv2d`. It's more convenient than `F.conv2d` when creating an architecture, because it creates the weight matrix for us automatically when we instantiate it.

现在希望创建一个与这个线性模型类似的结构，只是使用卷积层替代线性层。`nn.Conv2d`是与`F.conv2d`等价的模块。当创建一个架构时，它比`F.conv2d`更方便，因为当我们实例化它时，它会为我们自动创建权重矩阵。

Here's a possible architecture:

这是一个合适的架构：

```
broken_cnn = sequential(
    nn.Conv2d(1,30, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(30,1, kernel_size=3, padding=1)
)
```

One thing to note here is that we didn't need to specify 28×28 as the input size. That's because a linear layer needs a weight in the weight matrix for every pixel, so it needs to know how many pixels there are, but a convolution is applied over each pixel automatically. The weights only depend on the number of input and output channels and the kernel size, as we saw in the previous section.

有一个事情要注意，那就是我们不需要具体说明 28×28 作为输入尺寸。这是因为线性层对于每个像素在权重矩阵是需要一个权重，因此它需要知道有多少个像素，但是一个卷积在每个像素上自动的应用。权重仅仅依赖于输入的数值和输出通道及卷积核尺寸，就像我们在之前部分看到的那样。

Think about what the output shape is going to be, then let's try it and see:

想一下输出形状会是什么，然后我们实验一下并查看：

```
broken_cnn(xb).shape
```

Out: torch.Size([64, 1, 28, 28])

This is not something we can use to do classification, since we need a single output activation per image, not a 28×28 map of activations. One way to deal with this is to use enough stride-2 convolutions such that the final layer is size 1. That is, after one stride-2 convolution the size will be 14×14, after two it will be 7×7, then 4×4, 2×2, and finally size 1.

这不是我们能够用于分类的内容，因为我们需要每张图像一个单输出激活，而不是一个 28×28 的映射映射。有一个方法来处理这个内容是使用足够的步长2卷积，这样最后的层是尺寸 1 了。即，一个步长2卷积后尺寸会是 14×14，两个后它会是 7×7，其后是 4×4,，2×2，最终尺寸是 1。

Let's try that now. First, we'll define a function with the basic parameters we'll use in each convolution:

让我们现在实验一下。首先，我们会定义一个函数，并带有我们在每个卷积上使用的基础参数：

```
def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res
```

> important: Refactoring: Refactoring parts of your neural networks like this makes it much less likely you'll get errors due to inconsistencies in your architectures, and makes it more obvious to the reader which parts of your layers are actually changing.

> 重要提示：重构：像这样重构我们神经网络部分，使得它不太可能由于在你的架构中前后矛盾而导致出错，并使得它对于阅读者更明了你的层实际正在改变的部分。

When we use a stride-2 convolution, we often increase the number of features at the same time. This is because we're decreasing the number of activations in the activation map by a factor of 4; we don't want to decrease the capacity of a layer by too much at a time.

当你使用步长2卷积，我们通常会同时增加特征数值。这是因为在激活映射中我们激活数量减少了 4 倍。我们不希望每次减小太多层的容量。

> jargon: channels and features: These two terms are largely used interchangeably, and refer to the size of the second axis of a weight matrix, which is, the number of activations per grid cell after a convolution. *Features* is never used to refer to the input data, but *channels* can refer to either the input data (generally channels are colors) or activations inside the network.

> 术语：通道和特征：这两个概念很大程度上交替使用，参照权重矩阵第二个轴线的尺寸，那是一个卷积后每个表格单元激活的数量。特征永远不会用于涉及输入数据，但是通道能够要么涉及输入数据（通常通道是彩色的）要么涉及网络内部的激活。

Here is how we can build a simple CNN:

下面是我们如何创建一个简单的卷积神经网络：

```
simple_cnn = sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,32),           #2x2
    conv(32,2, act=False), #1x1
    Flatten(),
)
```

> j: I like to add comments like the ones here after each convolution to show how large the activation map will be after each layer. These comments assume that the input size is 28*28

> 杰：我喜欢在每个卷积后添加像这样的注释，来说明激活映射每层后会是多大。这些注释假定输入尺寸为 28*28

Now the network outputs two activations, which map to the two possible levels in our labels:

现在网络输出两个激活，在我们的标签上有两个可能级别的激活映射：

```
simple_cnn(xb).shape
```

Out: torch.Size([64, 2])

We can now create our `Learner`:

现在我们可以常见我们的`Learner`了：

```
learn = Learner(dls, simple_cnn, loss_func=F.cross_entropy, metrics=accuracy)
```

To see exactly what's going on in the model, we can use `summary`:

来看一下在模型中到底是怎么回事，我们可以使用`summary`：

```
learn.summary()
```

```
Sequential (Input shape: ['64 x 1 x 28 x 28'])
================================================================
Layer (type)         Output Shape         Param #    Trainable 
================================================================
Conv2d               64 x 4 x 14 x 14     40         True      
________________________________________________________________
ReLU                 64 x 4 x 14 x 14     0          False     
________________________________________________________________
Conv2d               64 x 8 x 7 x 7       296        True      
________________________________________________________________
ReLU                 64 x 8 x 7 x 7       0          False     
________________________________________________________________
Conv2d               64 x 16 x 4 x 4      1,168      True      
________________________________________________________________
ReLU                 64 x 16 x 4 x 4      0          False     
________________________________________________________________
Conv2d               64 x 32 x 2 x 2      4,640      True      
________________________________________________________________
ReLU                 64 x 32 x 2 x 2      0          False     
________________________________________________________________
Conv2d               64 x 2 x 1 x 1       578        True      
________________________________________________________________
Flatten              64 x 2               0          False     
________________________________________________________________

Total params: 6,722
Total trainable params: 6,722
Total non-trainable params: 0

Optimizer used: <function Adam at 0x7fbc9c258cb0>
Loss function: <function cross_entropy at 0x7fbca9ba0170>

Callbacks:
  - TrainEvalCallback
  - Recorder
  - ProgressCallback
```

Note that the output of the final `Conv2d` layer is `64x2x1x1`. We need to remove those extra `1x1` axes; that's what `Flatten` does. It's basically the same as PyTorch's `squeeze` method, but as a module.

注意最后`Conv2d`层的输出是`64x2x1x1`。我们需要移除这些额外的`1x1` 轴线。这就是`Flatten`做的事情。这与PyTorch的`squeeze`方法基本相同，但是它是一个模型。

Let's see if this trains! Since this is a deeper network than we've built from scratch before, we'll use a lower learning rate and more epochs:

让我们看一下是否可以训练！因为这是一个比我们之前从零开始所创建的网络更深的网络，我们会使用一个更低的学习率和更多的周期：

```
learn.fit_one_cycle(2, 0.01)
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   0.072684 |   0.045110 | 0.990186 | 00:05 |
|     1 |   0.022580 |   0.030775 | 0.990186 | 00:05 |

Success! It's getting closer to the `resnet18` result we had, although it's not quite there yet, and it's taking more epochs, and we're needing to use a lower learning rate. We still have a few more tricks to learn, but we're getting closer and closer to being able to create a modern CNN from scratch.

成功了！它更接近我们取得的`resnet18`结果，然而它还没有完全达到这个结果，它需要更多的周期及我们需要使用更低的学习率。我们依然有很多技巧来学习，但是我们已经越来越接近从零开始创建一个现代卷积神经网络的能力了。

### Understanding Convolution Arithmetic

### 理解卷积算法

We can see from the summary that we have an input of size `64x1x28x28`. The axes are `batch,channel,height,width`. This is often represented as `NCHW` (where `N` refers to batch size). Tensorflow, on the other hand, uses `NHWC` axis order. The first layer is:

我们能够从模型总结命令输出结果中看到，我们有一个 `64x1x28x28`尺寸的输入。轴维度是`批次`，`通道`，`高`，`宽`。这经常表示为`NCHW`（这里的`N`指的是批次尺寸）。另一方面，Tensorflow使用`NHWC`轴顺序。第一层是：

```
m = learn.model[0]
m
```

```
Sequential(
  (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (1): ReLU()
)
```

So we have 1 input channel, 4 output channels, and a 3×3 kernel. Let's check the weights of the first convolution:

所以我们有1个输入通道，及4个输出通道，和一个 3×3 卷积核。让我们检查一下第一个卷积的权重：

```
m[0].weight.shape
```

Out: torch.Size([4, 1, 3, 3])

The summary shows we have 40 parameters, and `4*1*3*3` is 36. What are the other four parameters? Let's see what the bias contains:

模型总结命令输出结果展示我们有40个参数，及 `4*1*3*3`是36。其它四个参数是什么呢？让我们看一下偏置包含什么内容：

```
m[0].bias.shape
```

Out: torch.Size([4])

We can now use this information to clarify our statement in the previous section: "When we use a stride-2 convolution, we often increase the number of features because we're decreasing the number of activations in the activation map by a factor of 4; we don't want to decrease the capacity of a layer by too much at a time."

现在我们能够使用这个信息来澄清之前部分我们的陈述：“当我们使用步长2卷积时，我们通常增加特征数，因为我们在激活映射中缩小了激活数4倍。我们不希望每次减小一个层的容量太多。

There is one bias for each channel. (Sometimes channels are called *features* or *filters* when they are not input channels.) The output shape is `64x4x14x14`, and this will therefore become the input shape to the next layer. The next layer, according to `summary`, has 296 parameters. Let's ignore the batch axis to keep things simple. So for each of `14*14=196` locations we are multiplying `296-8=288` weights (ignoring the bias for simplicity), so that's `196*288=56_448` multiplications at this layer. The next layer will have `7*7*(1168-16)=56_448` multiplications.

每个通道有一个偏置。（有时候当通道不是输入通道时，他们被称为*特征*或*过滤器*。）输出形状是`64x4x14x14`，因此这会变成下个层的输入形状。根据`summary`输出下个层有 296 个参数。让我们忽略批次轴维度来让事情简单化。所以对于 `14*14=196` 的每个定位，我们是乘以`296-8=288`个权重（为了简化忽略了偏置）所以这就是 在这层是`196*288=56_448`的乘法。下一层会有 `7*7*(1168-16)=56_448`的乘法。

What happened here is that our stride-2 convolution halved the *grid size* from `14x14` to `7x7`, and we doubled the *number of filters* from 8 to 16, resulting in no overall change in the amount of computation. If we left the number of channels the same in each stride-2 layer, the amount of computation being done in the net would get less and less as it gets deeper. But we know that the deeper layers have to compute semantically rich features (such as eyes or fur), so we wouldn't expect that doing *less* computation would make sense.

在这里发生了什么，我们步长2卷积从 `14x14` 到 `7x7`减半了*表格尺寸*，且我们从 8 到 16加倍了*过滤器的数量*，这就导致计算数量总体上没有变化。如果我们在每个步长2层让通道数保持相同，在网络中随着网络加深计算数量会变的越来越少。但我们知道更深的层必须计算语意丰富的特征（如眼睛或皮毛），所以我们不希望做 *少* 的计算会有很意义。

Another way to think of this is based on receptive fields.

另外一个思考方式是基于感受野。

### Receptive Fields

### 感受野

The *receptive field* is the area of an image that is involved in the calculation of a layer. On the [book's website](https://book.fast.ai/), you'll find an Excel spreadsheet called *conv-example.xlsx* that shows the calculation of two stride-2 convolutional layers using an MNIST digit. Each layer has a single kernel. <preced1> shows what we see if we click on one of the cells in the *conv2* section, which shows the output of the second convolutional layer, and click *trace precedents*.

*感受野*是一个图像涉及到层计算中的面积。在本书的[网站](https://book.fast.ai/)，你会找到一个电子表格文件称为*conv-example.xlsx*，这个文件展示了使用一个MNIST数字两个步长2卷积层的计算。每个层有一个单核。在图<Conv2层的直接引用单元>中展示了如果你点击*conv2*区域中的一个单元（该单元展示了第二个卷积层的输出），并点击*追踪引用单元*后的结果。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00068.png" alt="Immediate precedents of conv2 layer" width="308" caption="Immediate precedents of Conv2 layer" id="preced1" >
  </p>
  <p align="center">图：Conv2层的直接引用单元</p>
</div>

Here, the cell with the green border is the cell we clicked on, and the blue highlighted cells are its *precedents*—that is, the cells used to calculate its value. These cells are the corresponding 3×3 area of cells from the input layer (on the left), and the cells from the filter (on the right). Let's now click *trace precedents* again, to see what cells are used to calculate these inputs. <preced2> shows what happens.

这里，带有绿色边框的单元是我们点击的单元，蓝色高亮单元是它的*引用单元*，即，这些单元用于计算它的值。这些单元对应的是输入层（左侧）单元 3×3 的面积和过滤器（右侧）的那些单元。现在我们再次点击*追踪引用*，来看一下哪些单元被用于计算这些输入。图<Conv2层的第二引用单元>中展示了所发生的事情。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00069.png" alt="Secondary precedents of conv2 layer" width="601" caption="Secondary precedents of Conv2 layer" id="preced2" >
  </p>
  <p align="center">图：Conv2层的第二引用单元</p>
</div>

In this example, we have just two convolutional layers, each of stride 2, so this is now tracing right back to the input image. We can see that a 7×7 area of cells in the input layer is used to calculate the single green cell in the Conv2 layer. This 7×7 area is the *receptive field* in the input of the green activation in Conv2. We can also see that a second filter kernel is needed now, since we have two layers.

在这个例子中，我们只有两个卷积层，每个都是步长2，所以现在可以立刻回溯到输入图像。我们能够看到在输入层的 7×7   单元面积被用于计算Conv2层中的那个单一绿色单元。这个 7×7 面积是Conv2中绿色激活输入中的*感受野*。你也能够看到因为我们有两个层，现在第二个过滤器卷积核也是需要的。

As you see from this example, the deeper we are in the network (specifically, the more stride-2 convs we have before a layer), the larger the receptive field for an activation in that layer. A large receptive field means that a large amount of the input image is used to calculate each activation in that layer is. We now know that in the deeper layers of the network we have semantically rich features, corresponding to larger receptive fields. Therefore, we'd expect that we'd need more weights for each of our features to handle this increasing complexity. This is another way of saying the same thing we mentioned in the previous section: when we introduce a stride-2 conv in our network, we should also increase the number of channels.

如你在本例子中所看到的，我们在网络中越深（具体来说，在一个层后，我们有更多的步长2卷积），在那个层中对于一个激活的感受野就更大。一个巨大的感受野意味着巨大的图像数量被用于被用于计算在那个层中的每个激活。因此，我们期望对于我们每个特征会需要更多的权重来处理这个逐步增加的复杂度。这是我们在之前部分提到的同样内容的另外一个方法：当我们在网络中引入一个步长2卷积，我们应该也要增加通道的数量。

When writing this particular chapter, we had a lot of questions we needed answers for, to be able to explain CNNs to you as best we could. Believe it or not, we found most of the answers on Twitter. We're going to take a quick break to talk to you about that now, before we move on to color images.

在编写这一章节时，我们有很多问题需要回答，尽我们最大的努力来为你解释卷积神经网络。你相信吗，我们在Twitter上找到了绝大多数的答案。在我们开始彩色图像前，现在我们会做短暂的休息来和你讨论一下这方面的内容。

### A Note About Twitter

### Twitter的一条提问

We are not, to say the least, big users of social networks in general. But our goal in writing this book is to help you become the best deep learning practitioner you can, and we would be remiss not to mention how important Twitter has been in our own deep learning journeys.

至少对我们来说我们通常没有足够多的社交网络用户。但是在编写本书时我们的目标是帮助你能成为最好的深度学习行业人员，不提醒在我们的深度学习之旅中Twitter有多么的重要，这就是我们的失职。

You see, there's another part of Twitter, far away from Donald Trump and the Kardashians, which is the part of Twitter where deep learning researchers and practitioners talk shop every day. As we were writing this section, Jeremy wanted to double-check that what we were saying about stride-2 convolutions was accurate, so he asked on Twitter:

你会发现Twitter的一部分是深度学习研究人员和行业人员每天讨论本专业的地方，它的另一面是远离唐纳德·特朗普（Donald Trump）和卡戴珊（Kardashians）。在我们编写这一部分的时候，杰里米希望确认一下我们正在讨论的步长2卷积内容是否是准确的，所以他在Twitteer上做了提问：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00064.png" alt="twitter 1" width="500">
  </p>
  <p align="center"></p>
</div>

A few minutes later, this answer popped up:

几分钟后，这个答案突然出现了：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00065.png" alt="twitter 2" width="500">
  </p>
  <p align="center"></p>
</div>

Christian Szegedy is the first author of [Inception](https://arxiv.org/pdf/1409.4842.pdf), the 2014 ImageNet winner and source of many key insights used in modern neural networks. Two hours later, this appeared:

克里斯汀·塞格迪（Christian Szegedy）是 [Inception](https://arxiv.org/pdf/1409.4842.pdf)的第一作者，2014 ImageNet比赛获胜者及在现代神经网络中所采用的很多关键见解。两个小时后，出现了这个信息：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00066.png" alt="twitter 3" width="500">
  </p>
  <p align="center"></p>
</div>

Do you recognize that name? You saw it in <chapter_production>, when we were talking about the Turing Award winners who established the foundations of deep learning today!

你认识这个人吗？在<章节：产品>中我们讨论关于图灵奖获奖者奠定了今天深度学习的基础时，你看过这个名字。

Jeremy also asked on Twitter for help checking our description of label smoothing in <chapter_sizing_and_tta> was accurate, and got a response again from directly from Christian Szegedy (label smoothing was originally introduced in the Inception paper):

杰里米为了帮助检查在<章节：数据尺寸和测试数据增强>中我们对标签平滑的描述是否准确，也在Twitter上也做了提问，再次获得了克里斯汀·塞格迪（Christian Szegedy）直接回复（标签平滑最初是在Inception论文中引入的）：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00067.png" alt="twitter 4" width="500">
  </p>
  <p align="center"></p>
</div>

Many of the top people in deep learning today are Twitter regulars, and are very open about interacting with the wider community. One good way to get started is to look at a list of Jeremy's [recent Twitter likes](https://twitter.com/jeremyphoward/likes), or [Sylvain's](https://twitter.com/GuggerSylvain/likes). That way, you can see a list of Twitter users that we think have interesting and useful things to say.

很多顶级深度学习专家如今是Twitter的常客，且非常开放与广泛的社区交流。一个好的开始方法是查看[杰里米](https://twitter.com/jeremyphoward/likes)或[古格·西尔文](https://twitter.com/GuggerSylvain/likes)最近的Twitter喜好列表。那样，你能够看到Twitter用户的那些我们认为有意思和有益内容要讨论的列表。

Twitter is the main way we both stay up to date with interesting papers, software releases, and other deep learning news. For making connections with the deep learning community, we recommend getting involved both in the [fast.ai forums](https://forums.fast.ai/) and on Twitter.

Twitter是我们俩人紧跟最新感兴趣的论文、软件发布和其它深度学习新闻的主要方法。为了与深度学习社区建立联系，我们建议参与到[fast.ai](https://forums.fast.ai/)和Twitter这两者中来。

That said, let's get back to the meat of this chapter. Up until now, we have only shown you examples of pictures in black and white, with one value per pixel. In practice, most colored images have three values per pixel to define their color. We'll look at working with color images next.

那么，让我们返回到这个章节的实质部分。截至现在，我们仅仅展示了每个像素带有一个值的黑白图像例子。实践中，更多的彩色图像每个像素有三个值来定义它们的颜色。接下来我们看一下处理彩色图像。

## Color Images

## 彩色图像

A colour picture is a rank-3 tensor:

一张彩色图像是一个3阶张量：

```
im = image2tensor(Image.open(image_bear()))
im.shape
```

Out: torch.Size([3, 1000, 846])

```
show_image(im);
```

Out: <img src="./_v_images/bear1.png" style="zoom:100%;" />

The first axis contains the channels, red, green, and blue:

第一个轴包含通道，红，绿和蓝：

```
_,axs = subplots(1,3)
for bear,ax,color in zip(im,axs,('Reds','Greens','Blues')):
    show_image(255-bear, ax=ax, cmap=color)
```

Out: <img src="./_v_images/bear_channel.png" alt="bear_channel" style="zoom:100%;" />

We saw what the convolution operation was for one filter on one channel of the image (our examples were done on a square). A convolutional layer will take an image with a certain number of channels (three for the first layer for regular RGB color images) and output an image with a different number of channels. Like our hidden size that represented the numbers of neurons in a linear layer, we can decide to have as many filters as we want, and each of them will be able to specialize, some to detect horizontal edges, others to detect vertical edges and so forth, to give something like we studied in <chapter_production>.

我们看了卷积运算内容是对于图像一个通道上的一个过滤器（我们的例子在一个正方形上做的）。一个卷积层会取包含确定通道数的图像（对于常规的RGB彩色图像第一层是3个通道）和输出一个不同通道数的图像。像我们的隐含尺寸代表了在一个线性层中的神经元数量，我们可以决定有尽可能多的过滤器，且他们每一个会有专攻的能力，他们中的一些来探测水平边缘，另一些探测垂直边缘等等，来提供一些如我们在<章节：产品>中学习的那些内容。

In one sliding window, we have a certain number of channels and we need as many filters (we don't use the same kernel for all the channels). So our kernel doesn't have a size of 3 by 3, but `ch_in` (for channels in) is 3 by 3. On each channel, we multiply the elements of our window by the elements of the coresponding filter, then sum the results (as we saw before) and sum over all the filters. In the example given in <rgbconv>, the result of our conv layer on that window is red + green + blue.

在一个滑行窗口中，我们有确定数目的通道且我们需要尽可能多的过滤器（我们对所有的通道不会使用相同的卷积核）。所以我们的卷积核没有 3 乘以 3 的尺寸，`ch_in`（通道中）是 3 乘以 3 的尺寸。在每个通道上，我们窗口元素乘以相应的过滤器元素，然后加总结果（如之前我们所学）和加上所有的过滤器上。在图<RGB图像上的卷积>中给出的例子，在窗口上卷积层的结果是红+绿+蓝。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/chapter9_rgbconv.svg" id="rgbconv" caption="Convolution over an RGB image" alt="Convolution over an RGB image" width="550">
  </p>
  <p align="center">图：RGB图像上的卷积</p>
</div>

So, in order to apply a convolution to a color picture we require a kernel tensor with a size that matches the first axis. At each location, the corresponding parts of the kernel and the image patch are multiplied together.

所以，为了应用一个卷积到一个彩色图像上，我们需要一个匹配第一个轴维度尺寸的卷积核张量。在每个位置，卷积核和相应部分和图像部分是乘在一起的。

These are then all added together, to produce a single number, for each grid location, for each output feature, as shown in <rgbconv2>.

然后把它们所有加总起来，对于每个表格位置和每个输出特征都产出一个数值，如下图<添加RGB过滤器>所示。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/chapter9_rgb_conv_stack.svg" id="rgbconv2" caption="Adding the RGB filters" alt="Adding the RGB filters" width="500">
  </p>
  <p align="center">图：添加RGB过滤器</p>
</div>

Then we have `ch_out` filters like this, so in the end, the result of our convolutional layer will be a batch of images with `ch_out` channels and a height and width given by the formula outlined earlier. This give us `ch_out` tensors of size `ch_in x ks x ks` that we represent in one big tensor of four dimensions. In PyTorch, the order of the dimensions for those weights is `ch_out x ch_in x ks x ks`.

于是我们有了像这样的`ch_out`过滤器，所以最后，我们卷积层的结果会有带有之前给出概述公式的`ch_out`通道和高、宽的图像批次。这提供给我们在四维度的一个大张量中表示的`ch_in x ks x ks` 大小的`ch_out`张量。

Additionally, we may want to have a bias for each filter. In the preceding example, the final result for our convolutional layer would be $y_R + y_G + y_B + b$ in that case. Like in a linear layer, there are as many bias as we have kernels, so the biases is a vector of size `ch_out`.

此外，我们可能希望每个过滤器有一个偏置。在之前的例子中，这种情况下我们卷积层最终结果是 $y_R + y_G + y_B + b$ 。如线性层中那样，有与我们卷积核一样从的偏置，所以偏置是`ch_out`大小的向量。

There are no special mechanisms required when setting up a CNN for training with color images. Just make sure your first layer has three inputs.

当设置一个训练彩色图像的CNN不需要特别的机制。只要确保你的第一层有三个输入。

There are lots of ways of processing color images. For instance, you can change them to black and white, change from RGB to HSV (hue, saturation, and value) color space, and so forth. In general, it turns out experimentally that changing the encoding of colors won't make any difference to your model results, as long as you don't lose information in the transformation. So, transforming to black and white is a bad idea, since it removes the color information entirely (and this can be critical; for instance, a pet breed may have a distinctive color); but converting to HSV generally won't make any difference.

有很多个方法处理彩色图像。例如，你能够把彩色图像变为黑白色，从RGB变为HSV（色相，饱和度和值）的色彩空间，诸如此类。实验证明改变色彩编码对于你的模型结果不会产生任何差异，只要你不丢失变换中的信息。所以黑白色的变换不是个好想法，因为它移除了所有的色彩信息（这是很重要的，例如，宠物品种可能有不同的颜色），但是通常转换为HSV不会产生任何差异。

Now you know what those pictures in <chapter_intro> of "what a neural net learns" from the [Zeiler and Fergus paper](https://arxiv.org/abs/1311.2901) mean! This is their picture of some of the layer 1 weights which we showed:

现在你知道在<章节：概述>中来自[Zeiler和Fergus论文](https://arxiv.org/abs/1311.2901) 的那些图像“一个卷积网络学了什么”的意思了吧！下面是我们所展示的他们第一层权重的一些图像：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00031.png" alt="Layer 1 kernels found by Zeiler and Fergus" width="120">
  </p>
  <p align="center">图：由Zeiler和Fergus发现的层一卷积核</p>
</div>

This is taking the three slices of the convolutional kernel, for each output feature, and displaying them as images. We can see that even though the creators of the neural net never explicitly created kernels to find edges, for instance, the neural net automatically discovered these features using SGD.

这选取了卷积核的三个部分，并以图形的方式展示了它们。例如，我们甚至能够看到，卷积网络的创建者从来没有说明创建寻找边缘的卷积核，卷积网络使用SGD自动的发现这些特征。

Now let's see how we can train these CNNs, and show you all the techniques fastai uses under the hood for efficient training.

现在让我们看一下我们如何可以训练这些CNN，并给你展示为了有效训练在低层fastai使用的所有技术。

## Improving Training Stability

## 改进训练稳定性

Since we are so good at recognizing 3s from 7s, let's move on to something harder—recognizing all 10 digits. That means we'll need to use `MNIST` instead of `MNIST_SAMPLE`:

因为我们很好的识别了数字3和7，让我们继续做一些更难的事情吧，识别全部10个数字。这意味着我们会需要使用`MNIST`来替代`MNIST_SAMPLE`:

```
path = untar_data(URLs.MNIST)
```

```
#hide
Path.BASE_PATH = path
```

```
path.ls()
```

Out: (#2) [Path('testing'),Path('training')]

The data is in two folders named *training* and *testing*, so we have to tell `GrandparentSplitter` about that (it defaults to `train` and `valid`). We did do that in the `get_dls` function, which we create to make it easy to change our batch size later:

这个数据在两个名字为*training*和*testing*两文件夹中，所以我们必须告诉`GrandparentSplitter`关于这个信息（它莫热民为`train`和`valid`）。我们在`get_dls`函数做这个事情，我们创建这个函数是稍后用它很容易的改变我们的批次尺寸：

```
def get_dls(bs=64):
    return DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
        get_items=get_image_files, 
        splitter=GrandparentSplitter('training','testing'),
        get_y=parent_label,
        batch_tfms=Normalize()
    ).dataloaders(path, bs=bs)

dls = get_dls()
```

Remember, it's always a good idea to look at your data before you use it:

记住，在你使用数据前总是看一下它是一个好主意：

```
dls.show_batch(max_n=9, figsize=(4,4))
```

Out:![img](data:image/png;base64,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)

Now that we have our data ready, we can train a simple model on it.

现在我们已经准备好我们的数据了，我们能够用它训练一个简单的模型。

### A Simple Baseline

### 一个简单的基线

Earlier in this chapter, we built a model based on a `conv` function like this:

本章早些时候，我们基于下面的`conv`函数创建了一个模型：

```
def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res
```

Let's start with a basic CNN as a baseline. We'll use the same one as earlier, but with one tweak: we'll use more activations. Since we have more numbers to differentiate, it's likely we will need to learn more filters.

让我们以基础CNN作为基线开始。我们会使用先前同样的方法，但是会有一个调整：我们会使用更多的激活。因为我们有更多的数据来区别，这可能我们需要学习更多的过滤器。

As we discussed, we generally want to double the number of filters each time we have a stride-2 layer. One way to increase the number of filters throughout our network is to double the number of activations in the first layer–then every layer after that will end up twice as big as in the previous version as well.

就像我们之前讨论过的，我们通常希望每次有一个步长2层双倍过滤器的数量。一种增加整个我们的网络过滤器数量方法是在第一层双倍激活的数量，然后之后的每层激活结果也会之前版本的双倍大小。

But there is a subtle problem with this. Consider the kernel that is being applied to each pixel. By default, we use a 3×3-pixel kernel. That means that there are a total of 3×3 = 9 pixels that the kernel is being applied to at each location. Previously, our first layer had four output filters. That meant that there were four values being computed from nine pixels at each location. Think about what happens if we double this output to eight filters. Then when we apply our kernel we will be using nine pixels to calculate eight numbers. That means it isn't really learning much at all: the output size is almost the same as the input size. Neural networks will only create useful features if they're forced to do so—that is, if the number of outputs from an operation is significantly smaller than the number of inputs.

但是这个方法也有一个小问题。考虑到卷积核被应用于每个像素。默认情况下，我们使用 3×3 像素卷积核。这表示卷积核应用于每个位置上总共有  3×3 = 9 个像素。先前，我们的第一层有四个输出过滤器。这表示在每个位置上的9个像素计算出四个值。思考一下如果我们加倍这个输出到8个过滤器会发生什么。那么当我们应用我们的卷积核时，我们会使用9个像素来计算8个数值。 这表示完全没有真正学习到太多内容：输出尺寸几乎与输入尺寸一样大。神经网络只有在被强迫这样做时，它们才只创建有用的特征。即，如果运算的输出数量比输入数量显著小。

To fix this, we can use a larger kernel in the first layer. If we use a kernel of 5×5 pixels then there are 25 pixels being used at each kernel application. Creating eight filters from this will mean the neural net will have to find some useful features:

在第一层我们能够使用一个更大的卷积核来修正这个问题。如果我们使用一个 5×5 像素的卷积核，那么每个卷积核的应用就会有 25 个像素被被使用。由此创建8个过滤器这表示神经网络必定会找到很多有用的特征：

```
def simple_cnn():
    return sequential(
        conv(1 ,8, ks=5),        #14x14
        conv(8 ,16),             #7x7
        conv(16,32),             #4x4
        conv(32,64),             #2x2
        conv(64,10, act=False),  #1x1
        Flatten(),
    )
```

As you'll see in a moment, we can look inside our models while they're training in order to try to find ways to make them train better. To do this we use the `ActivationStats` callback, which records the mean, standard deviation, and histogram of activations of every trainable layer (as we've seen, callbacks are used to add behavior to the training loop; we'll explore how they work in <chapter_accel_sgd>):

正如我们马上要看到的，在模型训练期间我们能够查看它们的内部，主了尽量找方法以便模型训练的更好。我们使用`ActivationStats`回调来做这个操作，它记录了平均值、标准偏差和每个可训练的层的激活直方图（我们已经学习过，回调被用于添加行为到训练循环中，我们会在<章节：加速随机梯度下降>中探索它们是如何运行的）：

```
from fastai.callback.hook import *
```

We want to train quickly, so that means training at a high learning rate. Let's see how we go at 0.06:

我们希望快速训练，所以这表示在高学习率下训练。让我们看一下在 0.06 的学习率下怎么样：

```
def fit(epochs=1):
    learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy,
                    metrics=accuracy, cbs=ActivationStats(with_hist=True))
    learn.fit(epochs, 0.06)
    return learn
```

```
learn = fit()
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   2.307071 |   2.305865 | 0.113500 | 00:16 |

This didn't train at all well! Let's find out why.

训练的一点都不好！让我们找出原因。

One handy feature of the callbacks passed to `Learner` is that they are made available automatically, with the same name as the callback class, except in `snake_case`. So, our `ActivationStats` callback can be accessed through `activation_stats`. I'm sure you remember `learn.recorder`... can you guess how that is implemented? That's right, it's a callback called `Recorder`!

回调传递给`Learner`的一个有用的特性是它们可以自动获取，并与回调类同名，在`snake_case`中除外。所以我们的`ActivationStats`回调能够通过`activation_stats`获取。我确信你记得`learn.recorde`...你能够猜出这是如何实现的吗？是的，它的回调称为`Recorder`！

`ActivationStats` includes some handy utilities for plotting the activations during training. `plot_layer_stats(idx)` plots the mean and standard deviation of the activations of layer number *`idx`*, along with the percentage of activations near zero. Here's the first layer's plot:

`ActivationStats`包含一些在训练期间对于激活绘图有用的工具。`plot_layer_stats(idx)`绘制`idx`层激活的平均值和标准偏差，连同近零激活百分比。下面是第一层的图：

```
learn.activation_stats.plot_layer_stats(0)
```

Out:![img](data:image/png;base64,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)

Generally our model should have a consistent, or at least smooth, mean and standard deviation of layer activations during training. Activations near zero are particularly problematic, because it means we have computation in the model that's doing nothing at all (since multiplying by zero gives zero). When you have some zeros in one layer, they will therefore generally carry over to the next layer... which will then create more zeros. Here's the penultimate layer of our network:

通常我们的模型在训练期间应该用一个连续或至少平滑的层激活的平均值和标准偏差。零附近的激活是特别有问题的，因为它表示我们在模型中有完全没有做任何事情的计算（因为乘以零等于零）。当我们在一个层中有一些零，因而它们通常会继续存在在下个层...那么它会产生更多的零。下面是我们网络的倒数第二层：

```
learn.activation_stats.plot_layer_stats(-2)
```

Out:![img](data:image/png;base64,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)

As expected, the problems get worse towards the end of the network, as the instability and zero activations compound over layers. Let's look at what we can do to make training more stable.

正如预期的那样，这些问题在网络的最后变得更糟，在这些层上由不稳定和零激活组成。让我们看一下，我们能做什么以使得训练更加稳定。

### Increase Batch Size

### 增加批次尺寸

One way to make training more stable is to increase the batch size. Larger batches have gradients that are more accurate, since they're calculated from more data. On the downside, though, a larger batch size means fewer batches per epoch, which means less opportunities for your model to update weights. Let's see if a batch size of 512 helps:

一个使得训练更加稳定的方法是增加批次尺寸。更大的批次有更加精准的梯度，因为我们的计算来自更多数据。然而这个方法的缺点是，一个更大的批次尺寸意味着每个轮次更少的批次数，它表示对于你的模型有更少的机会更新权重。让我们看一下如果一个尺寸是512有什么帮助：

```
dls = get_dls(512)
```

```
learn = fit()
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   2.309385 |   2.302744 | 0.113500 | 00:08 |

Let's see what the penultimate layer looks like:

我们看一下倒数第二层是什么样子：

```
learn.activation_stats.plot_layer_stats(-2)
```

Out:![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAs8AAADWCAYAAAAuNG/NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deZxjZZn3/8+VpPalt6re9wWFRpuBVpRNRnTEhRco+uiACM4oDojLOOrMqDgt7j6j8/PxQQRHRBERR1kfEMdRUMAFGqWFBuyGhqYbet+rqmtJcv3+OCfVqXSqKtV1qpLU+b5fr/Oq5OTOyVXpujvXuXOd+zZ3R0REREREhpcodwAiIiIiItVCybOIiIiISImUPIuIiIiIlEjJs4iIiIhIiZQ8i4iIiIiUSMmziIiIiEiJlDyLiMScmZ1uZm5mc8sdi4hIpVPyLCIyAZnZ3DAhPr3csYhUAjN7h5k9a2Z7zey7ZlaT91jSzP5gZm8vZ4xSHZQ8i4iIyIRmZtOA7wKfBE4FTgIuzmvyEeAFd7+pDOGVzMxqJ+JrVRslzzFhZvea2XfM7HNmtj088/68mSXM7NNmts3MdpjZ5/OekzKzVWb2jJl1m9laM3tfwXE/ZGaPmFmHmW01sx+Z2ay8x3NfB7/WzH5jZl1m9riZvW48f3+RicrMTjGzB8zsQLitCfvXprDJPWEffDbvOR8ws81hf/w5ML8csYuMo8XAPne/wd0fBW4FjgEws2XAh4BLSjlQ+Ln4lJmdbWZPmlmnmd1jZksK2p1gZv8dfj7uMLObzWxB3uOLwn0vhH3xUTO7oOAYuc/uz5rZFuD5QWK6N+znhdtFeW0+EMbbbWbrzeyTZpbKe/zZMEf4ppntAh4I988KP9v3mtnB8LVWlvJeTVRKnuPlrUANcArBWfYngP8HNBOciX8U+ISZvT5s/5/AW4D3AUcDVwBfNrO/LzjuR4GXAG8m+BD+UZHX/nfgC8AKYDVwk5lNjuw3E4khM0sCtwN/AI4Pt1VAV3gb4FxgFvCy8DlnA/8BfA04Dvgx8L/HM26RMngKaDSzlWbWBLwK+JOZGfAd4BPuvnUEx5tFkGyfTzCKPRm4NvegmR0D/Br4HbASeDWQAX5hZvVhs2bgl8CZBJ+h1wDfNbO/Lnit/wW0A2eExynmLWFMue1zQCfwUBjPKoLP6n8l+Dz/EMFn+78VHOeDwHbglcCF4ftzK/Bi4E3Ay4Ft4e/RNuQ7NJG5u7YYbMC9wCMF+9YCjxbsW0OQ6C4CssCLCx7/dOFxCh7/K8CBOeH908P7b8lrMzPc97pyvy/atFXzBkwJ+9LpRR6bW+wx4H7ghoJ9/x62nVvu30mbtrHagLPCz7gNBCeQSeADwJ3h59Kt4WPfB5qHOM4qIA205+17R/iZWR/evw74UcHz6ghObM8Z4ti3Ad/Ou38vsA5IjOD3fB3QA5wV3m8MX/fMgnbvAvbm3X8W+GVBmzPC/xuOKfg9tgCfLve/abm2/uF6iYU1Bfe3hlvhvukEZ8oGrA5OPPulCM6egaAsg+BM9hiCM+/ctxkLGPj10iO5G+6+1cwywIwj/D1EBHD3PWb2n8DPzexXBCNdt7j7X4Z42jHAjQX77gf+aYzCFKkI7n4HcEfuvpktBP4FOBH4P8ATBN/Q/gC4HPjnIQ73grvvyLv/PMFn5nTgOYJvepaaWUfB8+qBZeHrNxIMSJ1FMFpcS5CY3lPwnIfdPVvK72hmywm+Tfrn8PcFWA40AD81M89rngTqzaw973d5sOCQy4Fd7v54boe795jZH8LHYknJc7z0Fdz3QfYlOJQEn0RwxlrYBjObD9wFXE9Q0rGTYLTrfwj+E8jXWyQelQ2JjJK7v9fMvg78DfBa4LNmdhnBaNqgTxuX4EQq27eBVe6+2cxeA3zG3dNm9gPgM8M8t/AzLdenEnk/rwe+VOS5u8Kf/xs4m+DE9UmCMouvApMK2ncO94sAmNl0glLMH7j7/5f3UC6mtxGMYhfaPcxrFfv/wgbZHwtKnmUwD4c/57v7/xukzcsIzmY/7O4HIbhAYjyCE5FD3P0x4DHga2b2LYJZBG4JH04WNH8cOBn4Zt6+k8c8SJEKYmbvBczdvx3uShBcEwTB4M9oB3dWAy8Fnvaw1qGI0whKqG4KY0oARxHUFI+ImdUR9PknCeqW860FuoHF7n7XCA+9Fmgzs2Nyo8/ha72cgf+HxIqSZynK3Z8ys2uBb5vZxwkuemgCTiCo8/oysJ7gzPOfzOwGgosBP12umEXixsyWAu8l+Cp6EzCb4OLfPxJ8E9QB/I2ZrQV63H0PwcjWf5nZgwTfHJ0CXFDk8CITkpnNIbhQ7pS83b8BPhLOOPV+ghKo0fgCQQnED8JvhnYAC4FzgK+7+wbgL8DZZvZTgr76EYI+POLkGbg6fO67gfa8cst97t5hZl8AvhDu/wVB/vcS4K/cfajylF+Fv8cPzez9wD6CkpZ64KojiHNC0NfmMpSLCS6q+CTBaNUvgQsJLqjA3f9McLHF+8LHPwp8uCyRisRTJ0H95I8Ivo79KfBb4LKwRvL9BFfqbwL+BODutxB8Tfxx4M8EswUM9eEpMtFcDXzJ3Z/N2/dBguR2NUG/Gq5sY0ju/gRB2WMz8HOCz8hvE3xbuzds9o/ARoIa518S1E3/5Ahf8nSC+P9CcDFfbnt7GM9nw9d7D8H1T/eH958d5vdwgoT/SYJSsIcILq58rbvvPMJYq54N/m2CiIiIiIjk08iziIiIiEiJlDyLiIiIiJRIybOIiIiISImUPIuIiIiIlEjJs4iIiIhIiapqnue2tjZfuHBhucMQqRgPP/zwTndvL3ccxai/igxUyf0V1GdFCg3WZ6sqeV64cCGrV68udxgiFcPMNpY7hsGov4oMVMn9FdRnRQoN1mdVtiEiIiIiUiIlzyIiIiIiJYo0eTazqWZ2i5l1mtlGMztvkHZmZl82s13h9hXLW4hdREREDjGzy8xstZn1mNl1w7T9RzPbamb7zOxaM6sbpzBFYiHqkecrgV5gBnA+cJWZLS/S7mKCtdJXAC8F3gS8L+JYREREJooXgM8B1w7VyMxeB/wLcAawEFgMfGasgxOJk8guGDSzJuBc4Fh37wDuN7PbgQsIOnK+C4Gvuvvm8LlfBd4LfGs0MXz7NxvoSWe47NXLRnMYkVgws8uAi4CXADe6+0WDtLsI+A5wMG/3m9z93ijiuOvRLcxoreeEBVOiOJzIhOTuNwOY2Upg7hBNLwS+4+5rw/afBW7g8M/hMffbp3Zy12NbSJjxwTOW0dZ8ZAPg2/Z3c9W9T9PZk444Qomz179kJq9+8Ywjem6Us20cBWTcfV3evjXAq4q0XR4+lt+u2Aj1iDzw9E52d/YqeRYpTW4k63VAwzBtf+fup4xFEF+5+0leOneykmeRaCwHbsu7vwaYYWbT3H1XYWMzu5jg22Dmz58fWRDP7eri77+3moRBZ2+G6S11/Z/NG3d1sn5bB2ccPZ3BKjY7etLc9sjzHOhO85/3bWB/d5q2ptrI4hN5ydxJR/zcKJPnZmBfwb59QEsJbfcBzWZm7u75DUfSsRtrk2zekxlh2CLxNIKRrDGVcSeT9eEbikgpin2+QvBZfFjy7O7XANcArFy5MpKOmM06H/3JGlIJ478/chqX3vBH7l67lctevYy7Ht3Cx3/yZzp60pxz3Gz+18p5UJA/H+zN8Pm7nmDDjk4Alk1v5sb3voJlM4qlEyLjL8rkuQNoLdjXChwooW0r0FGYOMPIOnZDTYqDvUqeRcbAX5nZTmA3cD3wRXeP5DvUbBayh3d9ETkyxT5fofhn8Zj41ZPbefCZ3XzpLS9h1qQGXn/sTL5w15Pc/MfNfOTHazhu3mROXjqNb977NLc+8kLRY7Q113HDe07k2DmTaKlLkUhoTgGpHFEmz+uAlJktc/f14b4VwNoibdeGjz04TLsRaaxN0tWrmiiRiP0GOBbYSPCV8E1AGvhiYcMj+Qo4k9XIs0iEcp+vPw7vrwC2FSvZGCt3PbqFSQ01nHtC8IXWmctn8YW7nuSj/7WGOZMbuOE9J9JUl+KtJ8xj2/7uosc4emYrkxprxitkkRGJLHl2904zuxm4wszeAxwHnA2cVKT594GPmNldgAP/BHxjtDEEybNGnkWi5O4b8u4+amZXAB+jSPJ8JF8BZ9018iwyDDNLEXxmJ4GkmdUD6SLfAH0fuM7MbgC2AJ8CrhuvOHvTWX7xxDZet3wmNclgQq/50xo5ZlYrj2/Zz+fffCxNdUHqsaitiUVtTeMVmkhkop6q7lKCC4+2AzcCl7j7WjM71cw68tpdDdwBPAo8BtwZ7huVhtokPemsRrFExpZzWJXikcuq5lmkFJ8imPHmX4B3hrc/ZWbzzazDzOYDuPvdwFeAewi+LdoI/Nt4BfnA0zs50J3m9cfOHLD/42e+iE++4WhOf9H08QpFZMxEWbaBu+8mmL+5cP99BBcx5O478PFwi0xjbRKAg30Zmusi/dVEJpxSR7LM7PXAH919m5m9GLgc+K+o4shknYxyZ5EhufsqYNUgDzfn33H3rwFfG+OQirr70a0016U4ZVnbgP2nv2i6EmeZMCbU8tyNtUHCrLpnkZKUNJJFsNjCn82sE7gLuBn4QlRBZB2KXCssIlVm/bYD3PKn53njS2ZRl0qWOxyRMTOhhmf7R55V9ywyrFJHstz9o8BHxyqOrC4YFKl6mazzsZ/8maa6JB8780XlDkdkTE2wkecgedZFgyLVQ/M8i1S//3liG49s2svlbzrmiFcSFKkWEyp5bugv21DyLFItNNuGSPV7answJ8CZBRcKikxEEyp5PjTyrJpnkWoRLJJS7ihEZDQ27e6irbm2/9ojkYlsQiXPDTUq2xCpNirbEKl+m/Z0MXdKY7nDEBkXEyp51gWDItVHZRsi1W/T7oPMm6rkWeJhgiXPqnkWqSbujjsaeRapYpms88Leg8yb0lDuUETGxYRKnhtU8yxSVXJJs5Jnkeq1Zd9B0lnXyLPExoRKnlW2IVJdcjmzqjZEqtem3QcBmKeaZ4mJCZU81yQT1CSNrj4lzyLVIFfrnFH2LFK1Nu3pAmDeVJVtSDxMqOQZghk3NPIsUh1y5RpZlW2IVK3Nu7tIGMyerORZ4mHCJc+NtSnVPItUCY08i1S/TXsOMmtSAzXJCZdSiBQ14f7SG2uTmm1DpEpks+FPJc8iVWvT7i7maqYNiZEJlzw31KpsQ6Ra5Eacc0m0iFSf53Z3MV8zbUiMTLjkWSPPItWjv2xDNc8iVelAdx/bD/SwqL2p3KGIjJsJmDyn6OrL8Ksnt3H1r58udzgiMoTchYKqeRapTht2dAKwpL25zJGIjJ8JmDwnOdib5vrfbeTLdz/Jtv3d5Q5JRAZxqGxDybNINdqwswOAJRp5lhiZcMlzQ22Szp4Mz+3uIutwy5+eL3dIIjKIXM6sCwZFqtPT2ztJJoz5U5U8S3xMuOQ5qHlOs3lPsOLRTx7ejOuDWaQiZbU8t0hV27Czg/lTG6lNTbh0QmRQE+6vvbE2xZ6uPnrSWZbPbuWp7R08+MzucoclIkX0L5Ki3FlkWGY21cxuMbNOM9toZucN0m6ymX3PzLaH26qxiunp7Z0q2ZDYiSR5LrVDh23/2szuMbN9ZvZsFK+fr6Em2X/7H161hBmtdfz991bzqye3Rf1SIjJKmm1DZESuBHqBGcD5wFVmtrxIu/8AGoGFwMuBC8zs3VEHk8k6z+zqZLEuFpSYiWrkudQODdAJXAt8LKLXHqCx9lDyfMzsVm59/8nMm9rIB298RB/QIhUmlzyr5llkaGbWBJwLXO7uHe5+P3A7cEGR5mcBX3H3Lnd/FvgO8HdRx/T8noP0prMsbtPIs8TLqJPnEXZo3P1Bd78e2DDa1y4mlzybwZzJDcya1MBFJy2goyfN5j1dY/GSInKEMlphUKRURwEZd1+Xt28NMNhAlRXcPrZoI7OLzWy1ma3esWPHiAJ6OjfTxnSNPEu8RDHyPNIOPSIj7dgNtSkAZrbWUx+WcCyb0QLAum0dUYQkIhFR2YZIyZqBfQX79gEtRdreDfyLmbWY2VKCUeeiSwC6+zXuvtLdV7a3t48ooAfW7wQ0x7PETxTJ80g69IiNtGPnRp7nTTn0/8Sy8Kx4/fYDUYQkIhHJv2BQs+KIDKkDaC3Y1woU+2D7IHAQWA/cBtwIbI4ymGd2dvK93z3LW0+Yy9Sm2igPLVLxhk2ezexeM/NBtvsZWYcecw255HnqoeS5pb6GWZPqWa+RZ5GKkl+uocFnkSGtA1Jmtixv3wpgbWFDd9/t7ue7+0x3X07wWf9glMF8/s4nqEsl+fiZL4rysCJVITVcA3c/fajHw5rnlJktc/f14e6iHXo8NNbkkueGAfuXzWjRyLNIhclPmLPuJAeUaYpIjrt3mtnNwBVm9h7gOOBs4KTCtma2BNgbbn8DXAy8Ksp47v3Ldi545QKmt9RHeViRqjDqsg137wRyHbrJzE4m6NDXF2tvZgkzqwdqgrtWb2aRfefTVBecD8yfOrC866jpzTy1vUPLAItUkPxaZ9U9iwzrUqAB2E5QinGJu681s1PNLP+r1ROARwm+Af4icL67Rzag5e6ks05LfU1UhxSpKsOOPJfoUoLp57YDuwg7NICZnQr8zN1zVxScBtyT99yDwK+B06MI5JhZraw66xjOPHbmgP3LZjTT3Zdl054uFkzTtDoilWBg2YaSZ5GhuPtu4Jwi++8juP4od//HwI/HKo7ceW4qoW+KJJ4iSZ4H69DhY4Wd+l4Yu+9mEwnjopMXHbY/N+PG+m0dSp5FKkRWI88iVSfXV5NKniWmJtzy3INZGs64sU51zyIVI6MLBkWqTi55TpiSZ4mn2CTPrfU1tLfUsXGnFkoRqRTZbP5tZc8i1SB30quyDYmr2CTPAAunNfLMrs5yhyEiofw654xqnkWqQiYTjjwreZaYilXyvGBaExuVPItUjAFlGxp5FqkKGnmWuItV8rxwWiPb9vfQ1ZsudygiQsEFgxp5FqkK6bDeSiPPElfxSp7bglk2ntutumeRSjBwkZTyxSEipctdq6CRZ4mreCXP4RR1z+5U6YZIJcifnk5lGyLVIfctUVKzbUhMxSp5nj8tWHXw2V0aeRapBAMuGFTyLFIVchcMap5niatYJc+t9TVMa6rVRYMiFUKzbYhUn/6RZyXPElOxSp4BFkxr5BmVbYhUhPzRZlfyLFIVMmHRs5JniavYJc8L25rYqLINkYowsGyjjIGISMlyfVXJs8RV/JLnaU1s2ddNd1+m3KGIlJWZXWZmq82sx8yuG6btP5rZVjPbZ2bXmlldFDHkrzCommeR6tA/VZ0uGJSYil3yPHdKAwDP7z1Y5khEyu4F4HPAtUM1MrPXAf8CnAEsBBYDn4kigAGLpKhsQ6QqaKo6ibsYJs/BjBub9yh5lnhz95vd/VZg1zBNLwS+4+5r3X0P8FngoihiGLBIikaeRSpWNuvsOBAsMpZWzbPEXAyT53DkWcmzSKmWA2vy7q8BZpjZtNEeeOAiKUqeRSrV/u4+Xvb5/+Gmhzb191UlzxJXsUueZ7TWk0oYm/fookGREjUD+/Lu5263FDY0s4vDOurVO3bsGPbAKtsQqQ65RDmTddKa51liLnbJczJhzJpcr7INkdJ1AK1593O3DxQ2dPdr3H2lu69sb28f9sADyzZGGaWIjJlUIkgX0lnXPM8Se7FLngHmTm7UBYMipVsLrMi7vwLY5u7D1UoPSysMilSH/JHnXF9V8ixxFc/keUqDyjYk9swsZWb1QBJImlm9maWKNP0+8PdmdoyZTQE+BVwXRQxaJEWkOuRm1ujLZJU8S+zFMnmeM6WBbft76ElrrmeJtU8BBwmmoXtnePtTZjbfzDrMbD6Au98NfAW4B9gYbv8WRQBanlukNGY21cxuMbNOM9toZucN0q7OzL5lZtvMbLeZ3WFmc0b7+omEkbDghLf/gkHN8ywxFUnyXGqnDtt+zMweM7MDZvaMmX0sihhGIjdd3Za93eP90iIVw91XubsVbKvc/Tl3b3b35/Lafs3dZ7h7q7u/2917ooghv1JDZRsiQ7oS6AVmAOcDV5nZ8iLtPgS8EngpMBvYC3wjigBSiQRpXTAoEtnIc6mdGsCAdwFTgDOBy8zsHRHFUZLcdHW6aFCkvPITZs22IVKcmTUB5wKXu3uHu98P3A5cUKT5IuDn7r7N3buBHxFMNzlqyYQNHHlW8iwxNerkeYSdGnf/irv/0d3T7v4X4Dbg5NHGMRKHkmfVPYuUk2bbECnJUUDG3dfl7VtD8aT4O8DJZjbbzBoJBrR+FkUQqYSRzjhp1TxLzEUx8jySTj2AmRlwKsHV/ONmZms9yYRp5FmkzLRIikhJCudaJ7x/2FzrwDrgOeB5YD9wNHDFYAceydzsyaSRyeqCQZEokueRdOpCq8IYvjtYg5EuulCKVDLBzNZ6XtB0dSJlNWCRFNU8iwymcK51wvuHzbUOXAXUA9OAJuBmhhh5Hsnc7KmEBfM8Z3XBoMTbsMmzmd1rZj7Idj8j69T5x72MoPb5jUNdfDTSRRdKNXtyveZ6FimzAWUbGnkWGcw6IGVmy/L2raD4t7YrgOvcfXf42foN4OVm1jbaIJJh2YZGniXuhk2e3f30Ilfk57ZTGFmnBsDM/o5geqwz3H3zaH+JIzF7cgMv7FPyLFJOWiRFZHju3kkwgnyFmTWZ2cnA2cD1RZo/BLzLzCaZWQ1wKfCCu+8cbRy52TaUPEvcjbpsY4SdGjM7H/gC8Fp33zDa1z9Ssyc3sHVft74qFimj/NFmDTyLDOlSoAHYDtwIXOLua83sVDPryGv3UaAbWA/sAN4AvDmKAFK5muews6aUPEtMFVtN7EhcClxL0Kl3EXZqADM7FfiZuzeHbT9HUIv1kB2ql/qBu/9DRLGUZPbkBvoyzs6OHqa31o/nS4tIaOBsG8qeRQbj7ruBc4rsv4/g2qPc/V0EM2xELllQ85xQ8iwxFUnyPFinDh8r7NiLonjN0ZozOUiYn997UMmzSJkMWCRFQ88iFS0VzvOcS5418ixxFcvluSEYeQZ4QasMipTNgEVSNPIsUtGSBTXPGnmWuIpt8jxrUi551kWDIuUy4IJBjTyLVDSNPIsEYps8t9anaK5Labo6kTLKT5418CxS2ZIJoy9z6ILBhOZ5lpiKbfJsZsyeXM8WTVcnUjaZ7KHprlS2IVLZ+keeM5qqTuIttskzhHM9q+ZZpGyyWacmGXwAa7YNkcqWSoazbbhWGJR4U/Kssg2Rssm6U5NI9N8WkcqVSiT6a57NdMGgxFesk+c5kxvY1dlLd1+m3KGIxFLGnVQ48qzkWaSy5c/zrIsFJc5inzwDbNrdVeZIROIpKNsI/hvKZMscjIgMKah5zpLJui4WlFiLdfK8dHqwdstT2zuGaSkiYyHr9CfPGnkWqWzJhJHOaORZJNbJ85L2ZsxgvZJnkbLIuC4YFKkWuQsG01lXvbPEWqyT54baJHOnNCh5FimTbNZJaeRZpCokwwsGs66RZ4m3WCfPAMumt7B+24FyhyESS/kfwprnWaSypRJGOpslnXXN8SyxFvvkeen0Zjbs7CStq5VExl0mGyxYlEyYlucWqXCphJHJOFklzxJzSp6nN9ObzrJpj+Z7FhlvWXeSiWCxBZ2/ilS2/kVSsq4FUiTWYp88Lwtn3FDphsj4y3rwIZxIqOZZpNIlc8tz64JBibnYJ8+56ep00aDI+AtWKjOSZqp5FqlwqUSif3luXTAocRb75LmlvoZZk+o117NIGQRlG0bCVPMsUumCeZ6zmqpOYi/2yTPA4vYmNuxQ8iwy3rJZwrINjTyLVLpUuDx3VoukSMwpeQYWtwUzbrhGvkTGVcYdMzTbhkgVyNU8p7U8t8SckmdgUVsTB7rT7OrsLXcoIrGSm/IqYYYGnkWGZmZTzewWM+s0s41mdt4g7X5mZh15W6+ZPTra108lE/2zbaSSSp4lvlLlDqASLGpvAuCZnZ20NdeVORqR+DhU86xFUkRKcCXQC8wAjgPuNLM17r42v5G7vz7/vpndC/xqtC+eK9Xoy2Q1VZ3EWiQjz6WeDYdtP2xmG8xsv5m9YGb/YWZlTeKXtAUzbqjuWWR8ZTxvkRQlzyKDMrMm4FzgcnfvcPf7gduBC4Z53kLgVOD60caQWxilJ53VIikSa1GVbeSfDZ8PXGVmywdpewdwvLu3AscCK4APRhTHEZkzpYGapLFhZ2c5wxCJnWzWSRqabUNkeEcBGXdfl7dvDTDYZ23Ou4D73P2Z0QaQUvIsAkSQPI/0bNjdn3b3vbmnA1lg6WjjGI1kwlgwrYlndih5FhlPubKNpGbbEBlOM7CvYN8+oGWY570LuG6wB83sYjNbbWard+zYMeSB+kee+zJKniXWohh5HvHZsJmdZ2b7gZ0EI89XD9G25I49GovbmnhGI88i4yq3SErC0AWDIkPrAFoL9rUCgy6Pa2anADOBnwzWxt2vcfeV7r6yvb19yAByI8+9GnmWmIsieR7x2bC7/zAs2zgK+BawbYi2JXfs0VjU3sTGXV2quxQZR4eW51bZhsgw1gEpM1uWt28FsHaQ9gAXAje7eyQX9CSTQcrQk85qqjqJtWGTZzO718x8kO1+juBsOMfd1xN0/G8eSfBRWtzWRG8my+Y9XeUORSQ2sh58FazluUWG5u6dwM3AFWbWZGYnA2czyIWAZtYAvI0hSjZGqiY38pzJapEUibVhk2d3P93dbZDtFI7sbDhfClgy8tCjtXz2JAD++NyeMkciEh/ZbN4iKUqeRYZzKdAAbAduBC5x97VmdqqZFY4un0PwLfA9Ub24ap5FAqMu2ziCs+H3mNn08PYxwL8CvxxtHKN1zKxWpjbV8pt1O8sdikhsZFyLpIiUyt13u/s57t7k7vPd/Yfh/vvcvbmg7Y3uvsAjXDo3tzBKb0Y1zxJvUU1VV/RsGKDIGfHJwKNm1gncFW6fiCiOI5ZIGKcsbeO+9Tv09bHIODlU8xzcFpHKlUwcqnlW8ixxFsniJO6+m+AromKP3UdwUWHu/rujeM2xcNpR7SHQsxQAABkOSURBVNy+5gWe2Lq/v4xDRMZONhsukmIq2xCpdLk6Z/dDibRIHOmvP89py9oAVLohMk4yWSeZCL750cizSGXLH21OauBZYkzJc57prfW8eGYL960fu/mkRSqFmU01s1vMrNPMNprZeYO0W2VmfWbWkbctjiKG/kVSNPIsUvHyZ9jQyLPEmf76C5y0pI2HN+6hJ50pdygiY+1KoBeYAZwPXGVmgy1udJO7N+dtG6IIIOu5RVI08ixS6QaMPCt7kBjTn3+BVyyeSk86y5pNheu+iEwcZtYEnAtc7u4d7n4/cDtwwXjGkcnmXTCYHc9XFpGRqsnLmDXyLHGmv/4CL180FTP4/YZd5Q5FZCwdBWTcfV3evjXAYCPPZ5nZbjNba2aXDHZQM7vYzFab2eodO4Yvf+pfJEUrDIpUPI08iwT0519gcmMtR89sVfIsE10zwQIK+fYBLUXa/hg4GmgH3gt82sz+tthB3f0ad1/p7ivb29uHDSK3SEpCNc8iFS+/5jmlkWeJMf31F/GKxdNU9ywTXQfQWrCvFThQ2NDdH3f3F9w94+6/Bb4OvDWKIDK5eZ7NiHAtBxEZA/kjzwnTdBsSX0qei1Dds8TAOiBlZsvy9q0A1pbwXAci+eTsn21DZRsiFS9/tFllGxJn+vMv4oQFUwD403N7yhyJyNhw907gZuAKM2sys5OBs4HrC9ua2dlmNsUCLwc+CNwWRRy5RVKCso0ojigiYyWpqepEACXPRU1rrmPe1AYe2bS33KGIjKVLgQZgO3AjcIm7rzWzU82sI6/dO4CnCEo6vg982d2/F0UAGQ8WSUkmgvpnEalcqaQuGBSBiJbnnoiOmzeF1c/uLncYImPG3XcD5xTZfx/BBYW5+0UvDoxCNqx5VtmGSOXTIikiAf31D+K4eZPZsq+bbfu7yx2KyITk7rgHZRumRVJEKt6AmmddMCgxpuR5EMfNmwyg0g2RMZKbmi63PLfKNkQqWzKvbCO/hEMkbpQ8D2L57FZSCVPyLDJGcrmyZtsQqQ4pTVUnAih5HlR9TZKjZ7XyyHNKnkXGQq5MI7dIipbnFqlsyQGLpCh5lvhS8jyE4+dPZs3mvaQ1h5ZI5PrLNsxIGKp5FqlwA0aelTxLjCl5HsIJC6fS1ZvhiS2HLbomIqOUS5b7yzZU8yxS0TTyLBJQ8jyEly0MFkt5SFPWiUQuV6ZhZiQSmm1DpNLV5E3urJFniTMlz0OYNamBOZMbeHijVhoUiVruAsGkBaUbGnkWqWwDVhjUBYMSY0qeh7Fy4RQeenY3rlExkUjlRpoTKtsQKYmZTTWzW8ys08w2mtl5Q7Q93sx+Y2YdZrbNzD402tfPT5hVtiFxFlnyPJJOnfecWjN70sw2RxVH1FYunMr2Az1s3nOw3KGITCi5eZ0TZpiBzk9FhnUl0AvMAM4HrjKz5YWNzKwNuBu4GpgGLAX+e7QvnkgEF/fmbovEVZQjzyV16gIfA7ZHGEPkVi4I6p4ffEZ1zyJRyvjARVI0z7PI4MysCTgXuNzdO9z9fuB24IIizT8C/Nzdb3D3Hnc/4O5PRBFHbpVBjTxLnEWSPI+wU+eeswh4J/DFKGIYKy+a0cKUxhp++/SucociMqHkqjQShso2RIZ3FJBx93V5+9YAxQapXgHsNrPfmtl2M7vDzOYXO6iZXWxmq81s9Y4dO4YNIlf3rJFnibOoRp5H0qlzvgF8AqjoeohEwnjlkmn87umdqnsWiVB+2YZm2xAZVjOwr2DfPqClSNu5wIXAh4D5wDPAjcUO6u7XuPtKd1/Z3t4+bBC5EWeNPEucRZU8j6RTY2ZvBlLufstwBx7pWfFYeOWSNl7Y182zu7rK8voiE1H/IimJ3CIpZQ5IpLJ1AK0F+1qBYgsRHARucfeH3L0b+AxwkplNGm0QyWQ48qzZNiTGSkqezexeM/NBtvsZQacOSzy+AnyglNce6VnxWDh5yTQAfvv0zrK8vshE1D/bhpmmqhMZ3jogZWbL8vatANYWaftnIL9D5W6POuNVzbNIicmzu5/u7jbIdgoj69TLgIXAfWa2FbgZmGVmW81s4Wh+mbGyqK2Jma31/PYp1T2LRCV/qrpc/WRWCbRIUe7eSfB5eYWZNZnZycDZwPVFmn8XeLOZHWdmNcDlwP3uvne0ceSS5qSSZ4mxSMo2RtipHwPmAceF23uAbeHtTVHEEzUz46Sl0/jdhl36cBeJSCZcYTAZjjwDmnFDZGiXAg0Es1TdCFzi7mvN7FQz68g1cvdfEVxTdGfYdikw7PSxpUgqeRYhFeGxLgWuJeiouwg7NYCZnQr8zN2b3T0NbM09ycx2A1l331rkmBXjlYuncfMfn2f99g5eNLNoKbeIjMChso1DV+5nsk5Ncmxe72Bvhv9+fCu3/ul5Ht+yn3TGeesJc3nHy+ezqK1pbF5UJELuvhs4p8j++wiuPcrfdxVwVdQxpJJKnkUiS54H69ThY4d17LzH7iW4MriivWJxUPf8+w27lDyLRCBX4xwsvBB8EI/FwPP6bQf40s+e5L6ndtKbzjJ3SgOnLG2no6eP/7z/Ga7+zQZWzJvMG46dyWuOmcHitiZMF0OJFNU/VZ36iMRYlCPPE9q8qY3MmdzA7zfs4sKTFpY7HJGqlxt5TpqRDAvIoi7buOEPG/nM7Y/TWJfk/BPn89pjZvCKRdP6R7q37e/mtkee5/Y1L/DFnz3JF3/2JHOnNHDSkmmcuGgaxy+YwsJpjUqmRUL9U9Ul1SckvpQ8j8ArFk/jnr9sJ5t1TRAvMkr9i6QkDo1iRTnjxs/XbuVTtz7Gacva+fe3raC9pe6wNjNa67n4tCVcfNoSNu/p4p6/7OC+dTv4+dpt/Hj1ZgAmNdTwkjmTOHpWC0fNCLbF7U201NdEFqtItUiGs21o5FniTMnzCLxyyTR++sfNqnsWiUAmb5GUZMSzbdy/ficf/tEjrJg7masvOIH6Egqp505p5IJXLOCCVywgm3We2tHBwxv3sGbTXta+sJ/v/W4jvelsf/u25joWTGtk3pQG5kxpYM7kRmZOqmNGaz3TW+qZ2lSrulCZcGqSWiRFRMnzCLxi8VQAHnhqp5JnkVHqL9vIq3mOYpXBmx56jk/c8hhL25v59rtWlpQ4F0okrH+U+W9fHqxqnM5k2bi7i6e2d/D0jg427uxi4+5OHnp2D3f8ectho+YJg6lNtUxrqmNyYw1TGmuZ3FjDpIYaWhtqaKlP0VKformuhqa6JE21KZrqkjTUpmisSdJQm6QulVDJiFQUzbYhouR5ROZOaWTZ9GbuXruVvztlUbnDEalqhctzw+hrnq974BlW3fE4px3VzpXn/VWkpRWpZIIl7c0saT/82udM1tm2v5ut+7vZvr+bbft72NnRw86OXnZ39rCns48NOzvY09XHvoN9A0awh1OXSlBfk6S+JkFdKkltKkFtMjHgZ03SqEkmqEkmSCWNVCLYl7udShipZPAzkTBSiWC0Pxnezo3+JxLWX4Oe25dMGBZOJ5ibGSVpRiIRTOOZyHvMcsex3GPhv68ZZoeOmf94fjvIXUDKgOck8toe2jewjTHwmDI2NM+ziJLnEXvDS2bxf361nu0HupneUl/ucESqVqZghUGAbOk55WGu/92zrLrjcV57zAyuPO94alORTGNfkmTCmD25gdmTG0pq392XYX93H509GTq60xzo6eNgb4bO3gwHe9Mc7M3Q1ZehuzdDdzpLd1+Gnr4svZksPekMveksPeksveksXb1pejNZ0hmnL5MlnXX60sHPdDbYl8k66YyTzmZjswx6sQTdKEi6E0HSHdwP9uXuF0vWD7tP+Jzw/jXvWsmcEv8GqpVGnkWUPI/YG186i6//cj0/f2wrF7xyYbnDEalauUQ5GN0Mbh/JyHM6k+Wm1Zu4/La1vObo6Xzz/OOpSY5f4nwkgpHkJJSh+svdyWSdTPgznXWy2UP7slnCn4f2uTtZD0bYM1nHPWwTPpbJhscNn+8cahfcZkDbrDtO8O2DEzwnty/3WrnXcA9fL+v9ZT3ZvDbZMNbguYTHCx8n+JkNHjjseR6+ZiZsD2HMeb/fgOeR97yC47l7LOqAc8tzK3mWOFPyPEJHzWhh6fRm7nx0i5JnkVHIXyTF+keeS0+e12zayy1/ep6fr93Kln3dvHLxNP7veZWfOJebWVjOUe5ApCr1jzyrNEZiTP9/HoE3hqUbz+3qYv60xnKHI1KV+ss2EnllGyWOPG/Zd5C3fet3JBJw8pI2/u2sYzjj6BlKnEXGmGqeRUCfNEfgvBPnk0oY1z7wTLlDEalauVHmZN5UdaXO83z1rzeQdecX//gqvnPRyzjz2FlKnEXGgWqeRZQ8H5EZrfWc9dLZ/Hj1JvZ19ZU7HJGq1L9ISt5sG6WMPO840MONDz7HW46fw7yp+uZHZDzlTlKVPEucKXk+Qu85dTFdvRlueHBjuUMRqUr9i6QkgrrnYN/Qz3F3vnjXE/Rlslxy+tIxjlBECmnkWUTJ8xE7ZnYrpy5r47sPPEtPOlPucESqTv4iKaXWPH/vt89y85+e50NnHMWitqYxj1FEBkrpgkERXTA4Gu87bQnv/M4fuPVPz/P2l80vdzgSsdyUXrm5coN5coO5cnNz6mayTl84f25fxknn9mWdTP++Q89JZ8P5d/Pm5E1nDu3Lv9/eUsf7/3rijq4emm3DaKwL/iv68I8e4ZNvPJrTjmo/rP1T2zv43J1P8Jqjp/OBV0/c90WkkvWPPCeVPEt8KXkehZOXTmP57Fau+c0G3nbCvP66TTnE3enNBIs59GU8/Bks9tDXvz9LbzpIJoPt0O10xsMFIIKEsjeTpS89MFnN7U/nFooIE9C+zMBEN11sfzZLJnxOXyZ3+1CiPJ4SRv8qcKmEcfSs1gmdPPeXbZhx6tI2vvq2FVx5z1O89/ur+eklJ3HsnEkD2n/+zsdpqEnypXNfqr4mUiappEaeRZQ8j4KZcenpS3n/D//Ijx7axHknVs/ocybrdPam6ehO09mTprM3Q1fuZ2+art4MXbnVzvoydPdlg5+9GbrTwWpnPbmVz8KVznrSwe3c/d50kCSPBTOoSeSWIh64NHFN8tByxLlli5MJo74mQaouFSxbnChsW2zfoccKXysZ/sztK3a8ZG6J5NxjiQTJ/vaH4sstnxy3hDBXoZFbFvrcE+byqhe1c9Y37ud91z/Mp954NCctaaO1IcVtj7zAPX/ZwSfe8GLamuvKG7hIjKnmWUTJ86i94SUzOXHRVL5895O8bvkMpo3zB3s6k2V3Vy+7OnrZ3Rlse7p62dvVx56uXvYd7GNfVx/7Dvaxv7uPA91pDnSn6ehJl/waCYOGmiQNtUnqUknqaxLU1ySpSyWoSyVpqU9Rl0pSV5OgNpmgNtxfm8rdDvbXJI3aVDL8mduXoCZ1KMmtSR5qm5/81uTdziWeUt0OjTwf2tfWXMdV7zyBd3/3QS654Y+YwfSWOrbt7+FFM1q48KSF5QlWRACtMCgCSp5Hzcz43DnH8vqv38fbr/k97zxxPq85ZgZzpxz5FFruzoGeNNv397D9QDc7DvT0395+oIcd4bazo4c9Q0yV11yXYlJDTf+2qK2J1voaWupraKlP0VKfoqku2JrrkjTVpmisTdFQm6SpLkljTYr62iCZNX1FJxHL5NU85ztu3mQe/ORreGTTXn739C6e2LKf045q5y3Hz6EulSxHqCIS0gWDIkqeI7FsRgv/97zjufKep1h1x+OsuuNx5kxu4OhZrcyf2sj01jpa6lP9SWg6E5RAdPVm2H8wGCHe3dnHrs4gId55oJeDfYfP4FFfk2B6Sz3tLXUsbm/ixMVTmdZUR1tzLdOa65jSWMvUpmCb3FijRSOkonnebBuFapIJXrZwKi9bOHW8wxKRISSThhmxKzMTyafkOSJnHjuTM4+dyVPbO/j1uh386bk9/GXrAR54amfRRDinNpVgSmMNUxpraWuuY8H8Rtqa65jeWsf0lnqmtwS321vqaa1PaQRYJoxcOXzhyLOIFGdmU4HvAH8D7AT+1d1/WKTdKuCTQE/e7pe6+4bRxjBncgOzJzWM9jAiVU3Jc8SWTm9m6fRmYBEQjK519Wbo6EnTm87iHlytnKshrq/R19AST2etmMXJS6fR1lxb7lBEqsWVQC8wAzgOuNPM1rj72iJtb3L3d0YdwDtPXMDbXzYv6sOKVJXIvtc3s6lmdouZdZrZRjM7b4i2q8ysz8w68rbFUcVSScyMproUM1rrmTe1kfnTGpk9uYEpTbVKnKWsSu2zFviyme0Kt69YBF+BtNTXsGBaky7+FCmBmTUB5wKXu3uHu98P3A5cMJ5xJBKmaw8k9qL81Mo/Iz4fuMrMlg/R/iZ3b87bRv11koiMSKl99mLgHGAF8FLgTcD7xitIEQHgKCDj7uvy9q0BBvucPcvMdpvZWjO7ZLCDmtnFZrbazFbv2LEjynhFJqxIkudKOSMWkdKMsM9eCHzV3Te7+/PAV4GLxi1YEQFoBvYV7NsHtBRp+2PgaKAdeC/waTP722IHdfdr3H2lu69sbz98ZU8ROVxUI88jPSMGnRWLlNNI+uzy8LHh2qm/ioydDqC1YF8rcKCwobs/7u4vuHvG3X8LfB146zjEKBILUSXPIzkjBp0Vi5TbSPpsYdt9QHOxumf1V5Exsw5ImdmyvH0rgGIXCxZyQNPaiESkpOTZzO41Mx9ku58RnBGDzopFKsBI+mxh21agw3MTNYvImHP3TuBm4AozazKzk4GzgesL25rZ2WY2JbzY9+XAB4HbxjdikYmrpKnq3P30oR4P6ydTZrbM3deHu0s9I4YSz4offvjhnWa2cZhmbQTzX0rp9J6NXKW8ZwuO8Hn9o1gl9Nm14WMPDtNuAPXXMaX3beQq4T070v6acylwLbAd2AVc4u5rzexU4Gfu3hy2e0fYrg7YDHzZ3b833MHVZ8eM3rORq5T3rGiftagGj8zsRwRJ8HsI5p+8Czip2PyTZnY28BtgL/Ay4BbgE6V07hLiWO3uK0d7nDjRezZyE+E9K7XPmtk/AB8CXhO2/wXwDXf/VgQxVP37WA5630ZO71k09D6OnN6zkav09yzKqeouBRoIzohvJDwjBjCzU82sI6/tO4CnCL4i/j4lnhWLSKSK9tki/fVq4A7gUeAx4M5wn4iISOxEtsKgu+8mmAu22GP3EVx0lLtf9OJAERk/g/XZIv3VgY+Hm4iISKxNxKW9ril3AFVI79nI6T2Lht7HI6P3beT0nkVD7+PI6T0buYp+zyKreRYRERERmegm4siziIiIiMiYUPIsIiIiIlKiCZM8m9lUM7vFzDrNbKOZnVfumCpRuOBNt5l1hNtf8h47L3zvOs3sVjObWs5Yy8HMLguXl+4xs+sKHjvDzJ40sy4zu8fMFuQ9Vmdm15rZfjPbamYfGffgq4j6a2nUX4enPjs+1GeHp/46vInSXydM8gxcCfQCM4DzgavMbHl5Q6pYl7l7c7i9CCB8r64GLiB4D7uAb5YxxnJ5AfgcwQID/cysjWB1r8uBqcBq4Ka8JquAZQQTqv818HEzO3Mc4q1W6q+lU38dmvrs+FCfLY3669AmRH+dEMmzBSscngtc7u4d7n4/cDvBH6qU5nzgDnf/jbt3EPwBv8XMWsoc17hy95vd/VaC1bvyvQVY6+7/5e7dBB15hZm9OHz8XcBn3X2Puz8BfBu4aJzCrirqr5FQfw2pz4499dlRU38NTZT+OiGSZ+AoIOPu6/L2rQF0VlzcF81sp5k9YGanh/uWE7xnALj70wSjDEeVIb5KVPj+dAJPA8vNbAowO/9x9Pc3FPXXkVF/PTLqs9FRny2d+uuRqar+GtkiKWXWDOwr2LcPiN1ZXQn+GXicoOO+A7jDzI5D7+FwmoEdBfty709z3v3Cx+Rw+lsrnfrrkVOfjY7+3kqj/nrkqqq/TpSR5w6gtWBfK8Hy35LH3f/g7gfcvSdcEv0B4A3oPRzOUO9PR979wsfkcPpbK5H666ioz0ZHf28lUH8dlarqrxMleV4HpMxsWd6+FcDaMsVTTRwwgvdqRW6nmS0G6gjeWzn8/WkClhDUaO0BtuQ/jv7+hqL+euTUX0unPhsd9dkjo/5auqrqrxMieQ5rY24GrjCzJjM7GTgbuL68kVUWM5tsZq8zs3ozS5nZ+cBpwM+BG4CzzOzU8I/2CuBmd4/VmXH4vtQDSSCZe6+AW4Bjzezc8PFPA3929yfDp34f+JSZTQkvcHgvcF0ZfoWKp/5aGvXX0qjPjj312eGpv5ZmwvRXd58QG8HUJrcCncBzwHnljqnSNqAdeIjgq469wO+B1+Y9fl743nUCtwFTyx1zGd6jVQSjBfnbqvCx1wBPAgeBe4GFec+rI5h6Zz+wDfhIuX+XSt7UX0t6j9RfS3uf1GfH531Wnx36/VF/Le19mhD91cKgRERERERkGBOibENEREREZDwoeRYRERERKZGSZxERERGREil5FhEREREpkZJnEREREZESKXkWERERESmRkmcRERERkRIpeRYRERERKZGSZxERERGREv3/iu+ht1/5Ym4AAAAASUVORK5CYII=)

Again, we've got most of our activations near zero. Let's see what else we can do to improve training stability.

同样，我们取得了大多数零附近的激活。让我们看一下我们还能做什么其它的事情来改善训练的稳定性。

### 1cycle Training

### 一循环训练

Our initial weights are not well suited to the task we're trying to solve. Therefore, it is dangerous to begin training with a high learning rate: we may very well make the training diverge instantly, as we've seen. We probably don't want to end training with a high learning rate either, so that we don't skip over a minimum. But we want to train at a high learning rate for the rest of the training period, because we'll be able to train more quickly that way. Therefore, we should change the learning rate during training, from low, to high, and then back to low again.

我们的初始权重不会很好的适合我们尝试解决的任务。因此，用一个高的学习率开始训练这是危险的：如我们之前学过的，我们完全可以使得训练立刻偏离。我们大概也不想用一个高的学习率结束训练，所以我们不会略过最小学习率上的训练。但是我们希望对于其余的训练期间在一个高学习率上训练，因为这样我们就能够训练的更快。因此，你们应该在训练期间改变学习率，从低到高，然后再返回到低学习率。

Leslie Smith (yes, the same guy that invented the learning rate finder!) developed this idea in his article ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"](https://arxiv.org/abs/1708.07120). He designed a schedule for learning rate separated into two phases: one where the learning rate grows from the minimum value to the maximum value (*warmup*), and one where it decreases back to the minimum value (*annealing*). Smith called this combination of approaches *1cycle training*.

莱斯利·史密斯（Leslie Smith）（是的，发明学习率查找器的家伙！）在他的论文[“超收敛：使用大学习率非常快速的训练神经网络”](https://arxiv.org/abs/1708.07120)中介绍了这一想法。他对学习率设计了一个计划，分割为两个阶段：一个阶段学习率从最小值到最大值增长（预热），另一个阶段是减小返回到最小值（退火）。史密斯称这个组合方法为*一循环训练*（1cycle training）。

1cycle training allows us to use a much higher maximum learning rate than other types of training, which gives two benefits:

- By training with higher learning rates, we train faster—a phenomenon Smith named *super-convergence*.
- By training with higher learning rates, we overfit less because we skip over the sharp local minima to end up in a smoother (and therefore more generalizable) part of the loss.

相比其它类型的训练，一循环训练允许我们使用更高的最大学习率，这个方法提供了两个好处：

- 通过使用更高的学习率训练，我们训练的更快。史密斯命名这一现象*超级收敛*。
- 通过使用更高的学习率训练，我们会减少过拟，因此我们略过了突出的局部最小值，最终有了更加平滑（因而能够更好泛华）的损失部分。

The second point is an interesting and subtle one; it is based on the observation that a model that generalizes well is one whose loss would not change very much if you changed the input by a small amount. If a model trains at a large learning rate for quite a while, and can find a good loss when doing so, it must have found an area that also generalizes well, because it is jumping around a lot from batch to batch (that is basically the definition of a high learning rate). The problem is that, as we have discussed, just jumping to a high learning rate is more likely to result in diverging losses, rather than seeing your losses improve. So we don't jump straight to a high learning rate. Instead, we start at a low learning rate, where our losses do not diverge, and we allow the optimizer to gradually find smoother and smoother areas of our parameters by gradually going to higher and higher learning rates.

第二点的好处是有趣及精妙的。基于观察这个模型泛华的很好，即使你改变很小的输入数量，这个模型也不会有很大的改变。如果一个模型在一个大的学习率上训练一段时间，在当样做时能够发现一个好的损失，它一定是也找到了很好泛华的区域，因为它在批次间跳转很多（这是基础的高学习率定义）。问题在于，我们已经讨论过的，只是在一个高学习率上跳转更可能导致损失偏离，而不是我要所要寻找的损失改善。所以我们不会直接跳到一个高的学习率上。相反，我们从一个低学习率开始，在这里我们的损失不会偏离，及我们允许优化器通过越来越高的学习率来逐步寻找我们参数越来越平滑的区域。

Then, once we have found a nice smooth area for our parameters, we want to find the very best part of that area, which means we have to bring our learning rates down again. This is why 1cycle training has a gradual learning rate warmup, and a gradual learning rate cooldown. Many researchers have found that in practice this approach leads to more accurate models and trains more quickly. That is why it is the approach that is used by default for `fine_tune` in fastai.

然后，一旦我们发现参数好的平滑区域，我们希望找到那个区域的最佳部分，这就表示我们不弱让我们的学习率再次下降。这就是为什么1周期学习有学习率逐步预热，及学习率逐步退火。在实践中很多研究人员发现这一方法会使得模型更加精准及训练更加快速。这就是为什么这一方法在fastai中对于`fine_tune`是默认使用的。

In <chapter_accel_sgd> we'll learn all about *momentum* in SGD. Briefly, momentum is a technique where the optimizer takes a step not only in the direction of the gradients, but also that continues in the direction of previous steps. Leslie Smith introduced the idea of *cyclical momentums* in ["A Disciplined Approach to Neural Network Hyper-Parameters: Part 1"](https://arxiv.org/pdf/1803.09820.pdf). It suggests that the momentum varies in the opposite direction of the learning rate: when we are at high learning rates, we use less momentum, and we use more again in the annealing phase.

在<章节：加速随机梯度下降>中我们会学习SGD中所有*动量*内容。简短来说，动量是一项技术，优化器前进的一步不仅仅是在梯度的方向上，而且也是继续之前步骤的方向。莱斯利·史密斯（Leslie Smith）在["对神经网络超参的一个自律方法"](https://arxiv.org/pdf/1803.09820.pdf)中引入了*周期动量*思想。这个思想认为动量在学习率的相反方向变化：当我们是高学习率时，我们使用更少的动量，我们在退火阶段再次使用更多动量。

We can use 1cycle training in fastai by calling `fit_one_cycle`:

在fastai中通过调用`fit_one_cycle`我们能够使用一周期训练：

```
def fit(epochs=1, lr=0.06):
    learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy,
                    metrics=accuracy, cbs=ActivationStats(with_hist=True))
    learn.fit_one_cycle(epochs, lr)
    return learn
```

```
learn = fit()
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   0.210838 |   0.084827 | 0.974300 | 00:08 |

We're finally making some progress! It's giving us a reasonable accuracy now.

我最后取得了很大进步！现在它提供给我们一个合理的精度。

We can view the learning rate and momentum throughout training by calling `plot_sched` on `learn.recorder`. `learn.recorder` (as the name suggests) records everything that happens during training, including losses, metrics, and hyperparameters such as learning rate and momentum:

能够调用`learn.recorder`上的`plot_sched`我们能够查看整个训练过程的学习率和动量。`learn.recorder`记录了训练期间所发生的所有事项，包括损失、指标和超参，例如学习率和动量：

```
learn.recorder.plot_sched()
```

Out:![img](data:image/png;base64,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)

Smith's original 1cycle paper used a linear warmup and linear annealing. As you can see, we adapted the approach in fastai by combining it with another popular approach: cosine annealing. `fit_one_cycle` provides the following parameters you can adjust:

- `lr_max`:: The highest learning rate that will be used (this can also be a list of learning rates for each layer group, or a Python `slice` object containing the first and last layer group learning rates)
- `div`:: How much to divide `lr_max` by to get the starting learning rate
- `div_final`:: How much to divide `lr_max` by to get the ending learning rate
- `pct_start`:: What percentage of the batches to use for the warmup
- `moms`:: A tuple `(mom1,mom2,mom3)` where *`mom1`* is the initial momentum, *`mom2`* is the minimum momentum, and *`mom3`* is the final momentum

史密斯的原始一周期论文使用了线性预热和线性退火。正如你所看到的，我们在fastai中采纳了这一方法，把它与另一个流行方法“余弦退火”做了组合。`fit_one_cycle`提供了下述参数供我们调整：

- `lr_max`：会被使用的最高学习率（也可以是对每个层组的学习率列表，或一个包含第一个或最后一个层组学习率的Python`slice`对象）
- `div`：除以`lr_max`多少来获取开始的学习率
- `div_final`：除以`lr_max`多谢来获取最终的学习率
- `pct_start`：多少批次的百分比用于预热
- `moms`：一个元组`(mom1,mom2,mom3)`中，*`mon1`*是初始的动量，*`mom2`*是最小动量，*`mom3`*是最终动量

Let's take a look at our layer stats again:

让我们再看一下我们层的状态：

```
learn.activation_stats.plot_layer_stats(-2)
```

Out:![img](data:image/png;base64,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)

The percentage of near-zero weights is getting much better, although it's still quite high.

We can see even more about what's going on in our training using `color_dim`, passing it a layer index:

近零权重的百分比变的好多了，虽然它还是十分的高。

传递给`color_dim`层的索引值，我们甚至能够看到更多关于在我们的训练中发生了什么：

```
learn.activation_stats.color_dim(-2)
```

Out:![img](data:image/png;base64,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pFAndaVC/bVjJ71GlaesGHBtGxJxPe3DR9vzQaVFREREniFOeERERGR6nPCIiIjI9HwUh+f//Lt/evL523/8rpX513/4Z23bv3jz39q2n67+8uTzA8R9rcBbuT90P2dTPJiXyx4yuFlQOBOs4Axhgd/t+wrtlQM4FxRQWKHgwU3osrTjgbu0PvS63kO7DuAb1baRw5O4IGOM8VAcJ/otl/qQrlEtlzgk545JXPtbelLXGsbhYdnPu/bXGODswH9j6HrU++NcudrXT3FZHor3Rg4anSMFYCZtSP2Na30mIlqJOxybieNGPCUMtPIU36het4/hYTxWF5H2Re1req5dC43za1lEK7HnVE9zE65m/lF9SKiqXt+PcTy/4REREZHpccIjIiIi0+OER0RERKbHCY+IiIhMz6PS8n/6n100JpbjP5fPsGourPS9PXY56leH09C/P9/1lcs3iy70ElVufrnqgYJvll2wvl1u27YHkJa/Xrw5+UwhfSQ7k9xXRTGSll+E22r9JKF9B/1KkiYJpDelvrtV7684DC0QuFNpMgnlS1djJ6rEmO5H7a911T4d48z5wNDf1VW9QTSn86YxhvJuaWsqYGLg4pXibHJtX5DIHoQ+jsHBj10sT1ehh2DRYyLn03Oz13XtSuLXsgpd0aRdqax9reT7lPNerS5f32tl8FSoT8o95aUBYj2uk7OvFb+prfQsTaTxDz1vv+ERERGR6XHCIyIiItPjhEdERESmxwmPiIiITM/iePy4qY0iIiIinxp+wyMiIiLT44RHREREpscJj4iIiEyPEx4RERGZHic8IiIiMj1OeERERGR6/j8jStwSQ8E4SwAAAABJRU5ErkJggg==)

`color_dim` was developed by fast.ai in conjunction with a student, Stefano Giomo. Stefano, who refers to the idea as the *colorful dimension*, provides an [in-depth explanation](https://forums.fast.ai/t/the-colorful-dimension/42908) of the history and details behind the method. The basic idea is to create a histogram of the activations of a layer, which we would hope would follow a smooth pattern such as the normal distribution (colorful_dist).

`color_dim`是由fast.ai与学生斯特凡诺·乔莫（Stefano Giomo）合作开发的。斯特凡诺提供了这一方法[深入研究的历史](https://forums.fast.ai/t/the-colorful-dimension/42908)和背后的细节。基本想法是创建一个层的激活直方图，我们希望它遵循一个平滑模式，如正态分布（色彩分布）。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/colorful_dist.jpeg" id="colorful_dist" caption="Histogram in 'colorful dimension'" alt="Histogram in 'colorful dimension'" width="800">
  </p>
  <p align="center">图：色彩度量直方图</p>
</div>

To create `color_dim`, we take the histogram shown on the left here, and convert it into just the colored representation shown at the bottom. Then we flip it on its side, as shown on the right. We found that the distribution is clearer if we take the log of the histogram values. Then, Stefano describes:

> : The final plot for each layer is made by stacking the histogram of the activations from each batch along the horizontal axis. So each vertical slice in the visualisation represents the histogram of activations for a single batch. The color intensity corresponds to the height of the histogram, in other words the number of activations in each histogram bin.

<colorful_summ> shows how this all fits together.

在上图左侧我们展示了直方图，来创建`color_dim`，并在下方转化它为颜色表示。然后我们翻转它的侧面，如右侧所示。如果取了日志的直方图数值，我们发现分布更清晰了。于是，斯特凡诺描述到：

> ：最终每层绘图是通过每个批次沿着水平轴堆砌激活的直方图。所以每个可视化中的垂直切片表示一个单批次激活的直方图。色彩强度的对应直方图的高度，换句话说在每个直方桶中的激活数量。

<色彩度量总结>展示了这些内容如何组合在一起。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/colorful_summ.png" id="colorful_summ" caption="Summary of the colorful dimension (courtesy of Stefano Giomo)" alt="Summary of the colorful dimension" width="800">
  </p>
  <p align="center">图：色彩度量总结</p>
</div>

This illustrates why log(f) is more colorful than *f\* when \*f* follows a normal distribution because taking a log changes the Gaussian in a quadratic, which isn't as narrow.

So with that in mind, let's take another look at the result for the penultimate layer:

这个个图解中，当 *f* 符合一个正态分布时，为什么log(f)比 *f* 更加色彩丰富呢，因为求对数改变了二次方中的高斯函数，它不再那么狭窄了。

```
learn.activation_stats.color_dim(-2)
```

Out:![img](data:image/png;base64,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pFAndaVC/bVjJ71GlaesGHBtGxJxPe3DR9vzQaVFREREniFOeERERGR6nPCIiIjI9HwUh+f//Lt/evL523/8rpX513/4Z23bv3jz39q2n67+8uTzA8R9rcBbuT90P2dTPJiXyx4yuFlQOBOs4Axhgd/t+wrtlQM4FxRQWKHgwU3osrTjgbu0PvS63kO7DuAb1baRw5O4IGOM8VAcJ/otl/qQrlEtlzgk545JXPtbelLXGsbhYdnPu/bXGODswH9j6HrU++NcudrXT3FZHor3Rg4anSMFYCZtSP2Na30mIlqJOxybieNGPCUMtPIU36het4/hYTxWF5H2Re1req5dC43za1lEK7HnVE9zE65m/lF9SKiqXt+PcTy/4REREZHpccIjIiIi0+OER0RERKbHCY+IiIhMz6PS8n/6n100JpbjP5fPsGourPS9PXY56leH09C/P9/1lcs3iy70ElVufrnqgYJvll2wvl1u27YHkJa/Xrw5+UwhfSQ7k9xXRTGSll+E22r9JKF9B/1KkiYJpDelvrtV7684DC0QuFNpMgnlS1djJ6rEmO5H7a911T4d48z5wNDf1VW9QTSn86YxhvJuaWsqYGLg4pXibHJtX5DIHoQ+jsHBj10sT1ehh2DRYyLn03Oz13XtSuLXsgpd0aRdqax9reT7lPNerS5f32tl8FSoT8o95aUBYj2uk7OvFb+prfQsTaTxDz1vv+ERERGR6XHCIyIiItPjhEdERESmxwmPiIiITM/iePy4qY0iIiIinxp+wyMiIiLT44RHREREpscJj4iIiEyPEx4RERGZHic8IiIiMj1OeERERGR6/j8jStwSQ8E4SwAAAABJRU5ErkJggg==)

This shows a classic picture of "bad training." We start with nearly all activations at zero—that's what we see at the far left, with all the dark blue. The bright yellow at the bottom represents the near-zero activations. Then, over the first few batches we see the number of nonzero activations exponentially increasing. But it goes too far, and collapses! We see the dark blue return, and the bottom becomes bright yellow again. It almost looks like training restarts from scratch. Then we see the activations increase again, and collapse again. After repeating this a few times, eventually we see a spread of activations throughout the range.

这展示了一个典型的“糟糕训练”结果图。我们从几乎所有激活为零开始。即，我们在最左侧看到的全是深蓝色的内容。在底部的亮黄色代表零附近的激活。然后，在开始的几个批次我们看到近零激活数量呈几何倍数增长。但增长的太过了，并导致坍塌了！我们看到又回到了深蓝色和底部变为亮黄色。它看起来几乎从零重新开始训练。然后我们看到激活再次增长并再次坍塌。重复几次后，最终我们看到激活在整个范围展开。

It's much better if training can be smooth from the start. The cycles of exponential increase and then collapse tend to result in a lot of near-zero activations, resulting in slow training and poor final results. One way to solve this problem is to use batch normalization.

如果训练能够从一开始是平滑的会更好。几何增长且随后坍塌几个周期倾向产生很多近零激活，导致缓慢的训练和差的最终结果。解决这一问题的方法是使用批次标准化。