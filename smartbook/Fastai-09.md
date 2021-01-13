# Tabular Modeling Deep Dive

# 表格模型深潜

Tabular modeling takes data in the form of a table (like a spreadsheet or CSV). The objective is to predict the value in one column based on the values in the other columns. In this chapter we will not only look at deep learning but also more general machine learning techniques like random forests, as they can give better results depending on your problem.

表格模型接受表格形式的数据（像电子表格或CSV）。目标是基于其它列的值预测某一列的值。在本章节我们不仅会看深度学习，而且看更多常用的如随机森林这样的机器学习技术，基于我们的问题它们能够产生更好的结果。

We will look at how we should preprocess and clean the data as well as how to interpret the result of our models after training, but first, we will see how we can feed columns that contain categories into a model that expects numbers by using embeddings.

我们会看到我们应该如何处理和清理术语，和如何解释训练后我们模型的结果，但首先，我们会看通过嵌入我们如何能够把包含分类的那些列送入期望数字的模型中。

## Categorical Embeddings

## 分类嵌入

In tabular data some columns may contain numerical data, like "age," while others contain string values, like "sex." The numerical data can be directly fed to the model (with some optional preprocessing), but the other columns need to be converted to numbers. Since the values in those correspond to different categories, we often call this type of variables *categorical variables*. The first type are called *continuous variables*.

在表格数据中，一些列可能包含数值数据，如“年龄”，与此同时其它列包含字符串值，如“性别”。数值数据能够直接喂给模型（和可远的预处理），但是其它列需要转换为数值。因为那些值对应不同的分类，我们经常称这些变量类型为*分类变量*。第一种类型称为*连续变量*。

> jargon: Continuous and Categorical Variables: Continuous variables are numerical data, such as "age," that can be directly fed to the model, since you can add and multiply them directly. Categorical variables contain a number of discrete levels, such as "movie ID," for which addition and multiplication don't have meaning (even if they're stored as numbers).

> 术语：连续和分类变量：联系变量是数值数据，如“年龄”，能够直接喂给模型，因为你能够直接加和乘他们。分类变量包含包含一些分离并列的数据，如“电影ID”，对其加和乘法运算没有意思（即使它被以数值型存储）。

At the end of 2015, the [Rossmann sales competition](https://www.kaggle.com/c/rossmann-store-sales) ran on Kaggle. Competitors were given a wide range of information about various stores in Germany, and were tasked with trying to predict sales on a number of days. The goal was to help the company to manage stock properly and be able to satisfy demand without holding unnecessary inventory. The official training set provided a lot of information about the stores. It was also permitted for competitors to use additional data, as long as that data was made public and available to all participants.

在2015年末，Kaggle举办了[罗斯曼销售竞赛](https://www.kaggle.com/c/rossmann-store-sales) 。参赛者被提供了关于在德国的各种商店范围广泛的信息，并在许多天数上尝试预测销量的任务。目标是帮助公司管理库存资产并能够不持有不必要的库存清单来满足需求。官方的训练集提供了许多商店的信息。也允许参赛者使用附加数据，只要数据是公共产生的和所有参与者都可获得的。

One of the gold medalists used deep learning, in one of the earliest known examples of a state-of-the-art deep learning tabular model. Their method involved far less feature engineering, based on domain knowledge, than those of the other gold medalists. The paper, ["Entity Embeddings of Categorical Variables"](https://arxiv.org/abs/1604.06737) describes their approach. In an online-only chapter on the [book's website](https://book.fast.ai/) we show how to replicate it from scratch and attain the same accuracy shown in the paper. In the abstract of the paper the authors (Cheng Guo and Felix Berkhahn) say:

一名金奖得主使用了深度学习，是一个最早被熟知的先进深度学习表格模型的例子。相比那些其它金奖得主他们方法依据领域知识涉及更少的特征工程。论文[“分类变量的实体嵌入”](https://arxiv.org/abs/1604.06737)描述了他们的方法。只有在[本书网站](https://book.fast.ai/) 上的在线章节我们描述了如何从零再现它并获取在文章中相同的精度。在论文的摘要中作者（郭程和费利克斯·伯哈恩）说到：

> : Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables... [It] is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit... As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.

> ：相比独热编码，实体嵌入不仅仅减少了内存使用和加速了神经网络，而且更重要的通过映射在嵌入空间中彼此接近的相似值，它揭示了分类变量的本质属性...【它】对有很多高基数特征的数据集尤为有用，其它方法会导致过拟...实体嵌入定义了分类变量的一个距离测量，它能够被用于可视化分类数据和数据群集。

We have already noticed all of these points when we built our collaborative filtering model. We can clearly see that these insights go far beyond just collaborative filtering, however.

当我们构建协同过滤模型时，我们已经注意到了这些所有点。然而，我们能够清晰的看到这些深刻见解远不只是协同过滤。

The paper also points out that (as we discussed in the last chapter) an embedding layer is exactly equivalent to placing an ordinary linear layer after every one-hot-encoded input layer. The authors used the diagram in <entity_emb> to show this equivalence. Note that "dense layer" is a term with the same meaning as "linear layer," and the one-hot encoding layers represent inputs.

这篇论文也指出（作为在本章最后我们讨论），一个嵌入层是完全等价于放置在每个独热编码输入层后的普通线性层。作者使用了<在神经网络中实体嵌入>示意图来展示了这个等价。注意“全连接层”是一个与“线性层”相同含义的术语，独热编码层代表输入。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00018.png" alt="Entity embeddings in a neural network" width="600" caption="Entity embeddings in a neural network (courtesy of Cheng Guo and Felix Berkhahn)" id="entity_emb"  />
  </p>
  <p align="center">图：在神经网络中实体嵌入</p>
</div>

The insight is important because we already know how to train linear layers, so this shows that from the point of view of the architecture and our training algorithm the embedding layer is just another layer. We also saw this in practice in the last chapter, when we built a collaborative filtering neural network that looks exactly like this diagram.

这一见解是很重要的，因为我们已经知道如何来训练线性层，所以这展示了来自训练算法和架构的观点，嵌入层只是另外的层。在最后的章节实践中我们也会看，当我们创建一个协同过滤神经网络看起来完全像这个示意图。

Where we analyzed the embedding weights for movie reviews, the authors of the entity embeddings paper analyzed the embedding weights for their sales prediction model. What they found was quite amazing, and illustrates their second key insight. This is that the embedding transforms the categorical variables into inputs that are both continuous and meaningful.

在我们分析了对于电影评价的嵌入权重的地方，实体嵌入论文作者分析了它们的销售预测模型的嵌入权重。他们的发现是十分令人振奋的，并插图说明了他们的第二个关键见解。这就是嵌入转换分类变量为连续和有意义的输入。

The images in <state_emb> illustrate these ideas. They are based on the approaches used in the paper, along with some analysis we have added.

在<州嵌入和地图>图像中说明了这些想法。他们是基于在论文中使用的方法，连同一些我们已经添加的分析。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00015.png" alt="State embeddings and map" width="800" caption="State embeddings and map (courtesy of Cheng Guo and Felix Berkhahn)" id="state_emb"/>
  </p>
  <p align="center">图：州嵌入和地图</p>
</div>

On the left is a plot of the embedding matrix for the possible values of the `State` category. For a categorical variable we call the possible values of the variable its "levels" (or "categories" or "classes"), so here one level is "Berlin," another is "Hamburg," etc. On the right is a map of Germany. The actual physical locations of the German states were not part of the provided data, yet the model itself learned where they must be, based only on the behavior of store sales!

左侧是对于`状态`分类可能值的嵌入矩阵图像。对于分类变量，我们称变量的可能值为“级别”（或“种类”或“类别”），所以这里一级别是“柏林”，另一个级别是“汉堡”，等等。在右侧是德国地图。德国州的实际物理位置不是数据提供的部分，只是基于商店销售的行为，模型还自己学习了他们一定在什么地方。

Do you remember how we talked about *distance* between embeddings? The authors of the paper plotted the distance between store embeddings against the actual geographic distance between the stores (see <store_emb>). They found that they matched very closely!

你还记得我们讲过的嵌入之间的距离吗？论文的作者以商店间实际的地理距离为背景绘制了商店嵌入间的距离（看<商店距离>）。我们发现他们的匹配非常接近！

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00016.png" alt="Store distances" width="600" caption="Store distances (courtesy of Cheng Guo and Felix Berkhahn)" id="store_emb"/>
  </p>
  <p align="center">图：商店距离</p>
</div>

We've even tried plotting the embeddings for days of the week and months of the year, and found that days and months that are near each other on the calendar ended up close as embeddings too, as shown in <date_emb>.

我们甚至尝试为周的天数和年的月数绘制了嵌入，且发现在日历上彼此接近的天和月最终也接近嵌入，如图<日期嵌入>所示：

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00017.png" alt="Date embeddings" width="900" caption="Date embeddings" id="date_emb"/>
  </p>
  <p align="center">图：日期嵌入</p>
</div>

What stands out in these two examples is that we provide the model fundamentally categorical data about discrete entities (e.g., German states or days of the week), and then the model learns an embedding for these entities that defines a continuous notion of distance between them. Because the embedding distance was learned based on real patterns in the data, that distance tends to match up with our intuitions.

在这两个例子中突出的内容是我们提供的关于分离实体（例如，德国洲或周的天数）基本的分类数据模型，然后模型对于这些界定了他们间距离的连续概念的实体学习一个嵌入。因为实体距离是基于数据中的真实模式学习的，距离趋向与我们的直觉相匹配。

In addition, it is valuable in its own right that embeddings are continuous, because models are better at understanding continuous variables. This is unsurprising considering models are built of many continuous parameter weights and continuous activation values, which are updated via gradient descent (a learning algorithm for finding the minimums of continuous functions).

另外，嵌入是连续的在它自己的正确性上是很有价值的，因为模型更擅长理解连续变量。考虑到模型是由很多连续参数权重和连续激活值的构建，就不奇怪了，这些值是通过梯度下降更新的（寻找连续函数最小值的学习算法）。

Another benefit is that we can combine our continuous embedding values with truly continuous input data in a straightforward manner: we just concatenate the variables, and feed the concatenation into our first dense layer. In other words, the raw categorical data is transformed by an embedding layer before it interacts with the raw continuous input data. This is how fastai and Guo and Berkhahn handle tabular models containing continuous and categorical variables.

其它好处是，用实际连续输入数据我们能够用一个直接了当的方式组合连续嵌入值：我们只连接变量，并把连接喂给我们第一个全连接层。换句话说，原生分类数据在它与原生连续输入数据交互前，通过一个嵌入层转换了。这就是fastai、郭和伯哈恩如何处理含有连续和分类变量表格模型的。

An example using this concatenation approach is how Google does its recommendations on Google Play, as explained in the paper ["Wide & Deep Learning for Recommender Systems"](https://arxiv.org/abs/1606.07792). <google_recsys> illustrates.

有一个使用这一连接方法，谷歌如何在谷歌播放器上做推荐的例子，在论文[推荐系统的广泛深度学习](https://arxiv.org/abs/1606.07792)中做了解释。下图<谷歌播放器推荐系统>的插图说明。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00019.png" alt="The Google Play recommendation system" width="800" caption="The Google Play recommendation system" id="google_recsys"/>
  </p>
  <p align="center">图：谷歌播放器推荐系统</p>
</div>

Interestingly, the Google team actually combined both approaches we saw in the previous chapter: the dot product (which they call *cross product*) and neural network approaches.

有意思的是，谷歌团队实际组合我们在上一章节看到的两个方法：点积（它们称其为*向量积* 和神经网络方法）

Let's pause for a moment. So far, the solution to all of our modeling problems has been: *train a deep learning model*. And indeed, that is a pretty good rule of thumb for complex unstructured data like images, sounds, natural language text, and so forth. Deep learning also works very well for collaborative filtering. But it is not always the best starting point for analyzing tabular data.

让我们暂停一会。到目前为止，对于我们所有建模问题的解决方案是：*训练一个深度学习模型*。事实上，对于如图像、声音、自然语言文本等等复杂的非结构数据，这是非常好的经验法则。对于协同过滤深度学习也工作的非常好。但对于分析表格数据它一直都不是最佳的起始点。

## Beyond Deep Learning

## 深度学习之外

Most machine learning courses will throw dozens of different algorithms at you, with a brief technical description of the math behind them and maybe a toy example. You're left confused by the enormous range of techniques shown and have little practical understanding of how to apply them.

大多数机器学习教程会抛给你很多困难的算法，在这些算法之后有数学的简短技术描述，有可能是一个小例子。通过庞大范围的技术展示和有一些如何应用它们的实践理解，你被搞糊涂了。

The good news is that modern machine learning can be distilled down to a couple of key techniques that are widely applicable. Recent studies have shown that the vast majority of datasets can be best modeled with just two methods:

1. Ensembles of decision trees (i.e., random forests and gradient boosting machines), mainly for structured data (such as you might find in a database table at most companies)
2. Multilayered neural networks learned with SGD (i.e., shallow and/or deep learning), mainly for unstructured data (such as audio, images, and natural language)

好消息是现代机器学习能够提炼几个被广泛应用的技术。最近的研究显示绝大多数数据集只用两个方法就能够被最好的建模：

1. 决策树集合（即，随机森林和梯度推进机），主要用于结构化数据（如在大多数公司的数据库表中你可以找到）
2. 多层神经网络随机梯度下降学习（即，浅层和/或深度学习），主要用于非结构化数据（如音频、图像和自然语言）

Although deep learning is nearly always clearly superior for unstructured data, these two approaches tend to give quite similar results for many kinds of structured data. But ensembles of decision trees tend to train faster, are often easier to interpret, do not require special GPU hardware for inference at scale, and often require less hyperparameter tuning. They have also been popular for quite a lot longer than deep learning, so there is a more mature ecosystem of tooling and documentation around them.

虽然深度学习对于非结构化数据几乎一直明显更优，对于很多类型的结构化数据这两个方法倾向给出完全相似的结果。但决策树集合训练更快，通常更容易解释，对于大规模推理不需要特定的GPU硬件，且通常需要较少的超参调优。相比机器学习它们已经流行了相当长的时间，所以在它们周边有更成熟的工作和文档生态。

Most importantly, the critical step of interpreting a model of tabular data is significantly easier for decision tree ensembles. There are tools and methods for answering the pertinent questions, like: Which columns in the dataset were the most important for your predictions? How are they related to the dependent variable? How do they interact with each other? And which particular features were most important for some particular observation?

更主要的是，对于决策树集合解释表格数据模型的关键步骤明显更容易。有工作和方法回答相关问题，如：在数据集中的哪一列对于你的预测是最重要的？他们如何关联到因变量？他们彼此如何交互？对于一些特定观察哪些特定特征是最重要的？

Therefore, ensembles of decision trees are our first approach for analyzing a new tabular dataset.

因此，决策树集合是我们分析一个新的表格数据集的第一个方法。

The exception to this guideline is when the dataset meets one of these conditions:

- There are some high-cardinality categorical variables that are very important ("cardinality" refers to the number of discrete levels representing categories, so a high-cardinality categorical variable is something like a zip code, which can take on thousands of possible levels).
- There are some columns that contain data that would be best understood with a neural network, such as plain text data.

当数据集遇到这些条件之一时的这个例外指导方针：

- 有一些高基数分类变量是非常重要的（“基数”指的是代表类别的许多离散级别数目，所以高基数分类变量是如地区码这样的内容，它能够接受数千可能的级别）。
- 有一些列包含了最好用神经网络来理解的数据，如纯文本数据。

In practice, when we deal with datasets that meet these exceptional conditions, we always try both decision tree ensembles and deep learning to see which works best. It is likely that deep learning will be a useful approach in our example of collaborative filtering, as we have at least two high-cardinality categorical variables: the users and the movies. But in practice things tend to be less cut-and-dried, and there will often be a mixture of high- and low-cardinality categorical variables and continuous variables.

在实践中，当我们处理数据集遇到这些例外条件时，我们总会尝试决策树和深度学习两者，来看哪一个工作的更好。可能深度学习在我们协同过滤例子中会是一个有用的方法，我们有至少两个高基数分类变量：用户和电影。但现实中的事情不是一成不变的，经常会有高和低基数分类变量和连续变量的混合体。

Either way, it's clear that we are going to need to add decision tree ensembles to our modeling toolbox!

不管哪种方式它都是很明显的，我们将需要添加决策树集合到我们建模工具箱！

Up to now we've used PyTorch and fastai for pretty much all of our heavy lifting. But these libraries are mainly designed for algorithms that do lots of matrix multiplication and derivatives (that is, stuff like deep learning!). Decision trees don't depend on these operations at all, so PyTorch isn't much use.

截至现在，对于我们几乎所有的困难任务我们已经使用PyTorch和fastai。但是这些库主要的设计为了用于算法，做大量的矩阵乘法和除法（即这些内容就像深度学习！）。决策树完全不依赖这些操作，所以PyTorch用的不太多。

Instead, we will be largely relying on a library called scikit-learn (also known as `sklearn`). Scikit-learn is a popular library for creating machine learning models, using approaches that are not covered by deep learning. In addition, we'll need to do some tabular data processing and querying, so we'll want to use the Pandas library. Finally, we'll also need NumPy, since that's the main numeric programming library that both sklearn and Pandas rely on.

作为替代，我们会极大的依赖名叫scikit-learn库（也被称为`sklearn`）。Scikit-learn是一个创建机器学习模型很流行的库，所使用的方法不是深度学习所覆盖的。另外，我们会需要做一些表格数据处理和查询，所以我们希望使用Pandas库。最后，我们也会需要NumPy，因为它是主要的数字程序库，sklearn和Pandas两者也依赖于它。

We don't have time to do a deep dive into all these libraries in this book, so we'll just be touching on some of the main parts of each. For a far more in depth discussion, we strongly suggest Wes McKinney's [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) (O'Reilly). Wes is the creator of Pandas, so you can be sure that the information is accurate!

在本书，我们没有时间对所有这些库做更入的研究，所以我们仅仅会接触每个库的一些主要部分。为了更深入的探讨，我们强烈推荐韦斯·麦金尼编写的[Python数据分析](http://shop.oreilly.com/product/0636920023784.do)（O'Reilly出版）。韦斯是Pandas的创建者，所以你能够确信书上的信息是准确的！

First, let's gather the data we will use.

首先，让我们收集将要使用的数据。

## The Dataset

## 数据集

The dataset we use in this chapter is from the Blue Book for Bulldozers Kaggle competition, which has the following description: "The goal of the contest is to predict the sale price of a particular piece of heavy equipment at auction based on its usage, equipment type, and configuration. The data is sourced from auction result postings and includes information on usage and equipment configurations."

在本章我们使用的数据集来自Kaggle比赛推土机蓝皮书，其有如下描述：“比赛的目的是基于用途、设备类型和配置来预测在拍卖市场上一种特殊重装备的销售价格。数据来源来自拍卖结果发布，包括使用和设备配置信息。”

This is a very common type of dataset and prediction problem, similar to what you may see in your project or workplace. The dataset is available for download on Kaggle, a website that hosts data science competitions.

这是一个非常普通的数据集类型和预测问题，与你可能在你的项目和工作场所看到的问题类似。这个数据集能够在Kaggle上有效下载，这是一个主办数据科学比赛的网站。

### Kaggle Competitions

### Kaggle 比赛

Kaggle is an awesome resource for aspiring data scientists or anyone looking to improve their machine learning skills. There is nothing like getting hands-on practice and receiving real-time feedback to help you improve your skills.

Kaggle对于渴望寻找改善他们机器学习技能的数据科学家或相关人是一个极佳的资源地。没有什么比获取手动实践和接收时时反馈更能帮助你改善你的技巧的了。

Kaggle provides:

- Interesting datasets
- Feedback on how you're doing
- A leaderboard to see what's good, what's possible, and what's state-of-the-art
- Blog posts by winning contestants sharing useful tips and techniques

Kaggle提供：

- 有趣的数据
- 反馈你做的怎么样
- 一个选手积分榜来看什么是好的，什么是可能的和什么是最先进的
- 微博发布获胜选手共享有用的技巧和技术

Until now all our datasets have been available to download through fastai's integrated dataset system. However, the dataset we will be using in this chapter is only available from Kaggle. Therefore, you will need to register on the site, then go to the [page for the competition](https://www.kaggle.com/c/bluebook-for-bulldozers). On that page click "Rules," then "I Understand and Accept." (Although the competition has finished, and you will not be entering it, you still have to agree to the rules to be allowed to download the data.)

截至目前，我们所有的数据集通过fastai的集成数据集系统有效下载获得。然而，本章使用的数据集只能从Kaggle上获取。因此，你将需要去这个网站注册，然后到[这个比赛的页面](https://www.kaggle.com/c/bluebook-for-bulldozers)。在那个页面上点击“规则”，然后“我理解并接受。”（虽然比赛已经结束，你将不能进入这个比赛，但你仍必须同意规则以被允许下载数据。）

The easiest way to download Kaggle datasets is to use the Kaggle API. You can install this using `pip` by running this in a notebook cell:

一个最容易下载Kaggle数据集的方法是使用Kaggle的API。你能够使用`pip`安装，通过notebook单元格来运行这个命令：

```
!pip install kaggle
```

You need an API key to use the Kaggle API; to get one, click on your profile picture on the Kaggle website, and choose My Account, then click Create New API Token. This will save a file called *kaggle.json* to your PC. You need to copy this key on your GPU server. To do so, open the file you downloaded, copy the contents, and paste them in the following cell in the notebook associated with this chapter (e.g., `creds = '{"username":"xxx","key":"xxx"}'`):

我需要一个API密钥来使用Kaggle API，点击Kaggle网站上你的形象头像，选择你的账户，然后点击创建新的API令牌，获取密钥。这会保存一个名为*kaggle.json*文件到你的计算机。在你的GPU服务器上你需要拷贝这个密钥。做了这个工作后，打开你下载的文件，拷贝内容并粘贴到与本章相关的notebook中的下述单元格上（例如，``creds = '{"username":"xxx","key":"xxx"}'``）:

```
creds = ''
```

Then execute this cell (this only needs to be run once):

执行这个单元格（这只需运行一次）：

```
cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)
```

Now you can download datasets from Kaggle! Pick a path to download the dataset to:

现在你能够从Kaggle上下载数据集了！选一个数据集下载到本地的路径：

```
path = URLs.path('bluebook')
path
```

Out: Path('/home/jhoward/.fastai/archive/bluebook')

```
#hide
Path.BASE_PATH = path
```

And use the Kaggle API to download the dataset to that path, and extract it:

使用Kaggle API下载数据集到这个路径，并抽取数据：

```
if not path.exists():
    path.mkdir(parents=true)
    api.competition_download_cli('bluebook-for-bulldozers', path=path)
    file_extract(path/'bluebook-for-bulldozers.zip')

path.ls(file_type='text')
```

Out: (#7) [Path('TrainAndValid.csv'),Path('Machine_Appendix.csv'),Path('random_forest_benchmark_test.csv'),Path('Test.csv'),Path('median_benchmark.csv'),Path('ValidSolution.csv'),Path('Valid.csv')]

Now that we have downloaded our dataset, let's take a look at it!

现在我们已经下载了我们的数据集，让我们查看一下它！

### Look at the Data

### 查看数据

Kaggle provides information about some of the fields of our dataset. The [Data](https://www.kaggle.com/c/bluebook-for-bulldozers/data) explains that the key fields in *train.csv* are:

- `SalesID`:: The unique identifier of the sale.
- `MachineID`:: The unique identifier of a machine. A machine can be sold multiple times.
- `saleprice`:: What the machine sold for at auction (only provided in *train.csv*).
- `saledate`:: The date of the sale.

Kaggle提供了一些我们数据集的字段信息。在*train.csv*中关键字段的[数据](https://www.kaggle.com/c/bluebook-for-bulldozers/data)解释是：

- `SalesID`：销售的唯一标示。
- `MachineID`：机械的唯一标示。一个机械能卖多次。
- `saleprice`：在拍卖上的机械销售情况（只在*train.csv*中提供了）。
- `saledate`：销售日期。

In any sort of data science work, it's important to *look at your data directly* to make sure you understand the format, how it's stored, what types of values it holds, etc. Even if you've read a description of the data, the actual data may not be what you expect. We'll start by reading the training set into a Pandas DataFrame. Generally it's a good idea to specify `low_memory=False` unless Pandas actually runs out of memory and returns an error. The `low_memory` parameter, which is `True` by default, tells Pandas to only look at a few rows of data at a time to figure out what type of data is in each column. This means that Pandas can actually end up using different data type for different rows, which generally leads to data processing errors or model training problems later.

Let's load our data and have a look at the columns:

任何形式的数据科学工作，*直接查看你的数据* 以确保你理解格式是很重要的，它是如何存贮的，它有什么类型的值，等等。即使你已经阅读了数据的描述，真实数据可能也不是你所期望的内容。我们会从通过阅读训练集到一个Pandas DataFrame中开始。通常具体说明`low_memory=False`是一个好主意，除非Pandas实际耗尽了内存并返回错误。`low_memory`参数默认为`真`，告诉Pandas一次只查看很少的几行数据，弄明白在每一列数据的类型是什么。意思是Pandas能够最终使用不同行的不同数据类型，其通常会导致数据处理错误或其后的模型训练问题。

```
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
```

```
df.columns
```

Out: Index(['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand',
       'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
       'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls'],
      dtype='object')

That's a lot of columns for us to look at! Try looking through the dataset to get a sense of what kind of information is in each one. We'll shortly see how to "zero in" on the most interesting bits.

At this point, a good next step is to handle *ordinal columns*. This refers to columns containing strings or similar, but where those strings have a natural ordering. For instance, here are the levels of `ProductSize`:

这对我们来说要查看很多列！尝试通看数据集来感受每一个列的信息是什么类型。我们会简短了解一下在最感兴趣的点上如何“锁定”。

```
df['ProductSize'].unique()
```

Out: array([nan, 'Medium', 'Small', 'Large / Medium', 'Mini', 'Large', 'Compact'], dtype=object)

We can tell Pandas about a suitable ordering of these levels like so:

我能够告诉Pandas合适的排序这些级别，像这样：

```
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
```

```
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
```

The most important data column is the dependent variable—that is, the one we want to predict. Recall that a model's metric is a function that reflects how good the predictions are. It's important to note what metric is being used for a project. Generally, selecting the metric is an important part of the project setup. In many cases, choosing a good metric will require more than just selecting a variable that already exists. It is more like a design process. You should think carefully about which metric, or set of metrics, actually measures the notion of model quality that matters to you. If no variable represents that metric, you should see if you can build the metric from the variables that are available.

最重要的数据列是因变量，即我们希望去预测的那个。回想一个模型的指标是一个函数，反映预测是如何的好。注意一个项目使用了什么指标是很重要的。通常，选择指标是项目设置的一个重要部分。在很多案例中，选择一个好的指标会需要不仅仅是只选择一个已经存在的变量。它更象是一个设计过程。你应该仔细的想那些指标，或指标集，实际测量对你来说很重要的那些模型质量概念。如果没有变量代表指标，你应该看是否你能够从那些有效的变量中创建指标。

However, in this case Kaggle tells us what metric to use: root mean squared log error (RMSLE) between the actual and predicted auction prices. We need do only a small amount of processing to use this: we take the log of the prices, so that `rmse` of that value will give us what we ultimately need:

然而，在本例中Kaggle告诉了我们所使用的指标：实际和预测拍卖价格间的均方根对数误差（RMSLE）。使用这个指标我们只需要做少量的处理：我们求价格的对数，所以值的`rmse`会提供给我们最终所需要的：

```
dep_var = 'SalePrice'
```

```
df[dep_var] = np.log(df[dep_var])
```

We are now ready to explore our first machine learning algorithm for tabular data: decision trees.

现在我们准备来探索我们第一个表格数据机器学习算法：决策树。