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

在本章我们使用的数据集来自Kaggle比赛推土机蓝皮书，其有如下描述：“比赛的目的是基于用途、设备类型和配置来预测在拍卖市场上一种特殊重装备的拍卖价格。数据来源来自拍卖结果发布，包括使用和设备配置信息。”

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
- `saleprice`：在拍卖会上的这台机械的出售情况（只在*train.csv*中提供了）。
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

## Decision Trees

## 决策树

Decision tree ensembles, as the name suggests, rely on decision trees. So let's start there! A decision tree asks a series of binary (that is, yes or no) questions about the data. After each question the data at that part of the tree is split between a "yes" and a "no" branch, as shown in <decision_tree>. After one or more questions, either a prediction can be made on the basis of all previous answers or another question is required.

决策树集合命名表明依赖的是决策树。所以我们从那里开始！一棵决策树问了一系列关于数据的二值（即是或否）问题。每个问题后，在树的相关位置数据分为一个“是”和一个“否”两个分支。一个或多个问题后，基于所有之前提供的答案或其它必须的问题，二择一的决策就可以做出了。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/decision_tree.png" alt="An example of decision tree" width="600" caption="An example of decision tree" id="decision_tree" />
  </p>
  <p align="center">图：决策树</p>
</div>

This sequence of questions is now a procedure for taking any data item, whether an item from the training set or a new one, and assigning that item to a group. Namely, after asking and answering the questions, we can say the item belongs to the same group as all the other training data items that yielded the same set of answers to the questions. But what good is this? The goal of our model is to predict values for items, not to assign them into groups from the training dataset. The value is that we can now assign a prediction value for each of these groups—for regression, we take the target mean of the items in the group.

这一系列的问题现在是一个获取任意数据项的过程（不管来自训练集的数据项或新的数据项），并把数据项分配到组中。即，问和回答一些问题后，我们能够描述数据项属于所有其它训练数据相同的组，这是由相同系列的问题回答产生的。但这有什么好处？我们模型的目标是预测数据项的值，而不是从训练集分配他们到组中去。这个值是我们现在能够对这些组的每一个分配一个预测值。为了回归，我们采用了组中数据项的目标平均数。

Let's consider how we find the right questions to ask. Of course, we wouldn't want to have to create all these questions ourselves—that's what computers are for! The basic steps to train a decision tree can be written down very easily:

让我们思虑我们如何查找正确的问题来回答。当然，我们不希望必须自己创建这些所有问题，那是计算机做的事情！训练一棵决策树基础步骤能够被非常容易的写下来：

1. Loop through each column of the dataset in turn.
2. For each column, loop through each possible level of that column in turn.
3. Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable).
4. Find the average sale price for each of those two groups, and see how close that is to the actual sale price of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average sale price of the item's group.
5. After looping through all of the columns and all the possible levels for each, pick the split point that gave the best predictions using that simple model.
6. We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each by going back to step 1 for each group.
7. Continue this process recursively, until you have reached some stopping criterion for each group—for instance, stop splitting a group further when it has only 20 items in it.

1. 依次循环数据集的每一列。
2. 对于每一列，依次循环那一列的每个可能等级。
3. 基于他们是否比那个值更大或更小（或基于他们是否等于或不等于那个分类变量的那个等级，它是否是一个分类变量），尝试划分数据为两个组，
4. 查找那两组的每一个平均销售价格，并查看是怎样接近那个组中设备的每个项目的实际售价。即，作为一个非常简单的“模型”来处理，我们的预测是简单的数据项组的平均售价。
5. 对于所有列和所有可能等级的每个循环后，使用简单模型选择给出的最佳预测的分割点。
6. 基于这个选择的分割，现在我们数据有了两个不同的组。处理每一个这些分开的数据集，并返回到第一步对每组查找每个最佳的分割。
7. 明治维新这个递归处理，知道你已经搜索到了每一组的停止标准。例如，当一个组中只有20个数据项时停止进一步的分割这个组。

Although this is an easy enough algorithm to implement yourself (and it is a good exercise to do so), we can save some time by using the implementation built into sklearn.

不过这是一个你自己能够实现的足够容易的算法（并且做这个事情它是一个好的练习），通过使用sklearn内置的实现，我们能够节省很多时间。

First, however, we need to do a little data preparation.

因此，首先，我们需要来做一点数据准备。

> A: Here's a productive question to ponder. If you consider that the procedure for defining a decision tree essentially chooses one *sequence of splitting questions about variables*, you might ask yourself, how do we know this procedure chooses the *correct sequence*? The rule is to choose the splitting question that produces the best split (i.e., that most accurately separates the items into two distinct categories), and then to apply the same rule to the groups that split produces, and so on. This is known in computer science as a "greedy" approach. Can you imagine a scenario in which asking a “less powerful” splitting question would enable a better split down the road (or should I say down the trunk!) and lead to a better result overall?

> 亚：这是一个值得深思的问题。如果你考虑处理定义一个决策树必须选择一系列分割问题的变量，你可能会问你自己，我们知道如何做这个选择正确系列的处理？规则是选择的分割问题产生最好的分割（即，最精确的分割数据项到两个不同的分类），然后应用相同的规则到这些组产生分割，诸如此类。在计算机科学中这被称为“贪婪”方法。问一个“稍微弱点”的分割问题能够更好的分解路径（或我应该说向下走！），就会有一个更好的整体结果，你能想像这个场景吗？

### Handling Dates

### 处理日期

The first piece of data preparation we need to do is to enrich our representation of dates. The fundamental basis of the decision tree that we just described is *bisection*— dividing a group into two. We look at the ordinal variables and divide up the dataset based on whether the variable's value is greater (or lower) than a threshold, and we look at the categorical variables and divide up the dataset based on whether the variable's level is a particular level. So this algorithm has a way of dividing up the dataset based on both ordinal and categorical data.

数据准备的第一部分，我们需要做的是丰富我们的日期描述。决策树基础的基础是我们只描述*对分*（把组一分为二）。我们查看序数变量，然后依据变量大于（或小于）阈值而划分数据集，并且我们查看分类变量，其后基于变量的等级是否是特定的等级而划分数据集。所以这个算法有基于序数和分类数据划分数据集的方法。

But how does this apply to a common data type, the date? You might want to treat a date as an ordinal value, because it is meaningful to say that one date is greater than another. However, dates are a bit different from most ordinal values in that some dates are qualitatively different from others in a way that that is often relevant to the systems we are modeling.

但如何应用到一个普通的数据类型，日期？你可能希望把日期作为一个原始值来处理，因为一个日期比别一个更大是有意义的。然而，日期日期与绝大多数序数变脸有一点不同，在那些日期中在一程度上与其它日期是有本质区别的，其经常与我们正在建模的系统相关联。

In order to help our algorithm handle dates intelligently, we'd like our model to know more than whether a date is more recent or less recent than another. We might want our model to make decisions based on that date's day of the week, on whether a day is a holiday, on what month it is in, and so forth. To do this, we replace every date column with a set of date metadata columns, such as holiday, day of week, and month. These columns provide categorical data that we suspect will be useful.

为了帮助我们算法有智慧的处理日期，我们希望我们的模型知道的更多，而不仅仅一个日期是否比另外的日期更新或不是更新。我们可能希望我们的模型基于那个日期是星期几、是否是一个假日、它是在什么月份，诸如此类等做出决策。我们用一系列日期原数据列来替换每一个日期列，如假日、星期几和月份来做这个事情。这些列提供了分类日期，我们推测会有用的。

fastai comes with a function that will do this for us—we just have to pass a column name that contains dates:

fastai提供了一个函数为我们做这个事情，我们只需要传递包含那些日期的列名：

```
df = add_datepart(df, 'saledate')
```

Let's do the same for the test set while we're there:

让我们对测试集做同样的操作

```
df_test = pd.read_csv(path/'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')
```

We can see that there are now lots of new columns in our DataFrame:

我们能够看到现在在我们的DataFrame中有一些新的列：

```
' '.join(o for o in df.columns if o.startswith('sale'))
```

Out: 'saleWeek saleYear saleMonth saleDay saleDayofweek saleDayofyear saleIs_month_end saleIs_month_start saleIs_quarter_end saleIs_quarter_start saleIs_year_end saleIs_year_start saleElapsed'

This is a good first step, but we will need to do a bit more cleaning. For this, we will use fastai objects called `TabularPandas` and `TabularProc`.

这是一个非常好的起步阶段，但是我们将需要做进一点的清溪。为此，我们会使用名为`TabularPandas`和`TrbularProc`的fastai对象。

### Using TabularPandas and TabularProc

### 使用 TabularPandas 和 TabularProc

A second piece of preparatory processing is to be sure we can handle strings and missing data. Out of the box, sklearn cannot do either. Instead we will use fastai's class `TabularPandas`, which wraps a Pandas DataFrame and provides a few conveniences. To populate a `TabularPandas`, we will use two `TabularProc`s, `Categorify` and `FillMissing`. A `TabularProc` is like a regular `Transform`, except that:

预处理的第二部分是确保我们能够处理字符串和缺失的数据。sklearn不能做到这些工作。我们会用fastai的类`TabularPandas`来替代，其包装了Pandas DataFrame并提供了一些便利性。我们会用两个`TabularProc`，`Catagorify`和`FillMissing`来填充`TabularPandas`。`TabularProc`像一个均匀的`转换`，除了：

- It returns the exact same object that's passed to it, after modifying the object in place.
- It runs the transform once, when data is first passed in, rather than lazily as the data is accessed.

- 在恰当的位置修改对象后，它返回与传给它的完全相同的对象。
- 当数据首次传入时，它运行一次转换，而不是在访问数据时有迟缓。

`Categorify` is a `TabularProc` that replaces a column with a numeric categorical column. `FillMissing` is a `TabularProc` that replaces missing values with the median of the column, and creates a new Boolean column that is set to `True` for any row where the value was missing. These two transforms are needed for nearly every tabular dataset you will use, so this is a good starting point for your data processing:

`Categorify`是一个`TabularProc`，用一个数值分类列来替换一个列。`FillMissing`是一个`TabularProc`，用这个列的中值来替换那些缺失值，并创建一个新的布尔列设置那些值确实的行为`真`。对于几乎所有表格数据集这两个转换是你所需要使用的，所以对于你的数据处理这是一个好的开始点：

```
procs = [Categorify, FillMissing]
```

`TabularPandas` will also handle splitting the dataset into training and validation sets for us. However we need to be very careful about our validation set. We want to design it so that it is like the *test set* Kaggle will use to judge the contest.

`TabularPandas`也会为我们处理分割数据集为训练集和验证集。然而我们需要小心的处理验证集。我们希望设计的它像Kaggle用于评判比赛的*测试集*那样。

Recall the distinction between a validation set and a test set, as discussed in <chapter_intro>. A validation set is data we hold back from training in order to ensure that the training process does not overfit on the training data. A test set is data that is held back even more deeply, from us ourselves, in order to ensure that *we* don't overfit on the validation data, as we explore various model architectures and hyperparameters.

回想在<章节：概述>中讨论的验证集和测试集之间的区别。一个验证集是从训练中隐藏下来的数据，为了确保在训练数据上的训练过程不会过拟合。一个测试集是对我们甚至更深层的隐藏数据，为了确保*我们* 不会在验证数据上过拟，作为我们探索模型架构和超参。

We don't get to see the test set. But we do want to define our validation data so that it has the same sort of relationship to the training data as the test set will have.

我们没有看到测试集。但是我们希望定义我们的验证集以便它像测试集那样与测试数据有相同的关系。

In some cases, just randomly choosing a subset of your data points will do that. This is not one of those cases, because it is a time series.

在一些案例中，只是随机选择一个你的数据点的子集来做这个事情。这和那些案例不一样，因为它是一个时间序列。

If you look at the date range represented in the test set, you will discover that it covers a six-month period from May 2012, which is later in time than any date in the training set. This is a good design, because the competition sponsor will want to ensure that a model is able to predict the future. But it means that if we are going to have a useful validation set, we also want the validation set to be later in time than the training set. The Kaggle training data ends in April 2012, so we will define a narrower training dataset which consists only of the Kaggle training data from before November 2011, and we'll define a validation set consisting of data from after November 2011.

如果你查看了测试集中数据范围描述，你会发现它覆盖了从2012年5月开始的6个月的期间，在时间上它比训练集中的任何数据都要晚。这是一个好的设计，因为比赛的赞助方希望确保一个模型能够预测未来。但这表示如果我们希望有一个有用的验证集，我们也要验证集在时间上比训练集晚。Kaggle训练数据截止于2012年4月，所以我们会定义一个差距小的训练集，只是从2011年11月以前的连续的Kaggle训练数据，且我们会定义验证集从2011年11月以后的连续数据。

To do this we use `np.where`, a useful function that returns (as the first element of a tuple) the indices of all `True` values:

我们使用`np.where`来做这个操作，这是一个有用的函数，返回（作为元组的第一个元素）所有`True`值的索引：

```
cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))
```

`TabularPandas` needs to be told which columns are continuous and which are categorical. We can handle that automatically using the helper function `cont_cat_split`:

`TabularPandas`需要被告知那些列是连续的，哪些是分类。我们能够使用帮助函数`cont_cat_split`自动处理：

```
cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
```

```
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
```

A `TabularPandas` behaves a lot like a fastai `Datasets` object, including providing `train` and `valid` attributes:

`TabularPandas`表现很像一个fastai `Datasets` 对象，提供包含 `train` 和 `valid` 属性：

```
len(to.train),len(to.valid)
```

Out: (404710, 7988)

We can see that the data is still displayed as strings for categories (we only show a few columns here because the full table is too big to fit on a page):

我们能够看到数据对于分类一直以字符串显示（我们在这里只展示了少数几列，因为全表太大了不能在一页放不下）：

```
#hide_output
to.show(3)
```

|      | saleWeek | UsageBand | fiModelDesc | fiBaseModel | fiSecondaryDesc | fiModelSeries | fiModelDescriptor | ProductSize |                                         fiProductClassDesc |          state | ProductGroup |   ProductGroupDesc | Drive_System |  Enclosure |               Forks | Pad_Type |        Ride_Control | Stick | Transmission | Turbocharged | Blade_Extension | Blade_Width | Enclosure_Type | Engine_Horsepower | Hydraulics | Pushblock | Ripper | Scarifier | Tip_Control |           Tire_Size |             Coupler |      Coupler_System |      Grouser_Tracks | Hydraulics_Flow | Track_Type | Undercarriage_Pad_Width | Stick_Length | Thumb | Pattern_Changer | Grouser_Type | Backhoe_Mounting | Blade_Type | Travel_Controls | Differential_Type | Steering_Controls | saleIs_month_end | saleIs_month_start | saleIs_quarter_end | saleIs_quarter_start | saleIs_year_end | saleIs_year_start | saleElapsed | auctioneerID_na | MachineHoursCurrentMeter_na | SalesID | MachineID | ModelID | datasource | auctioneerID | YearMade | MachineHoursCurrentMeter | saleYear | saleMonth | saleDay | saleDayofweek | saleDayofyear | SalePrice |
| ---: | -------: | --------: | ----------: | ----------: | --------------: | ------------: | ----------------: | ----------: | ---------------------------------------------------------: | -------------: | -----------: | -----------------: | -----------: | ---------: | ------------------: | -------: | ------------------: | ----: | -----------: | -----------: | --------------: | ----------: | -------------: | ----------------: | ---------: | --------: | -----: | --------: | ----------: | ------------------: | ------------------: | ------------------: | ------------------: | --------------: | ---------: | ----------------------: | -----------: | ----: | --------------: | -----------: | ---------------: | ---------: | --------------: | ----------------: | ----------------: | ---------------: | -----------------: | -----------------: | -------------------: | --------------: | ----------------: | ----------: | --------------: | --------------------------: | ------: | --------: | ------: | ---------: | -----------: | -------: | -----------------------: | -------: | --------: | ------: | ------------: | ------------: | --------: |
|    0 |       46 |       Low |        521D |         521 |               D |          #na# |              #na# |        #na# |                   Wheel Loader - 110.0 to 120.0 Horsepower |        Alabama |           WL |       Wheel Loader |         #na# | EROPS w AC | None or Unspecified |     #na# | None or Unspecified |  #na# |         #na# |         #na# |            #na# |        #na# |           #na# |              #na# |    2 Valve |      #na# |   #na# |      #na# |        #na# | None or Unspecified | None or Unspecified |                #na# |                #na# |            #na# |       #na# |                    #na# |         #na# |  #na# |            #na# |         #na# |             #na# |       #na# |            #na# |          Standard |      Conventional |            False |              False |              False |                False |           False |             False |  1163635200 |           False |                       False | 1139246 |    999089 |    3157 |        121 |          3.0 |     2004 |                     68.0 |     2006 |        11 |      16 |             3 |           320 | 11.097410 |
|    1 |       13 |       Low |      950FII |         950 |               F |            II |              #na# |      Medium |                   Wheel Loader - 150.0 to 175.0 Horsepower | North Carolina |           WL |       Wheel Loader |         #na# | EROPS w AC | None or Unspecified |     #na# | None or Unspecified |  #na# |         #na# |         #na# |            #na# |        #na# |           #na# |              #na# |    2 Valve |      #na# |   #na# |      #na# |        #na# |                23.5 | None or Unspecified |                #na# |                #na# |            #na# |       #na# |                    #na# |         #na# |  #na# |            #na# |         #na# |             #na# |       #na# |            #na# |          Standard |      Conventional |            False |              False |              False |                False |           False |             False |  1080259200 |           False |                       False | 1139248 |    117657 |      77 |        121 |          3.0 |     1996 |                   4640.0 |     2004 |         3 |      26 |             4 |            86 | 10.950807 |
|    2 |        9 |      High |         226 |         226 |            #na# |          #na# |              #na# |        #na# | Skid Steer Loader - 1351.0 to 1601.0 Lb Operating Capacity |       New York |          SSL | Skid Steer Loaders |         #na# |      OROPS | None or Unspecified |     #na# |                #na# |  #na# |         #na# |         #na# |            #na# |        #na# |           #na# |              #na# |  Auxiliary |      #na# |   #na# |      #na# |        #na# |                #na# | None or Unspecified | None or Unspecified | None or Unspecified |        Standard |       #na# |                    #na# |         #na# |  #na# |            #na# |         #na# |             #na# |       #na# |            #na# |              #na# |              #na# |            False |              False |              False |                False |           False |             False |  1077753600 |           False |                       False | 1139249 |    434808 |    7009 |        121 |          3.0 |     2001 |                   2838.0 |     2004 |         2 |      26 |             3 |            57 |  9.210340 |

```
#hide_input
to1 = TabularPandas(df, procs, ['state', 'ProductGroup', 'Drive_System', 'Enclosure'], [], y_names=dep_var, splits=splits)
to1.show(3)
```

|      |          state | ProductGroup | Drive_System |  Enclosure | SalePrice |
| ---: | -------------: | -----------: | -----------: | ---------: | --------: |
|    0 |        Alabama |           WL |         #na# | EROPS w AC | 11.097410 |
|    1 | North Carolina |           WL |         #na# | EROPS w AC | 10.950807 |
|    2 |       New York |          SSL |         #na# |      OROPS |  9.210340 |

However, the underlying items are all numeric:

然而，下面的数据项全是数字：

```
#hide_output
to.items.head(3)
```

|      | SalesID | SalePrice | MachineID | saleWeek |  ... | saleIs_year_start | saleElapsed | auctioneerID_na | MachineHoursCurrentMeter_na |
| ---: | ------: | --------: | --------: | -------: | ---: | ----------------: | ----------: | --------------: | --------------------------: |
|    0 | 1139246 | 11.097410 |    999089 |       46 |  ... |                 1 |        2647 |               1 |                           1 |
|    1 | 1139248 | 10.950807 |    117657 |       13 |  ... |                 1 |        2148 |               1 |                           1 |
|    2 | 1139249 |  9.210340 |    434808 |        9 |  ... |                 1 |        2131 |               1 |                           1 |

3 rows × 67 columns

3行 × 67列

```
#hide_input
to1.items[['state', 'ProductGroup', 'Drive_System', 'Enclosure']].head(3)
```

|      | state | ProductGroup | Drive_System | Enclosure |
| ---: | ----: | -----------: | -----------: | --------: |
|    0 |     1 |            6 |            0 |         3 |
|    1 |    33 |            6 |            0 |         3 |
|    2 |    32 |            3 |            0 |         6 |

The conversion of categorical columns to numbers is done by simply replacing each unique level with a number. The numbers associated with the levels are chosen consecutively as they are seen in a column, so there's no particular meaning to the numbers in categorical columns after conversion. The exception is if you first convert a column to a Pandas ordered category (as we did for `ProductSize` earlier), in which case the ordering you chose is used. We can see the mapping by looking at the `classes` attribute:

分类列转化为数值是通过用数字简单的替换每个唯一等级来完成的。与等级相关联的这些数字是在列是它们看到的内容连续选择的，所以没有转换后分类列中的数字没有特别的含义。例外是如果你第一次转换一个列到一个Pandas顺序分类（我们早前对`ProductSize`做的事情），在这种情况下顺序你的选择是有用的。我们能够通过看`classes`属性来查看映射：

```
to.classes['ProductSize']
```

Out: ['#na#', 'Large', 'Large / Medium', 'Medium', 'Small', 'Mini', 'Compact']

Since it takes a minute or so to process the data to get to this point, we should save it—that way in the future we can continue our work from here without rerunning the previous steps. fastai provides a `save` method that uses Python's *pickle* system to save nearly any Python object:

因为做这个事情花费了一些时间或处理数据到了这个阶段，我们应该保存下它。在这种情况下在未来我们能够从这里继续我们的工作，而不用重复运行之前的步骤。

```
save_pickle(path/'to.pkl',to)
```

To read this back later, you would type:

其后读取回这个信息，你可以输入：

```python
to = (path/'to.pkl').load()
```

Now that all this preprocessing is done, we are ready to create a decision tree.

现在这个预处理的所有工作做完了，我们准备创建一棵决策树。

### Creating the Decision Tree

### 创建决策树

To begin, we define our independent and dependent variables:

我们从定义自变量和因变量开始：

```
#hide
to = load_pickle(path/'to.pkl')
```

```
xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y
```

Now that our data is all numeric, and there are no missing values, we can create a decision tree:

因为我们的数据都是数值型，且没有丢失的数据，我们能够创建一棵决策树了：

```
m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);
```

To keep it simple, we've told sklearn to just create four *leaf nodes*. To see what it's learned, we can display the tree:

为保持简单操作，我们告诉sklearn只创建四*叶节点*。来看一下它学到了什么，我们展示这棵树：

```
draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)
```

Out: <img src="./_v_images/decision_tree.jpg" alt="decision_tree" style="zoom:40%;" />

Understanding this picture is one of the best ways to understand decision trees, so we will start at the top and explain each part step by step.

理解这个图像是理解决策树的最好的方法之一，我们会从顶端开始并一步步的解释每一部分。

The top node represents the *initial model* before any splits have been done, when all the data is in one group. This is the simplest possible model. It is the result of asking zero questions and will always predict the value to be the average value of the whole dataset. In this case, we can see it predicts a value of 10.10 for the logarithm of the sales price. It gives a mean squared error of 0.48. The square root of this is 0.69. (Remember that unless you see `m_rmse`, or a *root mean squared error*, then the value you are looking at is before taking the square root, so it is just the average of the square of the differences.) We can also see that there are 404,710 auction records in this group—that is the total size of our training set. The final piece of information shown here is the decision criterion for the best split that was found, which is to split based on the `coupler_system` column.

当所有数据在一个组时，顶部节点代表任何分割完成前的*初始模型*。这是最简单的可能模型。它是问零个问题的结果并会一直预测整个数据集的平均值。在这种情况下，我们能看到它预测了销售价格的对数为 10.10 的值。它给出均方误差为 0.48。平方根是 0.69。（记住，除非你看到 `m_rmse` 或一个*均方根误差*，然后你看到的是求平均根前的值，所以这只是差值平均的平均值。）我们也能够看到在这个组有 404710 条拍卖记录，这是我们训练集的全部大小。信息的最后部分展示的是所发现的最佳分割的决策标准，它是基于`coupler_system`列做的分割。

Moving down and to the left, this node shows us that there were 360,847 auction records for equipment where `coupler_system` was less than 0.5. The average value of our dependent variable in this group is 10.21. Moving down and to the right from the initial model takes us to the records where `coupler_system` was greater than 0.5.

向左下移动，这个节点给我们展示了有 360,847 条设备拍卖记录，这里`coupler_system`小于0.5。在这一组我们因变量的平均值为10.21。从初始模型向右下移动，带我们到了`coupler_system`比 0.5 大的记录。

The bottom row contains our *leaf nodes*: the nodes with no answers coming out of them, because there are no more questions to be answered. At the far right of this row is the node containing records where `coupler_system` was greater than 0.5. The average value here is 9.21, so we can see the decision tree algorithm did find a single binary decision that separated high-value from low-value auction results. Asking only about `coupler_system` predicts an average value of 9.21 versus 10.1.

底部行是我们的*叶节点*：这些节点不用回答问题就能得出，因为没有更多的问题来回答。这一行的最右侧包含 `coupler_system`比 0.5 大的记录。这里的平均值是 9.21 ，所以我们能看到决策树算法找到了一个单二分决策，其把低位值拍卖与高位值拍卖分开。只用问关于 `coupler_system` 一个平均值 9.21 对比 10.1 的预测。

Returning back to the top node after the first decision point, we can see that a second binary decision split has been made, based on asking whether `YearMade` is less than or equal to 1991.5. For the group where this is true (remember, this is now following two binary decisions, based on `coupler_system` and `YearMade`) the average value is 9.97, and there are 155,724 auction records in this group. For the group of auctions where this decision is false, the average value is 10.4, and there are 205,123 records. So again, we can see that the decision tree algorithm has successfully split our more expensive auction records into two more groups which differ in value significantly.

第一个决策点后找回到顶部节点，我们能够看到基于问 `YearMade` 是否是小于或等于 1991.5 ，第二个二分决策分割已经完成。对于为真（记住，现在遵循的二分决策是基于 `coupler_system` 和 `YearMade`）的这一组平均值是 9.97 ，在这一组有 155,724 条拍卖记录。对决策为假这一组的拍卖，平均值为 10.4 ，这一组有 205,123 条记录。所以再一次，我们能够看到决策算法已经成功的分割更多昂贵的拍卖记录进入另外两个组，这些组值的区分是显著的。

We can show the same information using Terence Parr's powerful [dtreeviz](https://explained.ai/decision-tree-viz/) library:

我们可以使用特伦斯·帕尔强大的 [dtreeviz](https://explained.ai/decision-tree-viz/) 库来展示相同的信息：

```
samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')
```

Out: <img src="./_v_images/decision_tree_dtreeviz.jpg" alt="decision_tree" style="zoom:40%;" />

This shows a chart of the distribution of the data for each split point. We can clearly see that there's a problem with our `YearMade` data: there are bulldozers made in the year 1000, apparently! Presumably this is actually just a missing value code (a value that doesn't otherwise appear in the data and that is used as a placeholder in cases where a value is missing). For modeling purposes, 1000 is fine, but as you can see this outlier makes visualization of the values we are interested in more difficult. So, let's replace it with 1950:

这展示了对每一个分割点的数据分配图。我们能够清晰的看到我们的 `YearMade` 数据有一个问题：有制造年份为1000年的推土机，很明显！大概实际上只是丢失了值代码（在数据中不会以其它方式显示，在案例中值是丢失的其作为一个占位符被使用）。对于建模的目的 1000 是没问题的，但正如我们看到的这个离群值使得我们感兴趣的数值的可视化更困难了。所以我们用 1950 来替换它：

```
xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950
```

That change makes the split much clearer in the tree visualization, even although it doesn't actually change the result of the model in any significant way. This is a great example of how resilient decision trees are to data issues!

这个改变使用的树可视化的分割更清晰了，虽然甚至它没有用任何显著的方法实际改变模型的结果。这是一个非常好的例子，来说明决策树对数据问题的适应情况如何！

```
m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')
```

Out: <img src="./_v_images/decision_tree_dtreeviz_2.jpeg" alt="decision_tree" style="zoom:40%;" />

Let's now have the decision tree algorithm build a bigger tree. Here, we are not passing in any stopping criteria such as `max_leaf_nodes`:

现在让决策树算法构建一棵更大的树。这里我们不会传递任何停止标准，如 `max_leaf_modes` ：

```
m = DecisionTreeRegressor()
m.fit(xs, y);
```

We'll create a little function to check the root mean squared error of our model (`m_rmse`), since that's how the competition was judged:

我们会创建一个小函数来检查我们模型的均方根误差（ `m_rmse` ），因为这是如何评判比赛的：

```
def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)
```

```
m_rmse(m, xs, y)
```

Out: 0.0

So, our model is perfect, right? Not so fast... remember we really need to check the validation set, to ensure we're not overfitting:

所以我们的模型是完美的，对吗？ 不要高兴的太早... 记住我们需要实际的检查验证集，以确保我们不会过拟合：

```
m_rmse(m, valid_xs, valid_y)
```

Out: 0.331466

Oops—it looks like we might be overfitting pretty badly. Here's why:

哎哟，它好像很严重的过拟了。原因如下：

```
m.get_n_leaves(), len(xs)
```

Out: (324544, 404710)

We've got nearly as many leaf nodes as data points! That seems a little over-enthusiastic. Indeed, sklearn's default settings allow it to continue splitting nodes until there is only one item in each leaf node. Let's change the stopping rule to tell sklearn to ensure every leaf node contains at least 25 auction records:

我们获取了太多的叶节点做为数据点！这好像是有点过于乐观了。实际上，sklearn的默认设置允许连续的分割节点，直到每个叶节点中只有一个数据项。让我们改变停止规则，告诉sklearn确保每个叶节点至少包含 25 条拍卖记录：

```
m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
```

Out: (0.248562, 0.323396)

That looks much better. Let's check the number of leaves again:

这看起来好多了。 我们再次检查一下叶子的数目：

```
m.get_n_leaves()
```

Out: 12397

Much more reasonable!

更合理了！

> A: Here's my intuition for an overfitting decision tree with more leaf nodes than data items. Consider the game Twenty Questions. In that game, the chooser secretly imagines an object (like, "our television set"), and the guesser gets to pose 20 yes or no questions to try to guess what the object is (like "Is it bigger than a breadbox?"). The guesser is not trying to predict a numerical value, but just to identify a particular object out of the set of all imaginable objects. When your decision tree has more leaves than there are possible objects in your domain, then it is essentially a well-trained guesser. It has learned the sequence of questions needed to identify a particular data item in the training set, and it is "predicting" only by describing that item's value. This is a way of memorizing the training set—i.e., of overfitting.

> 亚：这是我对叶节点比数据项多的过拟决策树的直觉。思虑20个问题游戏。在这个游戏中，选择者秘密的想像一个物体（如，“我们的电视机集”），猜测者摆出20个是的或没有问题尝试猜测物体是什么（如“它比面包箱更大吗？”）。猜测者不是尝试预测一个数字值，只是在所有可想像的物体的集中来辨认出一个特定的物体。当你的决策树的叶子比你想到的可能物体更多时，那么它必须是一个训练有素的猜测者。在训练集中它已经学习了一系列需要识别一个特定数据项的问题，且它只是通过描述数据项的值来“预测”。这是一种记忆训练集的方法，即过拟合。

Building a decision tree is a good way to create a model of our data. It is very flexible, since it can clearly handle nonlinear relationships and interactions between variables. But we can see there is a fundamental compromise between how well it generalizes (which we can achieve by creating small trees) and how accurate it is on the training set (which we can achieve by using large trees).

创建一棵决策树是一个创建我们数据模型的好方法。它是非常灵活的，因为它能够清晰的处理非线性关系和变量间的交互。但是我们能够看到，在它如何好的泛化和在训练集上的精度如何之间有一个基本的妥协（我们能够通过使用大型树来完成）。

So how do we get the best of both worlds? We'll show you right after we handle an important missing detail: how to handle categorical variables.

那么我们怎么做到两全其美呢？在我们处理完一个关键缺失细节后，我们会立刻给你展示：如何处理分类变量。

### Categorical Variables

### 分类变量

In the previous chapter, when working with deep learning networks, we dealt with categorical variables by one-hot encoding them and feeding them to an embedding layer. The embedding layer helped the model to discover the meaning of the different levels of these variables (the levels of a categorical variable do not have an intrinsic meaning, unless we manually specify an ordering using Pandas). In a decision tree, we don't have embeddings layers—so how can these untreated categorical variables do anything useful in a decision tree? For instance, how could something like a product code be used?

在上一章节，当与深度学习网络配合时，我们通过独热编码处理分类变量并把他们喂给一个嵌入层。嵌入层帮助模型发现这些不同等级变量的意义（变量的等级没有真正的意思，除非用Pandas我们手动具体排序）。在一棵决策树中，我们没有嵌入层，那么这些未处理的分类变量能够怎样在一棵决策树中做有帮助的事情呢？例如，像产品代码这种内容能够如何被使用？

The short answer is: it just works! Think about a situation where there is one product code that is far more expensive at auction than any other one. In that case, any binary split will result in that one product code being in some group, and that group will be more expensive than the other group. Therefore, our simple decision tree building algorithm will choose that split. Later during training the algorithm will be able to further split the subgroup that contains the expensive product code, and over time, the tree will home in on that one expensive product.

简短的回答是：它行的通！思考一种情况，有一个产品代码在拍卖价格上比其它要更昂贵。在那种情况下，任何二值分割将会使用那一产品代码分到某一组中，且那个组会比其它组更昂贵。因此，我们简单的决策树创建算法会选择那个分割。其后训练期间的算法能够进一步分割包含昂贵产品代码的子级，随着时间的推移，树会导向追踪那一昂贵的产品。

It is also possible to use one-hot encoding to replace a single categorical variable with multiple one-hot-encoded columns, where each column represents a possible level of the variable. Pandas has a `get_dummies` method which does just that.

这也可以用独热编码，用多个独热编码列替换一个单分类变量，很一独热编码列代表变量的可能等级。Pandas有一个正好做这个事情的`get_dummies`方法。

However, there is not really any evidence that such an approach improves the end result. So, we generally avoid it where possible, because it does end up making your dataset harder to work with. In 2019 this issue was explored in the paper ["Splitting on Categorical Predictors in Random Forests"](https://peerj.com/articles/6339/) by Marvin Wright and Inke König, which said:

然而，没有任何真实争取这一方面会改善最终的结果。所以，我们通常尽可能避免使用它，因为它最终使用你的数据集更难使用。在2019年这个问题在马文·赖特和因克·科尼格编写的论文中已经被探索过了，在[在随机树中分类预测因子的分割](https://peerj.com/articles/6339/)中描述到：

> : The standard approach for nominal predictors is to consider all $2^{k-1} − 1$ 2-partitions of the *k* predictor categories. However, this exponential relationship produces a large number of potential splits to be evaluated, increasing computational complexity and restricting the possible number of categories in most implementations. For binary classification and regression, it was shown that ordering the predictor categories in each split leads to exactly the same splits as the standard approach. This reduces computational complexity because only *k* − 1 splits have to be considered for a nominal predictor with *k* categories.

> ：对于名义预测因子的标准方法是思考所有的 $2^{k-1} − 1$  *k* 预测因子分类的二个分割。然而，这种指数关系产生了巨大数量的需要评估的潜在分割，在大多数实践中增加了计算的复杂度和限制了可能数量的分类。对于二值分类和回归，它显示 了在每一个分割中对预测因子分类排序，与标准方法完全相同的分割。这减小计算的复杂度，因为对于有 *k* 各分类名义预测因子来说只有*k* - 1个分割必须被考虑。

Now that you understand how decisions tree work, it's time for the best-of-both-worlds solution: random forests.

现在你理解了决策树如何工作的，两全其美的解决方案也是时候了：随机森林。

## Random Forests

## 随机森林

In 1994 Berkeley professor Leo Breiman, one year after his retirement, published a small technical report called ["Bagging Predictors"](https://www.stat.berkeley.edu/~breiman/bagging.pdf), which turned out to be one of the most influential ideas in modern machine learning. The report began:

在1994年，伯克利大学教授莱奥·布雷曼在它退休后的一年发表了一个小的技术报告称为["装袋预测因子"](https://www.stat.berkeley.edu/~breiman/bagging.pdf)，其结果它是在于机器学习领域中最具有影响力的思想之一。报告的开始：

> : Bagging predictors is a method for generating multiple versions of a predictor and using these to get an aggregated predictor. The aggregation averages over the versions... The multiple versions are formed by making bootstrap replicates of the learning set and using these as new learning sets. Tests… show that bagging can give substantial gains in accuracy. The vital element is the instability of the prediction method. If perturbing the learning set can cause significant changes in the predictor constructed, then bagging can improve accuracy.

> ：装袋预测器因子是一种生成多个预测因子版本并用其得到一个聚合预测因子。在这些版本之上的聚合平均... 这些多版本是通过使得学习集引导复制形成的并其作为新的学习庥。测试... 显示装袋能够提供可观的精度收获。关键因素是预测方法的不稳定。如果干扰学习集能够在预测器构建上引发显著的改变，然后装袋方法能够改善精度。

Here is the procedure that Breiman is proposing:

1. Randomly choose a subset of the rows of your data (i.e., "bootstrap replicates of your learning set").
2. Train a model using this subset.
3. Save that model, and then return to step 1 a few times.
4. This will give you a number of trained models. To make a prediction, predict using all of the models, and then take the average of each of those model's predictions.

这是布雷曼提出的步骤：

1. 随机选择你的数据行子集（即，“你的学习集的步进复制”）。
2. 用这个子集训练模型。
3. 保存模型，然后返回步骤1几次。
4. 这会给你许多训练后的模型。做出预测，用所有的模型预测，然后取那些每个模型预测的平均值。

This procedure is known as "bagging." It is based on a deep and important insight: although each of the models trained on a subset of data will make more errors than a model trained on the full dataset, those errors will not be correlated with each other. Different models will make different errors. The average of those errors, therefore, is: zero! So if we take the average of all of the models' predictions, then we should end up with a prediction that gets closer and closer to the correct answer, the more models we have. This is an extraordinary result—it means that we can improve the accuracy of nearly any kind of machine learning algorithm by training it multiple times, each time on a different random subset of the data, and averaging its predictions.

这个过程被称为“装袋”。它是基于一个深层且重要的理解：虽然每个在数据子集上训练的模型相比在数据全集上的训练模型会产生更多的错误，那些错误不会彼此相互关联。因此那些错误的平均值是：零！所以如果我们有更多的模型，我们求了所有模型预测的平均值，其后我们预测最终应该变的越来越接近正确答案。这是一个非凡的结果，它的意思是每一次在不同的随机数据子集上，通过训练它多次，并平均它们的预测，我们几乎能够改善任何类型机器学习算法的精度。

In 2001 Leo Breiman went on to demonstrate that this approach to building models, when applied to decision tree building algorithms, was particularly powerful. He went even further than just randomly choosing rows for each model's training, but also randomly selected from a subset of columns when choosing each split in each decision tree. He called this method the *random forest*. Today it is, perhaps, the most widely used and practically important machine learning method.

在2001年莱奥·布雷曼想要证明当应用到决策树构建算法时，这一方法来构建模型是尤为强大的。他甚至更加深入，不仅仅随机选择每个模型训练的行，而且当选择每个决策树中的每个分割时随机选择了列的子集。它称这一方法为*随机木森*。也许，现如今它是最为广泛使用和尤为重要的机器学习方法。

In essence a random forest is a model that averages the predictions of a large number of decision trees, which are generated by randomly varying various parameters that specify what data is used to train the tree and other tree parameters. Bagging is a particular approach to "ensembling," or combining the results of multiple models together. To see how it works in practice, let's get started on creating our own random forest!

一个随机森林的本质是平均大量决策树预测的模型，它是通过随机的各种不同参数产生的，这些参数指定什么时候被用于训练这种树和其它树的参数。装袋是一个“集成”的特殊方法，或多个模型的结果组合在一起。来查看在实践中它是如何运行的，让我们开始创建自己的随机森林吧！

```python
#hide
# pip install —pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn —U
```

### Creating a Random Forest

### 创建随机森林

We can create a random forest just like we created a decision tree, except now, we are also specifying parameters that indicate how many trees should be in the forest, how we should subset the data items (the rows), and how we should subset the fields (the columns).

我们能够像创建一棵决策树那样创建随机森林，可是现在，我们也要具体说明参数，指明在森林中应该有多少树，我们应该如何子集数据项（行数），我们如何子集领域（列数）。

In the following function definition `n_estimators` defines the number of trees we want, `max_samples` defines how many rows to sample for training each tree, and `max_features` defines how many columns to sample at each split point (where `0.5` means "take half the total number of columns"). We can also specify when to stop splitting the tree nodes, effectively limiting the depth of the tree, by including the same `min_samples_leaf` parameter we used in the last section. Finally, we pass `n_jobs=-1` to tell sklearn to use all our CPUs to build the trees in parallel. By creating a little function for this, we can more quickly try different variations in the rest of this chapter:

下述函数描述`n_estimators`定义了我们希望的树的数量，`max_samples`定义了训练每棵树多少行做样本，`max_features`定义了在每个分割点上多少列做样本（其`0.5`表示“取列总数的一半”）。我们也能够说明什么时候停止分割树节点，通过包含在上小节所使用的相同`min_samples_leaf`参数有效的限定树的深度。最后，我们专递`n_jobs=-1`来告诉sklearn使用我们所有的CPU来并行的创建树。通过创建这样一个小函数，在本章节的剩余部分我们能够更快速的尝试不同变化：

```
def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
```

```
m = rf(xs, y);
```

Our validation RMSE is now much improved over our last result produced by the `DecisionTreeRegressor`, which made just one tree using all the available data:

我们验证的RMSE现在是通过`DecisionTreeRegressor`大大改善我们最后产生的结果，其只是用所有的验证数据只做一棵树：

```
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
```

Out: (0.170917, 0.233975)

One of the most important properties of random forests is that they aren't very sensitive to the hyperparameter choices, such as `max_features`. You can set `n_estimators` to as high a number as you have time to train—the more trees you have, the more accurate the model will be. `max_samples` can often be left at its default, unless you have over 200,000 data points, in which case setting it to 200,000 will make it train faster with little impact on accuracy. `max_features=0.5` and `min_samples_leaf=4` both tend to work well, although sklearn's defaults work well too.

随机森林其中一个最重要的属性是它对超参选择不是非常敏感，如`max_features`。你能够设置`n_estimators`你有时间训练的最高值（你有更多的树，模型会有更高的精度）。`max_samples`能够能够设置它的默认值，除非你有超过2000,000个数据点，在一些情况下设置它为200,000会使得它在精度上有很小影响的情况下更快的训练。`max_features=0.5`和`min_samples_leaf=4`两者倾向工作的更加良好，虽然sklearn的默认值工作也是非常良好。

The sklearn docs [show an example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) of the effects of different `max_features` choices, with increasing numbers of trees. In the plot, the blue plot line uses the fewest features and the green line uses the most (it uses all the features). As you can see in <max_features>, the models with the lowest error result from using a subset of features but with a larger number of trees.

sklearn文档展示了增加树的数量不同`max_features`选择所生产影响的[例子](http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)。在图表中，蓝色绘图线使用了最少的特征，绿色线使用的绝大多数特征（它使用了所有特征）。正如你在图<最大特征>中看到的，有最少误差模型结果来自使用特征子集但有最多数量的树。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/sklearn_features.png" alt="sklearn max_features chart" width="500" caption="Error based on max features and number of trees (source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)" id="max_features" />
  </p>
  <p align="center">图：最大特征图</p>
</div>

To see the impact of `n_estimators`, let's get the predictions from each individual tree in our forest (these are in the `estimators_` attribute):

来看一下`n_estimators`的影响，让我们从森林中每一棵独立的树获取预测值（这里在`estimators_`属性中）：

```
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
```

As you can see, `preds.mean(0)` gives the same results as our random forest:

你可以看到，`preds.mean(0)`提供了我们随机森林相同的结果：

```
r_mse(preds.mean(0), valid_y)
```

Out: 0.233975

Let's see what happens to the RMSE as we add more and more trees. As you can see, the improvement levels off quite a bit after around 30 trees:

让我们看一下在我们添加了越来越多的树后对于RMSE会发生什么。你能够看到，大约30棵树后改善程度下降了一些：

```
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);
```

Out: ![img](data:image/png;base64,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)

The performance on our validation set is worse than on our training set. But is that because we're overfitting, or because the validation set covers a different time period, or a bit of both? With the existing information we've seen, we can't tell. However, random forests have a very clever trick called *out-of-bag* (OOB) error that can help us with this (and more!).

在我们验证集上的表现比在训练集上更糟糕。但那是因为我们过拟了，或因为验证集覆盖了不同的时间区间，或两者都有？基于我们已经看到的现存信息，我们不能判断。因此，随机森林有一个很聪明的技巧称为*袋外数据*（OOB）误差，对于这个问题（且更多！）它能够我们。

### Out-of-Bag Error

### 袋外数据误差

Recall that in a random forest, each tree is trained on a different subset of the training data. The OOB error is a way of measuring prediction error on the training set by only including in the calculation of a row's error trees where that row was *not* included in training. This allows us to see whether the model is overfitting, without needing a separate validation set.

回忆一下在随机森林中，每棵树在不同的训练子集数据上训练。OOB误差是一个测量预测误差的方法，其在训练集上只包含一行误差树计算，那一行在训练中不被包含。这允许我们查看模型是否过拟，不需要分割验证集。

> A: My intuition for this is that, since every tree was trained with a different randomly selected subset of rows, out-of-bag error is a little like imagining that every tree therefore also has its own validation set. That validation set is simply the rows that were not selected for that tree's training.

> 亚：我的理解是这样的，因为每棵树用不同的随机选择的行子集来训练的，袋外数据误差是有点像每棵树为此也有它自己的验证集。那个验证集是没有为树训练所选中的简单的行数据。

This is particularly beneficial in cases where we have only a small amount of training data, as it allows us to see whether our model generalizes without removing items to create a validation set. The OOB predictions are available in the `oob_prediction_` attribute. Note that we compare them to the training labels, since this is being calculated on trees using the training set.

这是特别有利的，因为我们只有少量的训练数据，它允许我们查看模型是否泛化，而不用移除数据项来创建一个验证集。OOB预测在`oob_prediction_`属性中可获取的。注意，我们把他们与训练标签做对比，因为这是使用训练集在树上被计算的。

```
r_mse(m.oob_prediction_, y)
```

Out: 0.210681

We can see that our OOB error is much lower than our validation set error. This means that something else is causing that error, in *addition* to normal generalization error. We'll discuss the reasons for this later in this chapter.

This is one way to interpret our model's predictions—let's focus on more of those now.

我们能够看到OOB误差比我们验证集误差还要低。这表示*除了*普通泛化误差，还有其它问题引起了误差。我们在本章节稍晚些时候会讨论原因。

这是一个解释我们模型预测的方法，现在让我们聚焦那些更多的模型解释。

## Model Interpretation

## 模型解释

For tabular data, model interpretation is particularly important. For a given model, the things we are most likely to be interested in are:

对于表格数据，模型解释尤为重要。对于一个给定的模型，我们最可能感兴趣的事情是：

- How confident are we in our predictions using a particular row of data?
- For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
- Which columns are the strongest predictors, which can we ignore?
- Which columns are effectively redundant with each other, for purposes of prediction?
- How do predictions vary, as we vary these columns?

- 我们是如何确信在我们的预测中使用特定数据行？
- 对于使用特定数据行的预测，最重要的因素是什么，它们如何影响预测？
- 哪些列是最强的预测因子，我们能够忽略哪些列？
- 以预测为目的，哪些列实际上是相互冗余的？
- 当我们改变那些列时，预测如果变化的？

As we will see, random forests are particularly well suited to answering these questions. Let's start with the first one!

正如我们将要看到的，随机森林特别适合来回答这些问题。让我们从第一个问题开始！

### Tree Variance for Prediction Confidence

### 树差异预测置信

We saw how the model averages the individual tree's predictions to get an overall prediction—that is, an estimate of the value. But how can we know the confidence of the estimate? One simple way is to use the standard deviation of predictions across the trees, instead of just the mean. This tells us the *relative* confidence of predictions. In general, we would want to be more cautious of using the results for rows where trees give very different results (higher standard deviations), compared to cases where they are more consistent (lower standard deviations).

我们看了模型如果平均单独的树预测来获取一个综合预测，即值的评估。但是我们如何能够知道评估的可信度？一个简单的方法是使用所有树预测的标准偏差，替代只是做平均。这告诉了我们预测可信的*相关性*。通常来说，相比树更一致的结果（低标准偏差），对于那些行数据，树给出差异非常大的结果，我们希望使用这些结果时要更加谨慎（高标准偏差）。

In the earlier section on creating a random forest, we saw how to get predictions over the validation set, using a Python list comprehension to do this for each tree in the forest:

在早先的部分创建一个随机森林，我们看到使用Python列表推导对森林中的每棵如何在验证集上做预测。

```
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
```

```
preds.shape
```

Out: (40, 7988)

Now we have a prediction for every tree and every auction (40 trees and 7,988 auctions) in the validation set.

Using this we can get the standard deviation of the predictions over all the trees, for each auction:

现在在验证集上，我们对每棵树有了一个预测和拍卖（40棵树和7,988拍卖）。

使用这个结果我们能够对每个拍卖记录取所有树的预测标准偏差，

```
preds_std = preds.std(0)
```

Here are the standard deviations for the predictions for the first five auctions—that is, the first five rows of the validation set:

这是对头五个拍卖预测的标准偏差，即：验证集的头五行：

```
preds_std[:5]
```

Out: array([0.25065395, 0.11043862, 0.08242067, 0.26988508, 0.15730173])

As you can see, the confidence in the predictions varies widely. For some auctions, there is a low standard deviation because the trees agree. For others it's higher, as the trees don't agree. This is information that would be useful in a production setting; for instance, if you were using this model to decide what items to bid on at auction, a low-confidence prediction might cause you to look more carefully at an item before you made a bid.

你能够看到，对于预测的可信性差异很大。对于一些拍卖，由于树的一致性有低的标准偏差。对于其它那些树不一致的时候标准差是高的。这些信息在生产环境中很有用。如何，如果你正使用这个模型来决策拍卖中出价的是什么物品，一个低可信预测可能导致你在出价前更加仔细的看该物品。

