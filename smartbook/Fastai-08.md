# Collaborative Filtering Deep Dive

# 协同过滤深度研究

One very common problem to solve is when you have a number of users and a number of products, and you want to recommend which products are most likely to be useful for which users. There are many variations of this: for example, recommending movies (such as on Netflix), figuring out what to highlight for a user on a home page, deciding what stories to show in a social media feed, and so forth. There is a general solution to this problem, called *collaborative filtering*, which works like this: look at what products the current user has used or liked, find other users that have used or liked similar products, and then recommend other products that those users have used or liked.

一个要解决的常见问题是，当你有一定数量的用户和一定数量的产品，你希望推荐哪些产品最可能对哪些用户有用。对于这个问题有很多变化：例如，推荐电影（如网飞），计算出用户的网页上要高亮显示哪些内容，决定社交媒体中显示哪些故事，等等。对于这个问题这有一个常用解决方案称为*协同过滤*，它的工作方式：看当前用户已经使用或喜爱了哪些产品，寻找其它用户已经使用或喜爱的类似产品，然后推荐其它那些用户已经使用或喜爱的产品。

For example, on Netflix you may have watched lots of movies that are science fiction, full of action, and were made in the 1970s. Netflix may not know these particular properties of the films you have watched, but it will be able to see that other people that have watched the same movies that you watched also tended to watch other movies that are science fiction, full of action, and were made in the 1970s. In other words, to use this approach we don't necessarily need to know anything about the movies, except who like to watch them.

例如，在网飞上你可能已经看了很多充满动作的科幻电影，这些电影在1970年代制作的。网飞可能不知道这些你看过的电影的特定属性，但是它将能够查看和你观看相同电影的其它人，倾向看其它的充满动作的在1970年代制作的科幻电影。换句话说，用这个方法除了人们喜欢观看这些电影，我们不需要知道关于电影的任何信息，

There is actually a more general class of problems that this approach can solve, not necessarily involving users and products. Indeed, for collaborative filtering we more commonly refer to *items*, rather than *products*. Items could be links that people click on, diagnoses that are selected for patients, and so forth.

事实上有一些更一般问题类型用这个方法能够解决，不必包含用户和产品。确实，对于协同过滤我们更常指的是*项目*，而不是*产品*。项目能够关联人们的点击，对病人选择的诊断，等等。

The key foundational idea is that of *latent factors*. In the Netflix example, we started with the assumption that you like old, action-packed sci-fi movies. But you never actually told Netflix that you like these kinds of movies. And Netflix never actually needed to add columns to its movies table saying which movies are of these types. Still, there must be some underlying concept of sci-fi, action, and movie age, and these concepts must be relevant for at least some people's movie watching decisions.

关键的基础想法是那些潜在因素。在网飞的例子中，我们从假设你喜欢老的科幻动作电影开始。但你实际上永远不会告诉网飞你喜欢这些类型的电影。并且网飞实际上永远不需要在它的电影表中添加一列，来描述这些电影是这些类型。然而，必然有一些潜在的科幻、运作和电影年代的概念，这些概念一定至少与一些人的观看决策是关联的。

For this chapter we are going to work on this movie recommendation problem. We'll start by getting some data suitable for a collaborative filtering model.

这一章，我们将处理这个电影推荐问题。我们会从通过取得一些对于协同过滤模型合适的数据开始。

## A First Look at the Data

## 首先查看数据

We do not have access to Netflix's entire dataset of movie watching history, but there is a great dataset that we can use, called [MovieLens](https://grouplens.org/datasets/movielens/). This dataset contains tens of millions of movie rankings (a combination of a movie ID, a user ID, and a numeric rating), although we will just use a subset of 100,000 of them for our example. If you're interested, it would be a great learning project to try and replicate this approach on the full 25-million recommendation dataset, which you can get from their website.

The dataset is available through the usual fastai function:

我们没有网飞电影观看历史的整个数据集权限，但有一个很好的数据集我们能够使用，称为[MovieLens](https://grouplens.org/datasets/movielens/)。这个数据集包含了数千万的电影排名（一个电影ID,一个用户ID和一个数字等级的组合），然而对于我们的事例我们只会用到这个数据集中的十万条数据的子集。如果你有兴趣，在完整的两千五百万条数据的推荐数据集上，这会是一个非常好的学习项目来尝试和复制这个方法，你能够从它们的网站上获取。

通过使用fastai函数可获取到这个数据集：

```
from fastai.collab import *
from fastai.tabular.all import *
path = untar_data(URLs.ML_100k)
```

According to the *README*, the main table is in the file *u.data*. It is tab-separated and the columns are, respectively user, movie, rating, and timestamp. Since those names are not encoded, we need to indicate them when reading the file with Pandas. Here is a way to open this table and take a look:

根据*说明*，主表是在文件*u.data*中。它是跳格分割，列分别是用户、电影、等级和时间戳。因为那些名子不是编码过的，当用Pandas读取文件时，我们需要标示它们。下面是打开这个表并查询的方法：

```
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()
```

|      | user用户 | movie电影 | rating等级 | timestamp时间戳 |
| ---: | -------: | --------: | ---------: | --------------: |
|    0 |      196 |       242 |          3 |       881250949 |
|    1 |      186 |       302 |          3 |       891717742 |
|    2 |       22 |       377 |          1 |       878887116 |
|    3 |      244 |        51 |          2 |       880606923 |
|    4 |      166 |       346 |          1 |       886397596 |

Although this has all the information we need, it is not a particularly helpful way for humans to look at this data. <movie_xtab> shows the same data cross-tabulated into a human-friendly table.

虽然有我们需要所有信息，对于人类看这些数据它不是特别有帮助的方法。图<电影和用户的交叉表>显示了同样的数据，交叉表对人类很友好的表格。

<div style="text-align:center">
  <p align="center">
    <img alt="Crosstab of movies and users" width="632" caption="Crosstab of movies and users" id="movie_xtab" src="./_v_images/att_00040.png"/>
  </p>
  <p align="center">图：电影和用户的交叉表</p>
</div>

We have selected just a few of the most popular movies, and users who watch the most movies, for this crosstab example. The empty cells in this table are the things that we would like our model to learn to fill in. Those are the places where a user has not reviewed the movie yet, presumably because they have not watched it. For each user, we would like to figure out which of those movies they might be most likely to enjoy.

这个交叉表例子中，我们选择了一些只是最流行的电影和用户观看最多的电影。在这个表中空的单元格是我们希望模型来学习填充的内容。那些地方用户还没有评论的地方，大概因为他们还没有观看。对于各个用户，我们希望计算出那些电影中的哪些他们可能最喜欢。

If we knew for each user to what degree they liked each important category that a movie might fall into, such as genre, age, preferred directors and actors, and so forth, and we knew the same information about each movie, then a simple way to fill in this table would be to multiply this information together for each movie and use a combination. For instance, assuming these factors range between -1 and +1, with positive numbers indicating stronger matches and negative numbers weaker ones, and the categories are science-fiction, action, and old movies, then we could represent the movie *The Last Skywalker* as:

如果我们知道每一名用户他们喜欢电影可能属于每个重要分类的程度，如流派、年代、偏好的导演和演员，等等，我们知道关于每部电影同样的信息，然后一个简单的填写这个表的方法也许是对于每部电影把这个信息乘起来并组合使用。例如，假设这些因素的范围在-1到+1之间，正数指标更强烈匹配，负数指标更弱的匹配，这些分类是科幻、动作和老电影，然后我们能够描述*最后的天行者*这部电影为：

```
last_skywalker = np.array([0.98,0.9,-0.9])
```

Here, for instance, we are scoring *very science-fiction* as 0.98, and *very not old* as -0.9. We could represent a user who likes modern sci-fi action movies as:

例如，在这里我们非常科幻的分类为0.98，不是非常老的电影为-0.9。我们能够描述喜欢现代科幻电影的用户为：

```
user1 = np.array([0.9,0.8,-0.6])
```

and we can now calculate the match between this combination:

现在我们能够计算这些组合之间的匹配度：

```
(user1*last_skywalker).sum()
```

Out: 2.1420000000000003

When we multiply two vectors together and add up the results, this is known as the *dot product*. It is used a lot in machine learning, and forms the basis of matrix multiplication. We will be looking a lot more at matrix multiplication and dot products in <chapter_foundations>.

> jargon: dot product: The mathematical operation of multiplying the elements of two vectors together, and then summing up the result.

On the other hand, we might represent the movie *Casablanca* as:

当我们把两个向量乘起来并加总结果，这被称为*点积*。在机器学习中它是被大量使用的，并构成矩阵乘法的基础。在<章节：基础>中我们会大量看到矩阵乘法和点积。

> 术语：点积：两个向量元素乘法的数学运算，然后把结果加总。

```
casablanca = np.array([-0.99,-0.3,0.8])
```

The match between this combination is:

这个组合的匹配是：

```
(user1*casablanca).sum()
```

Out: -1.611

Since we don't know what the latent factors actually are, and we don't know how to score them for each user and movie, we should learn them.

因为我们不知道实际的潜在因素是什么，且我们不知道对每名用户和电影如何来打分，我们应该学习他们。

## Learning the Latent Factors

## 学习潜在因素

There is surprisingly little difference between specifying the structure of a model, as we did in the last section, and learning one, since we can just use our general gradient descent approach.

正如我们在上一节做的那样，指定模型结构和学习模型之间只有惊人的细微差别，因此我们正好能够使用常用的梯度下降方法。

Step 1 of this approach is to randomly initialize some parameters. These parameters will be a set of latent factors for each user and movie. We will have to decide how many to use. We will discuss how to select this shortly, but for illustrative purposes let's use 5 for now. Because each user will have a set of these factors and each movie will have a set of these factors, we can show these randomly initialized values right next to the users and movies in our crosstab, and we can then fill in the dot products for each of these combinations in the middle. For example, <xtab_latent> shows what it looks like in Microsoft Excel, with the top-left cell formula displayed as an example.

这一方法的第一步，随机初始化一些参数。这些参数会作为每一用户和电影的一系列潜在因素。我们必须要决定使用多少个潜在因素。我们将会简短的讨论如何去选择，但是为了说明性的目标，让我们现在使用5个潜在因素。因为每个用户和每部电影都会有一系列的这些因素，在我们的交叉表中，我们能够在用户和电影旁边展示这些随机的初始化值，并且然后我们在交叉表中央能够对这些组合的每一个单元格填写点积值。例如，在图<交叉表的潜在因素>中展示了在微软电子表格中这个交叉表的样子，作为事例展示了左上单元格公式。

<div style="text-align:center">
  <p align="center">
    <img alt="Latent factors with crosstab" width="900" caption="Latent factors with crosstab" id="xtab_latent" src="./_v_images/att_00041.png"/>
  </p>
  <p align="center">图：交叉表的潜在因素</p>
</div>

Step 2 of this approach is to calculate our predictions. As we've discussed, we can do this by simply taking the dot product of each movie with each user. If, for instance, the first latent user factor represents how much the user likes action movies and the first latent movie factor represents if the movie has a lot of action or not, the product of those will be particularly high if either the user likes action movies and the movie has a lot of action in it or the user doesn't like action movies and the movie doesn't have any action in it. On the other hand, if we have a mismatch (a user loves action movies but the movie isn't an action film, or the user doesn't like action movies and it is one), the product will be very low.

本方法的第二步，来计算我们的预测。正如我们讨论过的，我们能够通过简单的求每个用户和每部电影的点积来实现。例如，如果第一个潜在用户因素代表多少用户喜欢运作电影，及第一个潜在电影因素代表这部电影是否有很多运作或没有，如果用户要么喜欢运作电影和电影中有很多动作，要么用户不喜欢运作电影及电影中没有任何运作情节，那些乘积会特别高。换句话说，如果我们有了错误的匹配（一名用户喜爱运作电影，但是这部电影不是一部运作片，或用户不喜欢运作电影及这部电影确是一部运作片），乘积会非常的低。

Step 3 is to calculate our loss. We can use any loss function that we wish; let's pick mean squared error for now, since that is one reasonable way to represent the accuracy of a prediction.

第三步是计算我们的损失。我们能够使用任意我们所希望的损失函数。现在让我们选择均方差，因为这是代表预测精度的一个合理方法。

That's all we need. With this in place, we can optimize our parameters (that is, the latent factors) using stochastic gradient descent, such as to minimize the loss. At each step, the stochastic gradient descent optimizer will calculate the match between each movie and each user using the dot product, and will compare it to the actual rating that each user gave to each movie. It will then calculate the derivative of this value and will step the weights by multiplying this by the learning rate. After doing this lots of times, the loss will get better and better, and the recommendations will also get better and better.

上述所有内容是我们所需要的。有了这些，使用随机梯度下降我们就能够优化我们的参数了（即，潜在因素），例如最小化损失。在每一步，随机梯度下降优化器会使用点积计算每部电影和每名用户的匹配度，且其会与每名用户给出的每部电影的真实等级做比对。它然后会计算这个值的导数和通过乘以学习率来步进权重。经过做这个工作很多次后，损失会变的越来越好，推荐也将会变的越来越好。

To use the usual `Learner.fit` function we will need to get our data into a `DataLoaders`, so let's focus on that now.

使用普通的`Learner.fit`函数，我们需要把我们的数据放入一个`DataLoaders`，所以现在让我们聚焦在这个事情上。

## Creating the DataLoaders

## 创建DataLoaders

When showing the data, we would rather see movie titles than their IDs. The table `u.item` contains the correspondence of IDs to titles:

当展示数据时，我们宁愿看到电影标题而不是他们的ID。表`u.item`包含ID与电影标题的对应关系：

```
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()
```

|      | movie电影 |     title电影标题 |
| ---: | --------: | ----------------: |
|    0 |         1 |  Toy Story (1995) |
|    1 |         2 |  GoldenEye (1995) |
|    2 |         3 | Four Rooms (1995) |
|    3 |         4 | Get Shorty (1995) |
|    4 |         5 |    Copycat (1995) |

We can merge this with our `ratings` table to get the user ratings by title:

我们能够把这个表与我们的`等级`表合并，通过电影标题来获取用户等级：

```
ratings = ratings.merge(movies)
ratings.head()
```

|      | user用户 | movie电影 | rating等级 | timestamp时间戳 | title电影标题 |
| ---: | -------: | --------: | ---------: | --------------: | ------------: |
|    0 |      196 |       242 |          3 |       881250949 |  Kolya (1996) |
|    1 |       63 |       242 |          3 |       875747190 |  Kolya (1996) |
|    2 |      226 |       242 |          5 |       883888671 |  Kolya (1996) |
|    3 |      154 |       242 |          3 |       879138235 |  Kolya (1996) |
|    4 |      306 |       242 |          5 |       876503793 |  Kolya (1996) |

We can then build a `DataLoaders` object from this table. By default, it takes the first column for the user, the second column for the item (here our movies), and the third column for the ratings. We need to change the value of `item_name` in our case to use the titles instead of the IDs:

然后我们能够利用这个表创建一个`DataLoaders`对象。它默认取第一列为用户，第二列为项目（在这里是我们的电影），及第三列为电影等级。在我们的例子中我们需要改变`item_name`的值，用电影标题替代电影的ID：

```
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
```

|      | user用户 |                     title电影标题 | rating等级 |
| ---: | -------: | --------------------------------: | ---------: |
|    0 |      542 |               My Left Foot (1989) |          4 |
|    1 |      422 |              Event Horizon (1997) |          3 |
|    2 |      311 |         African Queen, The (1951) |          4 |
|    3 |      595 |                   Face/Off (1997) |          4 |
|    4 |      617 |               Evil Dead II (1987) |          1 |
|    5 |      158 |              Jurassic Park (1993) |          5 |
|    6 |      836 |                Chasing Amy (1997) |          3 |
|    7 |      474 |                       Emma (1996) |          3 |
|    8 |      466 | Jackie Chan's First Strike (1996) |          3 |
|    9 |      554 |                     Scream (1996) |          3 |

To represent collaborative filtering in PyTorch we can't just use the crosstab representation directly, especially if we want it to fit into our deep learning framework. We can represent our movie and user latent factor tables as simple matrices:

在PyTorch中表示协同过滤，我们不能只是使用交叉表直接的代表，尤其如果我们想让它适合我们的深度学习框架的时候。我们能够描述电影和用户潜在因素表为简单的矩阵：

```
dls.classes
```

Out: {'user': (#944) ['#na#',1,2,3,4,5,6,7,8,9...],
 'title': (#1635) ['#na#',"'Til There Was You (1997)",'1-900 (1994)','101 Dalmatians (1996)','12 Angry Men (1957)','187 (1997)','2 Days in the Valley (1996)','20,000 Leagues Under the Sea (1954)','2001: A Space Odyssey (1968)','3 Ninjas: High Noon At Mega Mountain (1998)'...]}

```
n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)
```

To calculate the result for a particular movie and user combination, we have to look up the index of the movie in our movie latent factor matrix and the index of the user in our user latent factor matrix; then we can do our dot product between the two latent factor vectors. But *look up in an index* is not an operation our deep learning models know how to do. They know how to do matrix products, and activation functions.

对特定电影和用户组合来计算结果，我们必须在我们的电影潜在因素矩阵中查找电影的索引和在我们的用户潜在因素矩阵中找用户的索引。然后，我们能够做两个潜在因素向量的点积。但是*在一个索引中查找* 不是我们深度学习模型知道如何去做的一个操作。它们知道如果做矩阵乘积和激活函数。

Fortunately, it turns out that we can represent *look up in an index* as a matrix product. The trick is to replace our indices with one-hot-encoded vectors. Here is an example of what happens if we multiply a vector by a one-hot-encoded vector representing the index 3:

幸运的是，事实证明*在一个索引中的查找* 我们能够等同于为一个矩阵乘积。技巧是用独热编码替换我们的索引。这是一个例子，如果我们把一个向量乘以等同于索引3的一个独热编码向量，看会发生什么：

```
one_hot_3 = one_hot(3, n_users).float()
```

```
user_factors.t() @ one_hot_3
```

Out: tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])

It gives us the same vector as the one at index 3 in the matrix:

它提供给了我们在矩阵中索引3处同样的向量值：

```
user_factors[3]
```

Out: tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])

If we do that for a few indices at once, we will have a matrix of one-hot-encoded vectors, and that operation will be a matrix multiplication! This would be a perfectly acceptable way to build models using this kind of architecture, except that it would use a lot more memory and time than necessary. We know that there is no real underlying reason to store the one-hot-encoded vector, or to search through it to find the occurrence of the number one—we should just be able to index into an array directly with an integer. Therefore, most deep learning libraries, including PyTorch, include a special layer that does just this; it indexes into a vector using an integer, but has its derivative calculated in such a way that it is identical to what it would have been if it had done a matrix multiplication with a one-hot-encoded vector. This is called an *embedding*.

如果我们对几个索引一次性做这个操作，我们要有一个独热编码向量矩阵，这个操作会是一个矩阵乘法！使用这个类型架构，这也会是一个构建模型的可接收的完美方法，除了它会使用比必要条件下更多的内存和时间。我们知道没有真正的根本原因来存储独热编码，或通过它搜寻到存在的那个数字。我们应该只是用一个整数就能够直接的索引到一个数组中。因而，绝大多数深度学习库（包括PyTorch），包含只做这个事情的一个特定层。它用一个整数索引到向量中，但有它的导数计算，用这种方法是与如果本来用一个独热编码乘以一个矩阵是相同的。这被称为*嵌入*。

> jargon: Embedding: Multiplying by a one-hot-encoded matrix, using the computational shortcut that it can be implemented by simply indexing directly. This is quite a fancy word for a very simple concept. The thing that you multiply the one-hot-encoded matrix by (or, using the computational shortcut, index into directly) is called the *embedding matrix*.

> 术语：嵌入（Embedding）：乘以一个独热编码矩阵，使用简单的直接索引它能够实现计算的捷径。对于一个非常简单的概念这是一个十分奇幻的词。你将独热编码矩阵所乘以的矩阵（或使用计算捷径直接索引到）被称为*嵌入矩阵*。

In computer vision, we have a very easy way to get all the information of a pixel through its RGB values: each pixel in a colored image is represented by three numbers. Those three numbers give us the redness, the greenness and the blueness, which is enough to get our model to work afterward.

在计算机视觉中我们有一个非常容易的方法，通过它的RGB值来获取一个像素的所有信息：在一个彩色图像中每个像素是通过三个数值代表的。那三个数值给了我们红、绿和蓝，它足以让我们的模型做后续的工作。

For the problem at hand, we don't have the same easy way to characterize a user or a movie. There are probably relations with genres: if a given user likes romance, they are likely to give higher scores to romance movies. Other factors might be whether the movie is more action-oriented versus heavy on dialogue, or the presence of a specific actor that a user might particularly like.

对于手边的这个问题，我们没有相同的方法来描述一名用户或一部电影。有可能与流派有关：如果一个给定的用户喜欢罗曼蒂克，他们可能给了罗曼蒂克电影更高的分数。其它因素是否可能电影相对于对话更强调运作，或一名用户可能特别喜欢一个特定演员的仪态。

How do we determine numbers to characterize those? The answer is, we don't. We will let our model *learn* them. By analyzing the existing relations between users and movies, our model can figure out itself the features that seem important or not.

我们如何确定那些特征的数值？答案是，我们不做。我们会让我们的模型*学习* 它们。通过分析用户和电影之产已经存在的关系，我们的模型能够自己计算出那些特征似乎重要或不重要。

This is what embeddings are. We will attribute to each of our users and each of our movies a random vector of a certain length (here, `n_factors=5`), and we will make those learnable parameters. That means that at each step, when we compute the loss by comparing our predictions to our targets, we will compute the gradients of the loss with respect to those embedding vectors and update them with the rules of SGD (or another optimizer).

这就是嵌入。我们将分配给给每名用户和每部电影一个确定长度的随机张量（在这里n_factors = 5），且我们会使用那些参数可学习。这表示在每一步，当我们通过比对预测和目标来计算损失的时候，我们会用针对那些嵌入向量计算损失的梯度，并用随机梯度下降（或其它优化器）的规则更新它们。

At the beginning, those numbers don't mean anything since we have chosen them randomly, but by the end of training, they will. By learning on existing data about the relations between users and movies, without having any other information, we will see that they still get some important features, and can isolate blockbusters from independent cinema, action movies from romance, and so on.

在一开始，那些数字没有任何含义，因为我们随机选择的它们，但是在训练的结尾，它们就会有含义了。通过在已存在的数据上学习关于用户和电影之间的关系，不包含任何其它信息，我们会看到它们依然取得了一些重要特征，并能够从独立的影院把大片，罗曼蒂克把动作电影隔离开，等等。

We are now in a position that we can create our whole model from scratch.

现在我们处于能够从零开始创建我们整个模型的程度了。

## Collaborative Filtering from Scratch

## 从零开始创建协同过滤

Before we can write a model in PyTorch, we first need to learn the basics of object-oriented programming and Python. If you haven't done any object-oriented programming before, we will give you a quick introduction here, but we would recommend looking up a tutorial and getting some practice before moving on.

The key idea in object-oriented programming is the *class*. We have been using classes throughout this book, such as `DataLoader`, `string`, and `Learner`. Python also makes it easy for us to create new classes. Here is an example of a simple class:

我们在PyTorch中能够编写一个模型之间，我们首先需要学习面向对象编程和Python的基础知识。如果你之间没有做过任何面向对象编程，在这里我们会给你一个简短的介绍，但是我们推荐在继续之前找到一个教程并做一些练习。

在面向对象编程中一个关键思想是*类*。我们已经在整本书都使用了类，如`DataLoader`、`string`、和`Learner`。Python也使得我们很容易来创建一个新的类，这是一个简单的类事例：

```
class Example:
    def __init__(self, a): self.a = a
    def say(self,x): return f'Hello {self.a}, {x}.'
```

The most important piece of this is the special method called `__init__` (pronounced *dunder init*). In Python, any method surrounded in double underscores like this is considered special. It indicates that there is some extra behavior associated with this method name. In the case of `__init__`, this is the method Python will call when your new object is created. So, this is where you can set up any state that needs to be initialized upon object creation. Any parameters included when the user constructs an instance of your class will be passed to the `__init__` method as parameters. Note that the first parameter to any method defined inside a class is `self`, so you can use this to set and get any attributes that you will need:

这个事例最重要的部分是一个特定方法称为`__init__`（发音为*dunder init*）。在Python中，任何像现在这个方法一样被双下划线包围的方法被认为是特殊的。它表明有一个扩展行为与这个方法名有关联。如果是`__init__`，当你新的对象被创建时，Python会调用这个方法。所以在这个地方你能设置对象创建时需要初始化的任何声明。任何参数，包括当用户构建你的类的实例时，都会作为参数传递给`__init__`方法。注意，在类内部定义的任何方法的第一个参数是`self`，所以你能够使用它来设置和获取任何你将需要的属性：

```
ex = Example('Sylvain')
ex.say('nice to meet you')
```

Out: 'Hello Sylvain, nice to meet you.'

Also note that creating a new PyTorch module requires inheriting from `Module`. *Inheritance* is an important object-oriented concept that we will not discuss in detail here—in short, it means that we can add additional behavior to an existing class. PyTorch already provides a `Module` class, which provides some basic foundations that we want to build on. So, we add the name of this *superclass* after the name of the class that we are defining, as shown in the following example.

也要注意，创建一个新的PyTorch模型需要从`Module`上继承。`继承`（inheritance）是一个重要的面向对象概念，我们不会在这里讨论这个概念的细节。简短来说，意思是我们能够添加附加行为给一个存在的类。PyTorch已经提供了一个`Module`类，它提供了一些我们希望建立的基础。所以正如下面的例子中所展示的，我们在定义的类名后添加了这个*超类* 名。

The final thing that you need to know to create a new PyTorch module is that when your module is called, PyTorch will call a method in your class called `forward`, and will pass along to that any parameters that are included in the call. Here is the class defining our dot product model:

创建一个新的PyTorch模型你需要知道的最后事情，是你的模型被调用的时候，PyTorch会调用一个在你的类中称为`forward`的方法，且会传递调用中包含的任何参数。下面是我们定义的点积模型类：

```
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
```

If you haven't seen object-oriented programming before, then don't worry, you won't need to use it much in this book. We are just mentioning this approach here, because most online tutorials and documentation will use the object-oriented syntax.

Note that the input of the model is a tensor of shape `batch_size x 2`, where the first column (`x[:, 0]`) contains the user IDs and the second column (`x[:, 1]`) contains the movie IDs. As explained before, we use the *embedding* layers to represent our matrices of user and movie latent factors:

如果你之前没有看过面向对象编程，不要着急，在本书中你不需要太多的使用它。我们只是在这里提到这一方法，因为大多在线教程和文档会使用面向对象语法。

注意，模型的输入是一个形状`batch_size x 2`的张量，第一列（`x[:, 0]`）包含用户的ID，第二列（`x[:, 1]`）包含电影ID，正如之前所解释的那样，我们使用*嵌入*层来表示我们的用户和电影潜在因素矩阵：

```
x,y = dls.one_batch()
x.shape
```

Out: torch.Size([64, 2])

Now that we have defined our architecture, and created our parameter matrices, we need to create a `Learner` to optimize our model. In the past we have used special functions, such as `cnn_learner`, which set up everything for us for a particular application. Since we are doing things from scratch here, we will use the plain `Learner` class:

由于我们已经定义了我们的架构，且创建了我们的参数矩阵，我们需要创建一个`Learner`来优化我们的模型。在过去我们使用了专门的函数，如`cnn_learner`，对一个特定应用它为我们设置了所有内容。因为现在我们正在从零开始做，我们会使用简朴的`Learner`类：

```
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
```

We are now ready to fit our model:

现在我们准备来拟合我们的模型：

```
learn.fit_one_cycle(5, 5e-3)
```

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.993168 |   0.990168 | 00:12 |
|     1 |   0.884821 |   0.911269 | 00:12 |
|     2 |   0.671865 |   0.875679 | 00:12 |
|     3 |   0.471727 |   0.878200 | 00:11 |
|     4 |   0.361314 |   0.884209 | 00:12 |

The first thing we can do to make this model a little bit better is to force those predictions to be between 0 and 5. For this, we just need to use `sigmoid_range`, like in <chapter_multicat>. One thing we discovered empirically is that it's better to have the range go a little bit over 5, so we use `(0, 5.5)`:

我们能够做的使模型更好一点的事情是强制那些预测在0和5之间。为此，我们只需要使用如<章节：多分类>中的`sigmoid_range`。一件我们根据经验发现的事情是最好范围稍微超过5，所以我们使用`（0，5.5）`：

```
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)
```

```
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.973745 |   0.993206 | 00:12 |
|     1 |   0.869132 |   0.914323 | 00:12 |
|     2 |   0.676553 |   0.870192 | 00:12 |
|     3 |   0.485377 |   0.873865 | 00:12 |
|     4 |   0.377866 |   0.877610 | 00:11 |

This is a reasonable start, but we can do better. One obvious missing piece is that some users are just more positive or negative in their recommendations than others, and some movies are just plain better or worse than others. But in our dot product representation we do not have any way to encode either of these things. If all you can say about a movie is, for instance, that it is very sci-fi, very action-oriented, and very not old, then you don't really have any way to say whether most people like it.

That's because at this point we only have weights; we do not have biases. If we have a single number for each user that we can add to our scores, and ditto for each movie, that will handle this missing piece very nicely. So first of all, let's adjust our model architecture:

这是一个尚的开始，但是我们能够做的更好。一个明显缺失的部分是一些用户在推荐中只是比其它人更正面或负面，一些电影只是单纯比其它电影更好或更糟。但是在我们的点积中描述的，我们没有任何方法来对这些内容的任何一个编码。例如，如果我们所能够描述只是它非常科幻，非常注重动作和不是非常的老，然后我们真的没有任何办法来说大多数人是否喜欢它。

那是因为在这一点上我们只有权重，没有偏差。如果我们对每名用户有一个单一数值，我们能够添加我们的分数，并为每部添加同样的分数。这会很好的处理这个缺失部分。所以首先，让我们调整我们的模型架构：

```
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)
```

Let's try training this and see how it goes:

让我们尝试训练并查看它的效果怎么样：

```
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.929161 |   0.936303 | 00:13 |
|     1 |   0.820444 |   0.861306 | 00:13 |
|     2 |   0.621612 |   0.865306 | 00:14 |
|     3 |   0.404648 |   0.886448 | 00:13 |
|     4 |   0.292948 |   0.892580 | 00:13 |

Instead of being better, it ends up being worse (at least at the end of training). Why is that? If we look at both trainings carefully, we can see the validation loss stopped improving in the middle and started to get worse. As we've seen, this is a clear indication of overfitting. In this case, there is no way to use data augmentation, so we will have to use another regularization technique. One approach that can be helpful is *weight decay*.

最终变的更糟，而不是更好（在训练的结尾至少是这样的）。这是为什么？如果我们自己的查看两者的训练，我们能够看到在中间验证损失停止改善并开始变糟。正如我们见过的，这很清晰的表明过拟了。在这种情况下，没有办法使用数据增强，所以我们必须使用另一个正则化技术。一个能够有帮助的方法是*权重衰减*

### Weight Decay

### 权重衰减

Weight decay, or *L2 regularization*, consists in adding to your loss function the sum of all the weights squared. Why do that? Because when we compute the gradients, it will add a contribution to them that will encourage the weights to be as small as possible.

Why would it prevent overfitting? The idea is that the larger the coefficients are, the sharper canyons we will have in the loss function. If we take the basic example of a parabola, `y = a * (x**2)`, the larger `a` is, the more *narrow* the parabola is (<parabolas>).

权重衰减或*L2正则*，由附加到你的损失函数上的所有权重平方的总和组成。为什么这样做呢？因为当我们计算梯度时，会给他们添加一个贡献，鼓励权重变的尽可能的小。

为什么它会阻止过拟？思想是更大的系数，在损失函数中我们会有更陡峭的峡谷。如果我们取一个抛物线基本事例，`y = a * (x**2)`，`a` 越大，抛物线就越陡峭（如下图<抛物线>）。

```
#hide_input
#id parabolas
x = np.linspace(-2,2,100)
a_s = [1,2,5,10,50] 
ys = [a * x**2 for a in a_s]
_,ax = plt.subplots(figsize=(8,6))
for a,y in zip(a_s,ys): ax.plot(x,y, label=f'a={a}')
ax.set_ylim([0,5])
ax.legend();
```

Out: ![img](data:image/png;base64,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)

So, letting our model learn high parameters might cause it to fit all the data points in the training set with an overcomplex function that has very sharp changes, which will lead to overfitting.

Limiting our weights from growing too much is going to hinder the training of the model, but it will yield a state where it generalizes better. Going back to the theory briefly, weight decay (or just `wd`) is a parameter that controls that sum of squares we add to our loss (assuming `parameters` is a tensor of all parameters):

所以，让我们的模型学习大的参数可能会导致用一个超级增长函数在训练集上集合所有的数据点有非常剧烈的变化，会导致过拟。

限定我们的权重增长太多是将会阻碍模型的训练，但是它会产出一个更好泛化的情况。返回到简洁的理论，权重衰减（或只是`wd`）是一个控制我们增加到损失中的平均和参数，（假定`参数`是一个所有参数的张量）：

```python
loss_with_wd = loss + wd * (parameters**2).sum()
```

In practice, though, it would be very inefficient (and maybe numerically unstable) to compute that big sum and add it to the loss. If you remember a little bit of high school math, you might recall that the derivative of `p**2` with respect to `p` is `2*p`, so adding that big sum to our loss is exactly the same as doing:

然而，在实践中我们会非常无效率的（且可能数值上不稳定）来计算那个大的合计并添加它到损失中。如果你记得一点高中数学，你可能回想到关于`p`的`p**2` 的导数是 `2*p`，因此，添加大的合计到我们的损失与我们要做的是完全相同的：

```python
parameters.grad += wd * 2 * parameters
```

In practice, since `wd` is a parameter that we choose, we can just make it twice as big, so we don't even need the `*2` in this equation. To use weight decay in fastai, just pass `wd` in your call to `fit` or `fit_one_cycle`:

在实践中，因为`wb`是一个我们选择的参数，我们能够只使得它两倍大，因此在这个等式中我们不需要`*2`。在fastai中使用重要衰减，只需要在你的调用中传递`wb`来`fit`或`fit_one_cycle`：

```
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```

| epoch | train_loss | valid_loss |  time |
| ----: | ---------: | ---------: | ----: |
|     0 |   0.972090 |   0.962366 | 00:13 |
|     1 |   0.875591 |   0.885106 | 00:13 |
|     2 |   0.723798 |   0.839880 | 00:13 |
|     3 |   0.586002 |   0.823225 | 00:13 |
|     4 |   0.490980 |   0.823060 | 00:13 |

Much better!

好多了！