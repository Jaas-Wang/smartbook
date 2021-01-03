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

> 术语：嵌入：乘以一个独热编码矩阵，使用简单的直接索引它能够实现计算的捷径。对于非常简单的概念这是一个十分奇幻的词。你将独热编码矩阵所乘以的矩阵（或使用计算捷径直接索引到）被称为*嵌入矩阵*。

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

现在我们处于能够从零开始创建我们整个模型的位置了。