In [ ]:

```
#hide
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

In [ ]:

```
#hide
from fastbook import *
from fastai.vision.widgets import *
```

## From Model to Production

## **从模型到产品化**

The six lines of code we saw in <> are just one small part of the process of using deep learning in practice. In this chapter, we're going to use a computer vision example to look at the end-to-end process of creating a deep learning application. More specifically, we're going to build a bear classifier! In the process, we'll discuss the capabilities and constraints of deep learning, explore how to create datasets, look at possible gotchas when using deep learning in practice, and more. Many of the key points will apply equally well to other deep learning problems, such as those in <>. If you work through a problem similar in key respects to our example problems, we expect you to get excellent results with little code, quickly.

Let's start with how you should frame your problem.

在对于深度学习的使用中，上面这6行代码只是一小部分。在这个章节，我们将通过一个计算机视觉的案例来研究如何搭建一个深度学习的应用。更确切的说，我们要建造一个“熊熊识别器”！在这个过程中，我们将探讨深度学习的能力和局限，以及其他更多方面。这其中许多重要的观点都同样能很好的应用在其他深度学习的问题中，就好比<>中的那些。如果你正遇到和我们相似的状况，我们希望你能够用很少的代码快速得到很好的结果。

让我们从你该如何一步一步解决你的问题开始。

### The Practice of Deep Learning

### 深度学习的使用之道

We've seen that deep learning can solve a lot of challenging problems quickly and with little code. As a beginner, there's a sweet spot of problems that are similar enough to our example problems that you can very quickly get extremely useful results. However, deep learning isn't magic! The same 6 lines of code won't work for every problem anyone can think of today. Underestimating the constraints and overestimating the capabilities of deep learning may lead to frustratingly poor results, at least until you gain some experience and can solve the problems that arise. Conversely, overestimating the constraints and underestimating the capabilities of deep learning may mean you do not attempt a solvable problem because you talk yourself out of it.

我们已经知道深度学习能让我们用很少的代码快速解决一系列难题。包括我们的案例，这些难题都有这个相似突破点，作为新手的你可以利用它快速得到一个有用的结果。但是，机器学习不是万能的！诸如这6行代码不可能解决如今所有问题。至少在你有一些经验或者能够解决一些问题之前，都不要低估机器学习的局限或是高估它的能力，不然你可能会得到追悔莫及的糟糕结果。不过上述想法也可能只是你拒绝解决问题的借口。

 

We often talk to people who underestimate both the constraints and the capabilities of deep learning. Both of these can be problems: underestimating the capabilities means that you might not even try things that could be very beneficial, and underestimating the constraints might mean that you fail to consider and react to important issues.

我们经常会遇到一些人，他们认为深度学习即没什么局限也没什么作用。这都是不对的：低估深度学习的能力意味着你可能都没尝试过有益的使用方法，而低估它的局限性则意味着你没有很好的思考和处理重要的问题。

 

The best thing to do is to keep an open mind. If you remain open to the possibility that deep learning might solve part of your problem with less data or complexity than you expect, then it is possible to design a process where you can find the specific capabilities and constraints related to your particular problem as you work through the process. This doesn't mean making any risky bets — we will show you how you can gradually roll out models so that they don't create significant risks, and can even backtest them prior to putting them in production.

最好的做法是别抱任何成见。也许深度学习可以更简单高效地处理你的一部分问题，如果你愿意尝试，可能在过程中就能找到针对你的问题的能力和局限。这不是说让你去冒险—我们会告诉你如何一步步建立起模型，让他们不会造成严重风险。甚至在真正使用之前我们还可以再次测试他们。

### Starting Your Project

### **开始你的项目**

So where should you start your deep learning journey? The most important thing is to ensure that you have some project to work on—it is only through working on your own projects that you will get real experience building and using models. When selecting a project, the most important consideration is data availability. Regardless of whether you are doing a project just for your own learning or for practical application in your organization, you want something where you can get started quickly. We have seen many students, researchers, and industry practitioners waste months or years while they attempt to find their perfect dataset. The goal is not to find the "perfect" dataset or project, but just to get started and iterate from there.

所以你该从哪儿开始你的深度学习之旅呢？最重要的一点是你要做项目——只有真正做过项目你真的有搭建和使用模型的经验。当你选择项目的时候，一定要考虑是否能拿到你要的数据。不论是为了自学，

还是为了企业应用，你需要找一个能够快速开始的项目。我们已经见到许多学生，研究者，从业者花费数月尝试寻找他们认为最佳的数据集。我们的目的不是为了找出“最佳的”数据集或是项目，而是现在开始，不断迭代。  



If you take this approach, then you will be on your third iteration of learning and improving while the perfectionists are still in the planning stages!

如果你采用了这种方法，那么在完美主义者还在计划阶段时，你将不断学习进步。

 

We also suggest that you iterate from end to end in your project; that is, don't spend months fine-tuning your model, or polishing the perfect GUI, or labelling the perfect dataset… Instead, complete every step as well as you can in a reasonable amount of time, all the way to the end. For instance, if your final goal is an application that runs on a mobile phone, then that should be what you have after each iteration. But perhaps in the early iterations you take some shortcuts, for instance by doing all of the processing on a remote server, and using a simple responsive web application. By completing the project end to end, you will see where the trickiest bits are, and which bits make the biggest difference to the final result.

我们还建议你在项目中端到端地迭代；也就是不要花费太多时间润色你的模型或改进GUI或找到最佳数据集……而是从头到尾都在合理的时间内完成每一步。例如你要做一个手机应用，那你的目的就是在这段时间内做出这个应用来。也许开始你能走一些捷径，比如在远程服务器上用可靠的应用完成这个项目。通过端到端的完成项目，你会找到最难做的点以及那些对结果影响最大的点。

As you work through this book, we suggest that you complete lots of small experiments, by running and adjusting the notebooks we provide, at the same time that you gradually develop your own projects. That way, you will be getting experience with all of the tools and techniques that we're explaining, as we discuss them.

在阅读过程中，我们建议你在逐步完成你的项目同时，利用我们提供的习题集完成一些小实验。这样你能够在我们讨论的同时熟悉我们阐述的工具和技术。

>s: To make the most of this book, take the time to experiment between each chapter, be it on your own project or by exploring the notebooks we provide. Then try rewriting those notebooks from scratch on a new dataset. It's only by practicing (and failing) a lot that you will get an intuition of how to train a model.

> s：为了最大程度利用这本书，在每个章节结束后请花些时间在实践上，将知识点应用在自己的项目中或是钻研我们提供的习题集。然后再抓取一些新的数据集重新完成习题。只有通过不断的实践和失败你才能形成训练模型的直觉。

By using the end-to-end iteration approach you will also get a better understanding of how much data you really need. For instance, you may find you can only easily get 200 labeled data items, and you can't really know until you try whether that's enough to get the performance you need for your application to work well in practice.

在使用端到端迭代的方法时你也会对数据量的真实需求有一个更好的认识。例如你可能发现得到了200个被标记的数据元素非常轻松，直到在尝试使用时你才能知道这些能否让你得到需要的结果。

 

In an organizational context you will be able to show your colleagues that your idea can really work by showing them a real working prototype. We have repeatedly observed that this is the secret to getting good organizational buy-in for a project.

在工作中，一个真正有用的原型能够向你的同事证明你的想法是可行的。我们反复观察到，这是获得公司支持的秘籍。

 

Since it is easiest to get started on a project where you already have data available, that means it's probably easiest to get started on a project related to something you are already doing, because you already have data about things that you are doing. For instance, if you work in the music business, you may have access to many recordings. If you work as a radiologist, you probably have access to lots of medical images. If you are interested in wildlife preservation, you may have access to lots of images of wildlife.

既然最容易开始的是一个有数据基础的项目，那就意味着它可能是你正在做的行当，毕竟你已经有一些相关数据了。比如，如果你从事音乐行业，你会有很多唱片；如果你是放射科医生，你会有很多医学影像。如果你热衷于野生动物保护，你可能有很多野生动物的照片。

 

Sometimes, you have to get a bit creative. Maybe you can find some previous machine learning project, such as a Kaggle competition, that is related to your field of interest. Sometimes, you have to compromise. Maybe you can't find the exact data you need for the precise project you have in mind; but you might be able to find something from a similar domain, or measured in a different way, tackling a slightly different problem. Working on these kinds of similar projects will still give you a good understanding of the overall process, and may help you identify other shortcuts, data sources, and so forth.

有时候你得有点儿创造力，比如在Kaggle比赛里找到一点儿你感兴趣的机器学习项目。有时候你还得学会妥协，要是你找不到能够精准匹配的数据，你可以找些处理相似问题的数据，他们可以是相似领域的，也可以是用于其他标准的。研究这些类似的项目也能够让你对整个过程有一个很好的认识，也能帮你找到其他捷径，或是找到数据源，等等。

 

Especially when you are just starting out with deep learning, it's not a good idea to branch out into very different areas, to places that deep learning has not been applied to before. That's because if your model does not work at first, you will not know whether it is because you have made a mistake, or if the very problem you are trying to solve is simply not solvable with deep learning. And you won't know where to look to get help. Therefore, it is best at first to start with something where you can find an example online where somebody has had good results with something that is at least somewhat similar to what you are trying to achieve, or where you can convert your data into a format similar to what someone else has used before (such as creating an image from your data).

尤其是在你刚开始接触深度学习时，涉足你不熟悉的领域以及深度学习从未应用过的领域不是明智的选择。因为如果你的模型一开始就出错了，你无法判断是模型有错误还是这个问题根本没法用深度学习解决。而且你也不知道去哪儿寻求帮助。因此，你最好选择一个有成功先例的或是有相似先例的项目，或是你能把自己的数据转换成类似于别人使用过的格式（比如用你的数据创建一张图片）。

 

Let's have a look at the state of deep learning, just so you know what kinds of things deep learning is good at right now.

我们来看看深度学习的发展状况，这样你就能知道现在的深度学习擅长解决什么问题。

### The State of Deep Learning

### **深度学习现状**

Let's start by considering whether deep learning can be any good at the problem you are looking to work on. This section provides a summary of the state of deep learning at the start of 2020.

首先让我们思考一下深度学习是否对你所研究的问题有所帮助。这一章节将总结2020年初深度学习的发展状况。

 

However, things move very fast, and by the time you read this some of these constraints may no longer exist. We will try to keep the [book's website](https://book.fast.ai/) up-to-date; in addition, a Google search for "what can AI do now" is likely to provide current information.

然而，科技发展日新月异，当你读到这篇文章时有些局限可能早已消失了。我们尽可能保证 [book's website](https://book.fast.ai/)及时更新。此外，在谷歌搜索“现在AI能做什么”也许能给你一些最新的消息。

#### Computer vision

#### **计算机视觉**

There are many domains in which deep learning has not been used to analyze images yet, but those where it has been tried have nearly universally shown that computers can recognize what items are in an image at least as well as people can—even specially trained people, such as radiologists. This is known as *object recognition*. Deep learning is also good at recognizing where objects in an image are, and can highlight their locations and name each found object. This is known as *object detection* (there is also a variant of this that we saw in <>, where every pixel is categorized based on what kind of object it is part of—this is called *segmentation*). Deep learning algorithms are generally not good at recognizing images that are significantly different in structure or style to those used to train the model. For instance, if there were no black-and-white images in the training data, the model may do poorly on black-and-white images. Similarly, if the training data did not contain hand-drawn images, then the model will probably do poorly on hand-drawn images. There is no general way to check what types of images are missing in your training set, but we will show in this chapter some ways to try to recognize when unexpected image types arise in the data when the model is being used in production (this is known as checking for *out-of-domain* data).

至今还有很多领域没有使用深度学习分析图象，但在已使用过的领域中基本都证实了计算机能够和人类一样识别出图像中的物体，甚至是放射科医生这样的专业人员，这就是物体识别。深度学习还能够识别出物体在图像中的具体位置并且标记出他们的位置并为之署名。这就是物体探测（这也是<>内容的变体：一个像素都根据其所属物体分类，这也被称作划分）。对于和训练对象差别巨大的图像，深度学习算法基本上很难识别。比如说当训练数据里没有黑白图像，那这个模型对于黑白图像的识别能力就会很差。同样的，如果训练数据里没有手绘图象，那这个模型对于手绘图像的识别能力也会很差。其实并没有什么办法来检查你的训练集里缺少了什么类型的图像，但在这一章节我们会给出一些方法来试着识别在生产过程中是否出现未知图像类型（即范围外数据的检查）。

 

One major challenge for object detection systems is that image labelling can be slow and expensive. There is a lot of work at the moment going into tools to try to make this labelling faster and easier, and to require fewer handcrafted labels to train accurate object detection models. One approach that is particularly helpful is to synthetically generate variations of input images, such as by rotating them or changing their brightness and contrast; this is called *data augmentation* and also works well for text and other types of models. We will be discussing it in detail in this chapter.

物体探测系统面临的一大挑战在于图像标记是一项缓慢而昂贵的工程。现在很对人尝试着开发一些工具使标记的过程变得更快更简单，并且减少精确模型中对于手工标记的需要。一个较为有用的方法是改变输入图像的一些参数，比如旋转图像，或是调节他们的亮度和对比度，我们把它称为增加数据，它在文本或其他模型中也同样适用。我们会在这一章节详细讨论。

 

Another point to consider is that although your problem might not look like a computer vision problem, it might be possible with a little imagination to turn it into one. For instance, if what you are trying to classify are sounds, you might try converting the sounds into images of their acoustic waveforms and then training a model on those images.

另一个需要考虑的点是，有时候你的问题看着不像是计算机视觉的范畴，这时候你就得花一点想象力把它变成计算机视觉的问题。比如你要识别的是声音，那就得将声音转换为波形图然后再训练出一个模型。

#### Text (natural language processing)

#### **文本（自然语言加工）**

Computers are very good at classifying both short and long documents based on categories such as spam or not spam, sentiment (e.g., is the review positive or negative), author, source website, and so forth. We are not aware of any rigorous work done in this area to compare them to humans, but anecdotally it seems to us that deep learning performance is similar to human performance on these tasks. Deep learning is also very good at generating context-appropriate text, such as replies to social media posts, and imitating a particular author's style. It's good at making this content compelling to humans too—in fact, even more compelling than human-generated text. However, deep learning is currently not good at generating *correct* responses! We don't currently have a reliable way to, for instance, combine a knowledge base of medical information with a deep learning model for generating medically correct natural language responses. This is very dangerous, because it is so easy to create content that appears to a layman to be compelling, but actually is entirely incorrect.

计算机很善于基于类别的长短文档，比如是不是垃圾邮件，语言的情绪（积极或是消极），作者，源网站等。人们习惯了人类天生具有的分辨能力，所以很难意识到计算机识别的复杂和细微程度，不过有趣的是深度学习在这方面的表现和人类十分相似。深度学习还能够生成特定内容的文本，比如说对于媒体报道的评论，模仿某个作家的写作风格。但是现在深度学习还无法生成正确答案！比如说基于医学知识搭建的模型还无法完全正确地用自然语言生成医学问题的回复。这种做法也很危险，毕竟很有可能生成的回复极具说服力但实际是错误的。

 

Another concern is that context-appropriate, highly compelling responses on social media could be used at massive scale—thousands of times greater than any troll farm previously seen—to spread disinformation, create unrest, and encourage conflict. As a rule of thumb, text generation models will always be technologically a bit ahead of models recognizing automatically generated text. For instance, it is possible to use a model that can recognize artificially generated content to actually improve the generator that creates that content, until the classification model is no longer able to complete its task.

另一个让人担忧的场景是社会媒体中那些有针对性的，极具说服力的回复会被空前的大量使用来散播不实言论，制造不安，引起冲突。根据经验，文本生成模型在技术上总是会比自动识别已生成文档的模型先进一点点。比如那些能够识别文本是否为人工生成的模型或许可以用来不断改进文档生成器直到识别模型再也无法识别出文本来源。

 

Despite these issues, deep learning has many applications in NLP: it can be used to translate text from one language to another, summarize long documents into something that can be digested more quickly, find all mentions of a concept of interest, and more. Unfortunately, the translation or summary could well include completely incorrect information! However, the performance is already good enough that many people are using these systems—for instance, Google's online translation system (and every other online service we are aware of) is based on deep learning.

除了这些问题，深度学习还大量应用于NLP：它能将文本翻译成另一种语言，将很长的文件总结成更易理解的片段，找到所有令人感兴趣的部分在文中被提及的地方，等等。不幸的是，这种翻译或是总结也可能包含了完全错误的信息！不过深度学习在这方面的应用已经足够好到被许多人实际使用了——比如谷歌的线上翻译系统（以及其他所有我们所知的线上服务）就是基于深度学习的应用。

#### Combining text and images

#### 文本与图像结合

The ability of deep learning to combine text and images into a single model is, generally, far better than most people intuitively expect. For example, a deep learning model can be trained on input images with output captions written in English, and can learn to generate surprisingly appropriate captions automatically for new images! But again, we have the same warning that we discussed in the previous section: there is no guarantee that these captions will actually be correct.

深度学习将文本和图像结合进一个模型的能力通常比人们预计的好很多。比如，向一个深度学习的模型输入图像可以输出英文说明，而模型通过学习可以对新的图像生成异常精准的描述。但是再次警告：没有办法可以保证这些描述是正确的。

 

Because of this serious issue, we generally recommend that deep learning be used not as an entirely automated process, but as part of a process in which the model and a human user interact closely. This can potentially make humans orders of magnitude more productive than they would be with entirely manual methods, and actually result in more accurate processes than using a human alone. For instance, an automatic system can be used to identify potential stroke victims directly from CT scans, and send a high-priority alert to have those scans looked at quickly. There is only a three-hour window to treat strokes, so this fast feedback loop could save lives. At the same time, however, all scans could continue to be sent to radiologists in the usual way, so there would be no reduction in human input. Other deep learning models could automatically measure items seen on the scans, and insert those measurements into reports, warning the radiologists about findings that they may have missed, and telling them about other cases that might be relevant.

因为这些严重问题，我们通常不建议将深度学习用作一个完全不加监管的过程，而是作为一个与使用者亲密交互的过程。比起纯手工的方式，这样的方式可以更快完成人类大量的命令。也比一个人工作更加准确。比如一个自动系统可以通过CT扫描的方式识别有潜在中风风险的人，然后发送一个高级别警告让医生赶快看这个片子。中风的有效抢救时间只有三个小时，所以这种快速的反馈可以拯救生命。但同时，所有片子会继续按照正常方式传送到放射科医生那里，所以人工投入不会减少。另一个深度学习模型能够自动测量片子中的物体，然后在报告中记录下来，提醒放射科医生不要遗漏任何东西，并且向他们提供其他可能相关的病例。

 

#### Tabular data

#### 表格数据

For analyzing time series and tabular data, deep learning has recently been making great strides. However, deep learning is generally used as part of an ensemble of multiple types of model. If you already have a system that is using random forests or gradient boosting machines (popular tabular modeling tools that you will learn about soon), then switching to or adding deep learning may not result in any dramatic improvement. Deep learning does greatly increase the variety of columns that you can include—for example, columns containing natural language (book titles, reviews, etc.), and high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID). On the down side, deep learning models generally take longer to train than random forests or gradient boosting machines, although this is changing thanks to libraries such as [RAPIDS](https://rapids.ai/), which provides GPU acceleration for the whole modeling pipeline. We cover the pros and cons of all these methods in detail in <>.

最近，有很多人在研究利用深度学习分析时间序列和表格数据。但是通常深度学习的模型只是众多模型中的一部分。如果你已经有一个基于随机预测或是梯度增长机制（一个你很快会学到的表格建模工具）的系统，那么切换或增加深度学习的应用可能无法得到明显提升。深度学习确实可以增加类目多样性——比如包含自然语言的类目（书名，评论等），或是高维数列（包含大量无序数列如邮编，商品编号）。

深度学习的缺点是深度学习模型的训练时间确实比随机预测或梯度增长机制更长，不过像 [RAPIDS](https://rapids.ai/)这样的知识库为整个建模管道提供了GPU的加速，这也让上述问题得到了改善。我们将这些方法的优劣全都写在了<>中。



#### Recommendation systems

#### 推荐系统

Recommendation systems are really just a special type of tabular data. In particular, they generally have a high-cardinality categorical variable representing users, and another one representing products (or something similar). A company like Amazon represents every purchase that has ever been made by its customers as a giant sparse matrix, with customers as the rows and products as the columns. Once they have the data in this format, data scientists apply some form of collaborative filtering to *fill in the matrix*. For example, if customer A buys products 1 and 10, and customer B buys products 1, 2, 4, and 10, the engine will recommend that A buy 2 and 4. Because deep learning models are good at handling high-cardinality categorical variables, they are quite good at handling recommendation systems. They particularly come into their own, just like for tabular data, when combining these variables with other kinds of data, such as natural language or images. They can also do a good job of combining all of these types of information with additional metadata represented as tables, such as user information, previous transactions, and so forth.

推荐系统其实只是表格数据的一种。他们包含代表用户的高基数分类变量和代表商品的的高基数分类变量。像亚马逊这样的公司就把顾客的每一笔消费都看作是一个巨大的稀疏矩阵：矩阵的横轴是顾客，纵轴是商品。一旦得到这种格式的数据，数据研究者就会应用一些相关的过滤器来填充矩阵，比如顾客A购买了1，10号商品，顾客B购买了1，2，4，10号商品，计算机就会推荐A购买2，4号商品，这正是因为深度学习模型擅长于处理高基数分类变量问题，当然也就包括了推荐系统。深度学习模型不断完善，比如表格数据可以将变量与其他数据结合，类似于自然语言和图像。这些模型还善于将这些信息和其他表格形式的元数据结合，比如用户信息，历史交易等。

However, nearly all machine learning approaches have the downside that they only tell you what products a particular user might like, rather than what recommendations would be helpful for a user. Many kinds of recommendations for products a user might like may not be at all helpful—for instance, if the user is already familiar with the products, or if they are simply different packagings of products they have already purchased (such as a boxed set of novels, when they already have each of the items in that set). Jeremy likes reading books by Terry Pratchett, and for a while Amazon was recommending nothing but Terry Pratchett books to him (see <>), which really wasn't helpful because he already was aware of these books!

但是几乎所有机器学习的方法都有一个弊端，那就是他只能告诉你一个用户可能喜欢的商品是什么，而不能告诉你到底什么建议对用户更有用。许多针对用户喜好的推荐不一定都是有用的——比如用户已经对产品很熟悉了，或者这个产品和他们购买过的仅仅是包装不同而已（比如礼盒装的小说，用户可能已经有套装中的每一本了）。Jeremy喜欢 Terry Pratchett写的书，所以一段时间内亚马逊只向他推荐 Terry Pratchett的书（就像<>），这种推荐就毫无用处因为Jeremy早就知道这些书了！

![Terry Pratchett books recommendation](file:///C:/Users/ThinkPad/Desktop/trans/images/pratchett.png)



#### Other data types

#### 其他数据类型

Often you will find that domain-specific data types fit very nicely into existing categories. For instance, protein chains look a lot like natural language documents, in that they are long sequences of discrete tokens with complex relationships and meaning throughout the sequence. And indeed, it does turn out that using NLP deep learning methods is the current state-of-the-art approach for many types of protein analysis. As another example, sounds can be represented as spectrograms, which can be treated as images; standard deep learning approaches for images turn out to work really well on spectrograms.

有时你会找到一些特定领域的数据类型和现有的类目十分匹配。比如蛋白质链看上去很像自然语言组成的文档，因为他们都是一串串有复杂关系和意义的离散标记。而且事实证明NLP深度学习方法是目前最先进的蛋白质分析方法。另一个例子是，声音可以用图谱的形式展现，这样也就可以当作图像来处理。处理图像的标准深度学习方法确实同样适用于图谱的分析。

### The Drivetrain Approach

### 动力途径

There are many accurate models that are of no use to anyone, and many inaccurate models that are highly useful. To ensure that your modeling work is useful in practice, you need to consider how your work will be used. In 2012 Jeremy, along with Margit Zwemer and Mike Loukides, introduced a method called *the Drivetrain Approach* for thinking about this issue.

有很多复杂精细的模型其实毫无用处，反而一些相对简单的模型非常实用。所以为了能够让你搭建一个实用的模型，你得想想这个模型的实际应用场景。针对这个问题，在2012年Jeremy, Margit Zwemer 和Mike Loukides引进了一种名为“传动系方法”的方式。

 

The Drivetrain Approach, illustrated in <>, was described in detail in ["Designing Great Data Products"](https://www.oreilly.com/radar/drivetrain-approach-data-products/). The basic idea is to start with considering your objective, then think about what actions you can take to meet that objective and what data you have (or can acquire) that can help, and then build a model that you can use to determine the best actions to take to get the best results in terms of your objective.

如<>中所述的传动系方法可以参考["Designing Great Data Products"](https://www.oreilly.com/radar/drivetrain-approach-data-products/)获取更多细节。其基本思想是先考虑你的目的，再考虑什么方法能够达到你的目的以及哪些你现有的或者你能得到的数据对你有用，接着才是搭建一个能做出最佳判断的模型来达成你的目标。

 

Consider a model in an autonomous vehicle: you want to help a car drive safely from point A to point B without human intervention. Great predictive modeling is an important part of the solution, but it doesn't stand on its own; as products become more sophisticated, it disappears into the plumbing. Someone using a self-driving car is completely unaware of the hundreds (if not thousands) of models and the petabytes of data that make it work. But as data scientists build increasingly sophisticated products, they need a systematic design approach.

设想一个自动驾驶的模型：你要让一辆车在没有人为干预的情况下安全的从A点开到B点。这个解决方案里极为重要的一部分是出色的保护模型，但它并不是一个独立存在；只不过随着产品变得越来越复杂，它在研究中变得越来越不突出了。那些开着无人驾驶汽车的人们完全没有意识到这其中用到了数量庞大的模型以及数据字节。但随着数据学家搭建的复杂产品越来越多，他们非常需要一个系统化的设计方法。



We use data not just to generate more data (in the form of predictions), but to produce *actionable outcomes*. That is the goal of the Drivetrain Approach. Start by defining a clear *objective*. For instance, Google, when creating their first search engine, considered "What is the user’s main objective in typing in a search query?" This led them to their objective, which was to "show the most relevant search result." The next step is to consider what *levers* you can pull (i.e., what actions you can take) to better achieve that objective. In Google's case, that was the ranking of the search results. The third step was to consider what new *data* they would need to produce such a ranking; they realized that the implicit information regarding which pages linked to which other pages could be used for this purpose. Only after these first three steps do we begin thinking about building the predictive *models*. Our objective and available levers, what data we already have and what additional data we will need to collect, determine the models we can build. The models will take both the levers and any uncontrollable variables as their inputs; the outputs from the models can be combined to predict the final state for our objective.

我们的目的不仅仅是用数据创造更多的数据（以推测的形式），我们还要生产可执行的产出物。这才是传动系模型的目的。首先请确定一个明确的目标。举个例子：谷歌在设计第一个搜索引擎时首先考虑了“一个用户在搜索框中输入的目的是什么？”，从而明确了用户谷歌的目的，即“展示更多的相关结果”。然后针对这个目的去考虑要使用什么实现手段。在谷歌的案例中，他们所使用的方法是对搜索结果进行排序。第三步是考虑要做出这个排序需要哪些新的数据。他们发现了一个隐藏的信息即页面之间的相互关联关系对这个目的的实现非常有用。在这三步做完以后我们才能去考虑搭建预测模型。我们的目的和手段，我们已有的数据以及我们需要收集的额外数据决定了我们能搭建什么样的模型。这个模型会将手段和不受控变量都作为输入项，而他的输出项将会被结合起来去预测我们的目的最终的实现情况。

 

Let's consider another example: recommendation systems. The objective of a recommendation engine is to drive additional sales by surprising and delighting the customer with recommendations of items they would not have purchased without the recommendation. The lever is the ranking of the recommendations. New data must be collected to generate recommendations that will cause new sales. This will require conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers. This is a step that few organizations take; but without it, you don't have the information you need to actually optimize recommendations based on your true objective (more sales!).

还有另外一个例子：推荐系统。推荐引擎的作用是通过给客户推荐能让他们眼前一亮又未购买过的商品，来引导他们额外消费。这一算法使用的方法是给推荐商品排名，即需要通过收集新的数据来生成推荐从而促成销售。这些大量客户推荐数据则需要通过一定数量的随机试验来获得。能做到这一步的企业少之又少；但如果不做，你将无法得到有用的信息来做出能够切实促进销售目的的推荐。



Finally, you could build two models for purchase probabilities, conditional on seeing or not seeing a recommendation. The difference between these two probabilities is a utility function for a given recommendation to a customer. It will be low in cases where the algorithm recommends a familiar book that the customer has already rejected (both components are small) or a book that they would have bought even without the recommendation (both components are large and cancel each other out).

最后，你能够搭建两种购买概率的模型，分别基于能或不能看见商品推荐。其区别在于前者使用了一种能够为客户做出特定推荐的效用函数，它的算法极少会出现推荐给客户一本类似他拒绝过的书（内容都很少）或不用推荐他也会买的书（内容都很多且互相抵消）。



As you can see, in practice often the practical implementation of your models will require a lot more than just training a model! You'll often need to run experiments to collect more data, and consider how to incorporate your models into the overall system you're developing. Speaking of data, let's now focus on how to find data for your project.

如你所见，你的模型在实际的实施过程中需要的远不止训练而已。你常需要做一些实验来收集更多的数据，还需要如何将你的模型融入整合系统。说到数据，那我们就来看看怎么找到你要的数据。



### Gathering Data

### 收集数据

For many types of projects, you may be able to find all the data you need online. The project we'll be completing in this chapter is a *bear detector*. It will discriminate between three types of bear: grizzly, black, and teddy bears. There are many images on the internet of each type of bear that we can use. We just need a way to find them and download them. We've provided a tool you can use for this purpose, so you can follow along with this chapter and create your own image recognition application for whatever kinds of objects you're interested in. In the fast.ai course, thousands of students have presented their work in the course forums, displaying everything from hummingbird varieties in Trinidad to bus types in Panama—one student even created an application that would help his fiancée recognize his 16 cousins during Christmas vacation!

在很多项目里，你都能够在网上找到所有你要的数据。在这个章节里 我们要完成的项目是一个“熊熊探测器”，它将对三种不同品种的熊加以区分：灰熊，黑熊和泰迪熊。在网上能找到许多这三种熊的图片供我们使用，我们只需要找到并下载这些图片。为此我们向你提供一个工具，这样你就可以跟着这个章节描述的方法加以实践并针对任何你感兴趣的物体创建一个图像识别应用。在fast.ai课程中，无数学生讲他们的作品放在课程论坛上，他们识别的物体也千奇百怪，从Trinidad的蜂鸟品种，到巴拿马的公交车种类—有一个学生甚至为他的未婚夫做了一个应用来在圣诞节假期时区分她的16个表兄弟！

 

At the time of writing, Bing Image Search is the best option we know of for finding and downloading images. It's free for up to 1,000 queries per month, and each query can download up to 150 images. However, something better might have come along between when we wrote this and when you're reading the book, so be sure to check out the [book's website](https://book.fast.ai/) for our current recommendation.

在编写期间，必应的图片搜索是我们认为最好的寻找和下载图片的选择：每个月有最多1000次免费搜索，每次搜索可以下载最多150张图片。当然，在我们编写或你在阅读的时候可能已经出现了更好的工具，所以记得访问 [book's website](https://book.fast.ai/)查看我们最新的推荐。

>important: Keeping in Touch With the Latest Services: Services that can be used for creating datasets come and go all the time, and their features, interfaces, and pricing change regularly too. In this section, we'll show how to use the Bing Image Search API available as part of Azure Cognitive Services at the time this book was written. We'll be providing more options and more up to date information on the [book's website](https://book.fast.ai/), so be sure to have a look there now to get the most current information on how to download images from the web to create a dataset for deep learning.

> 特别说明：时刻关注最新的服务：用来创建数据集的服务时刻在更新换代，而且他们的特征，接口，价格也会经常改变。在这一小节，我们将会使用必应的图片搜索API接口，因为作为Azure认知服务，它在本书编写期是可用的。在本书网站上我们会提供更多的选择以及更多的最新信息，所以你一定要去浏览一下最新的图片下载和数据集搭建的信息。



# Clean

## 数据清洗

To download images with Bing Image Search, sign up at Microsoft for a free account. You will be given a key, which you can copy and enter in a cell as follows (replacing 'XXX' with your key and executing it):

要在必应的图片搜索引擎上下载图片，需要先注册一个微软的免费账号。然后你会拿到一个秘钥，你可以复制并输入如下代码（将'XXX'替换成你的秘钥并执行）：

```
key = 'XXX'
```

Or, if you're comfortable at the command line, you can set it in your terminal with:

或者，如果你更习惯用命令行，你可以在终端执行：

```
export AZURE_SEARCH_KEY=your_key_here
```

and then restart Jupyter Notebook, type this in a cell and execute it:

然后重启Jupyter笔记本，输入如下代码并执行

```
key = os.environ['AZURE_SEARCH_KEY']
```

Once you've set `key`, you can use `search_images_bing`. This function is provided by the small `utils` class included with the notebooks online. If you're not sure where a function is defined, you can just type it in your notebook to find out:

一旦你设置了`key`的值，你就能够使用`search_images_bing`了。这个函数来自于``utils``类，在网上的记事本中可以找到。如果你不能确定一个函数在哪里定义，你可以将其输入记事本来搜索：

```python
search_images_bing
```

<function utils.search_images_bing(key, term, min_sz=128)>

```
results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)
```

150

We've successfully downloaded the URLs of 150 grizzly bears (or, at least, images that Bing Image Search finds for that search term). Let's look at one:

我们成功地下载了150个灰熊图片（或者起码是通过这个关键词搜索到的图片）的链接。其中一个如下所示：



```python
#hide
ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
```

```python
dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
```

```python
im = Image.open(dest)
im.to_thumb(128,128)
```

![Result](data:image/png;base64,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)



This seems to have worked nicely, so let's use fastai's `download_images` to download all the URLs for each of our search terms. We'll put each in a separate folder:

看上去一切顺利，接下来我们将使用fastia的`download_images `来下载所有搜索到的图片URL。我们将它们分别存放在独立的文件夹中：

```python
bear_types = 'grizzly','black','teddy'
path = Path('bears')
```

```python
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('content_url'))
```



Our folder has image files, as we'd expect:

文件夹里有了预期的图片：

```
fns = get_image_files(path)
fns
```

(#421)[Path('bears/black/00000095.jpg'),Path('bears/black/00000133.jpg'),Path('bears/black/00000062.jpg'),Path('bears/black/00000023.jpg'),Path('bears/black/00000029.jpg'),Path('bears/black/00000094.jpg'),Path('bears/black/00000124.jpg'),Path('bears/black/00000056.jpeg'),Path('bears/black/00000046.jpg'),Path('bears/black/00000045.jpg')...]

> J: I just love this about working in Jupyter notebooks! It's so easy to gradually build what I want, and check my work every step of the way. I make a *lot* of mistakes, so this is really helpful to me...

> J: 我非常喜欢用Jupyter记事本完成工作！我能够轻松搭建我想搭建的东西，并且在过程中一步步检查。我犯了很多错误，所以它确实给了我很大的帮助。

 

Often when we download files from the internet, there are a few that are corrupt. Let's check:

当我们从互联网下载文件时，时常会出现损坏的文件。所以我们需要检查：

 

```
failed = verify_images(fns)
failed
```

(#0) []

To remove all the failed images, you can use `unlink` on each of them. Note that, like most fastai functions that return a collection, `verify_images` returns an object of type `L`, which includes the `map` method. This calls the passed function on each element of the collection:

你可以执行`unlink`命令来移除损坏文件。注意，和多数``fastai``函数返回值为一个集合一样，``verify_images ``返回值是一个``L``型对象，其中包含``map``方法。这是集合内每个元素的方向函数：`

```
failed.map(Path.unlink);
```



### Sidebar: Getting Help in Jupyter Notebooks

### 边栏：在Jupyter记事本中求助

upyter notebooks are great for experimenting and immediately seeing the results of each function, but there is also a lot of functionality to help you figure out how to use different functions, or even directly look at their source code. For instance, if you type in a cell:

Jupyter记事本善于完成实验并且快速得到每个函数的结果，除此之外还有很多功能帮助你了解如何使用不同的函数，甚至直接看到他们的源代码。比如当你输入以下命令：

 ```??verify_images```

a window will pop up with:

将弹出一个窗口显示：

```
Signature: verify_images(fns)
Source:   
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in
             enumerate(parallel(verify_image, fns)) if not o)
File:      ~/git/fastai/fastai/vision/utils.py
Type:      function
```

This tells us what argument the function accepts (`fns`), then shows us the source code and the file it comes from. Looking at that source code, we can see it applies the function `verify_image` in parallel and only keeps the image files for which the result of that function is `False`, which is consistent with the doc string: it finds the images in `fns` that can't be opened.

结果显示了函数能够使用什么参数（fns），然后还显示了源代码以及来源文件。当我们看这段源代码时我们能够看到他使用了 `verify_image`函数，并且只保留了函数运行结果为``False``的图片文件，这与doc字符串是一样的：它能够找出``fns``中无法打开的图片。



Here are some other features that are very useful in Jupyter notebooks:

还有一些在使用Jupyter记事本中非常有用的特点：

·  At any point, if you don't remember the exact spelling of a function or argument name, you can press Tab to get autocompletion suggestions.

·  无论什么时候，如果你记不住函数或者参数的完成片拼写，你可以按Tab键获得自动完成的建议。

·  When inside the parentheses of a function, pressing Shift and Tab simultaneously will display a window with the signature of the function and a short description. Pressing these keys twice will expand the documentation, and pressing them three times will open a full window with the same information at the bottom of your screen.

·  在函数的插入中，同时按Shift和Tab键会弹出一个对话框展示函数的签名以及简介。按两侧可以放大文档，按三次可以打开完成的窗口显示相同信息。

·  In a cell, typing `?func_name` and executing will open a window with the signature of the function and a short description.

·  在输入框中输入`?func_name`并执行会弹出一个对话框展示函数的签名以及简介。

·  In a cell, typing `??func_name` and executing will open a window with the signature of the function, a short description, and the source code.

·  在输入框中输入`??func_name`并执行会弹出一个对话框展示函数的签名，简介，还有源代码。

·  If you are using the fastai library, we added a `doc` function for you: executing `doc(func_name)` in a cell will open a window with the signature of the function, a short description and links to the source code on GitHub and the full documentation of the function in the [library docs](https://docs.fast.ai).

·  如果你在使用fastai知识库，我们为你增加了一个doc函数：在输入框里执行`doc(func_name) `会弹出一个对话框展示函数的签名，简介， GitHub上源代码的地址以及[library docs](https://docs.fast.ai)里关于这个函数的完整文档。

·  Unrelated to the documentation but still very useful: to get help at any point if you get an error, type `%debug` in the next cell and execute to open the [Python debugger](https://docs.python.org/3/library/pdb.html), which will let you inspect the content of every variable.

·  最后一点和文档使用无关但也十分有用：出现错误时，在下一个输入框中输入 `%debug`并运行可以打开[Python debugger](https://docs.python.org/3/library/pdb.html)，能够帮助你检查每个变量的内容

### End sidebar

### 边栏结尾

One thing to be aware of in this process: as we discussed in <>, models can only reflect the data used to train them. And the world is full of biased data, which ends up reflected in, for example, Bing Image Search (which we used to create our dataset). For instance, let's say you were interested in creating an app that could help users figure out whether they had healthy skin, so you trained a model on the results of searches for (say) "healthy skin." <> shows you the kinds of results you would get.

在这个过程中需要注意一点：正如我们在<>中所讨论的，模型只能用来反应训练它的数据。这个世界充满了具有偏向性的数据，它们最终显示在例如必应图片搜索引擎的地方，而我们则会用这些数据来搭建自己的数据集。假如你有兴趣搭建一个应用来帮助人们识别他们的皮肤状况，那么你将利用“健康皮肤”的搜索结果训练你的模型，<>显示了你将得到的结果。

![End sidebar](file:///C:/Users/ThinkPad/Desktop/trans/images/healthy_skin.gif)



With this as your training data, you would end up not with a healthy skin detector, but a *young white woman touching her face* detector! Be sure to think carefully about the types of data that you might expect to see in practice in your application, and check carefully to ensure that all these types are reflected in your model's source data. footnote:[Thanks to Deb Raji, who came up with the "healthy skin" example. See her paper ["Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products"](https://dl.acm.org/doi/10.1145/3306618.3314244) for more fascinating insights into model bias.]

如果这些数据成为你的训练数据，你最终得到的不是一个健康皮肤探测器，而是一个“年轻白人女性触摸脸部”探测器！所以请确保你仔细思考了你想要在你的应用中使用什么类型的数据，并且仔细检查这些类型是否能够正确反映在模型的源数据当中。注释：【感谢Deb Raji想出“健康皮肤”的例子。更多关于模型偏见的内容见论文["Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products"](https://dl.acm.org/doi/10.1145/3306618.3314244) 】

 

Now that we have downloaded some data, we need to assemble it in a format suitable for model training. In fastai, that means creating an object called `DataLoaders`.

既然我们已经下载了一些数据，我们需要将它转换成适合训练模型的格式。在fastai里就是要创建一个叫做 `DataLoaders`的对象。

## From Data to DataLoaders

## 从数据到DataLoaders

`DataLoaders` is a thin class that just stores whatever `DataLoader` objects you pass to it, and makes them available as `train` and `valid`. Although it's a very simple class, it's very important in fastai: it provides the data for your model. The key functionality in `DataLoaders` is provided with just these four lines of code (it has some other minor functionality we'll skip over for now):

`DataLoaders`是一个比较小的类，他只用来存储`DataLoader `没通过的数据，然后将他们转换成``train``和``valid``。虽然他是一个很简单的类，但在``fastai``中非常重要：它为你的模型提供数据。`DataLoaders`的关键功能由以下四行代码组成（还有一些次要功能本次跳过）：

```python
class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])
```

> jargon: DataLoaders: A fastai class that stores multiple `DataLoader` objects you pass to it, normally a `train` and a `valid`, although it's possible to have as many as you like. The first two are made available as properties.

> 术语解释：DataLoaders：一个fastai的类，用来储存各种`DataLoader `没通过的数据，通常是一个``train``或一个``valid``，当然亦可以很多。前两个被设置为固定值。



Later in the book you'll also learn about the` Dataset` and ` Datasets` classes, which have the same relationship.

在这本书后面的内容中你也会学到`Dataset`和`Datasets`类，他们也有同样的关系。

 

To turn our downloaded data into a `DataLoaders` object we need to tell fastai at least four things:

想要将下载的数据转换成`DataLoaders`对象，我们起码要告诉fastai以下四件事：

 

- What kinds of data we are working with
- How to get the list of items
- How to label these items
- How to create the validation set
- 我们在是用什么类型的数据
- 如何获得条目列表
- 如何标记这些条目
- 如何创建有效集

 

So far we have seen a number of factory methods for particular combinations of these things, which are convenient when you have an application and data structure that happen to fit into those predefined methods. For when you don't, fastai has an extremely flexible system called the data block API. With this API you can fully customize every stage of the creation of your `DataLoaders`. Here is what we need to create a `DataLoaders` for the dataset that we just downloaded:

时至今日我们已经见到很多工厂式的方法来结合这些东西，当你刚好有适合这个已知方法的应用和数据结构，那么使用这个方法将为你带来一些便捷。如果你没有，你也可以使用fastai中一个极其灵活的系统，它叫做data block API。利用这个API接口你可以自己定制每个搭建`DataLoaders`环节。这里向你展示我们为刚才下载的数据集创建一个`DataLoaders`所需的步骤：

```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```



Let's look at each of these arguments in turn. First we provide a tuple where we specify what types we want for the independent and dependent variables:

让我们依次看一下这些参数。首先我们提供一个数组来确定我们想要什么类型的自变量和应变量:

```
blocks=(ImageBlock, CategoryBlock)
```



The *independent variable* is the thing we are using to make predictions from, and the *dependent variable* is our target. In this case, our independent variables are images, and our dependent variables are the categories (type of bear) for each image. We will see many other types of block in the rest of this book.

*自变量*是用来进行预测的，而*应变量*才是我们的目标。在这个案例中，我们的自变量是图片，应变量是各张图片的分类（基于图中熊的种类）。在接下来的内容中我们还会见到许多其他种类的模块。



For this `DataLoaders` our underlying items will be file paths. We have to tell fastai how to get a list of those files. The `get_image_files` function takes a path, and returns a list of all of the images in that path (recursively, by default):

`DataLoaders`指我们隐藏的内容即文件的路径。我们要告诉fastai如何获得这些文件的列表。`get_image_files`函数会访问一个路径然后返回一张包含这个路径里所有的图片的列表（默认是递归的方式）：



```
get_items=get_image_files
```



Often, datasets that you download will already have a validation set defined. Sometimes this is done by placing the images for the training and validation sets into different folders. Sometimes it is done by providing a CSV file in which each filename is listed along with which dataset it should be in. There are many ways that this can be done, and fastai provides a very general approach that allows you to use one of its predefined classes for this, or to write your own. In this case, however, we simply want to split our training and validation sets randomly. However, we would like to have the same training/validation split each time we run this notebook, so we fix the random seed (computers don't really know how to create random numbers at all, but simply create lists of numbers that look random; if you provide the same starting point for that list each time—called the *seed*—then you will get the exact same list each time):

通常你下载的数据集已经包含了一个被定义的变量集。有时候是通过将用来训练的图片和变量集放进不同的文件夹，有时候是通过提供一个CSV文件，其中所有的文件名都根据他所属的数据集进行罗列。有很多方法可以实现这个功能，而fastai则是提供了一个非常通用的方法——你可以使用其中一个预定义的类或者自己定义一个类。在这个例子中我们只是想要随机划分我们的训练集和变量集。不过我们想要每次执行我们的笔记时都能够划分出同样的结果，所以我们修复了随机种子（计算机完全不知道如何生成随机数，他只是生成一张看上去很随机的列表；如果你每次都规定好起始点—称作种子—那你每次都能得到同样的列表）：

```
splitter=RandomSplitter(valid_pct=0.2, seed=42)
```



The independent variable is often referred to as `x` and the dependent variable is often referred to as `y`. Here, we are telling fastai what function to call to create the labels in our dataset:

自变量通常用`x`表示，应变量则是`y`。此处我们要告诉fastai该调用什么函数来创建我们数据集里的标签：

```
get_y=parent_label
```



`parent_label` is a function provided by fastai that simply gets the name of the folder a file is in. Because we put each of our bear images into folders based on the type of bear, this is going to give us the labels that we need.

`parent_label`是fastai提供的函数，用来获取包含文件的文件夹名。因为我们依据熊的种类将他们的图片放进不同的文件夹，我们将能够得到我们要的标签。

Our images are all different sizes, and this is a problem for deep learning: we don't feed the model one image at a time but several of them (what we call a *mini-batch*). To group them in a big array (usually called a *tensor*) that is going to go through our model, they all need to be of the same size. So, we need to add a transform which will resize these images to the same size. *Item transforms* are pieces of code that run on each individual item, whether it be an image, category, or so forth. fastai includes many predefined transforms; we use the `Resize` transform here:

我们的图片都是不同尺寸的，这也是深度学习面临的一个问题：我们会一次给模型几张图片（我们称之为*小批量*）。如果要将他们都放在一个输入模型的大向量里（通常叫做*张量*），他们需要尺寸一致。所以我们要添加一个转换操作将这些图片转换成同意尺寸。*对象转换*是每个独立项目中运行的代码片段，不论这个对象是一张图片，一个类目或是其他。fastai包含了许多预定义的转换；此处我们可以使用`Resize`：

```
item_tfms=Resize(128)
```



This command has given us a `DataBlock` object. This is like a *template* for creating a `DataLoaders`. We still need to tell fastai the actual source of our data—in this case, the path where the images can be found:

这个命令为我们生成了一个`DataBlock`对象。它类似于一个创建`DataLoaders`的模板。我们仍需要告诉fastai数据的确切来源—在这个例子中，能找到图片的路径为：

```
dls = bears.dataloaders(path)
```

A `DataLoaders` includes validation and training `DataLoader`s. `DataLoader` is a class that provides batches of a few items at a time to the GPU. We'll be learning a lot more about this class in the next chapter. When you loop through a `DataLoader` fastai will give you 64 (by default) items at a time, all stacked up into a single tensor. We can take a look at a few of those items by calling the `show_batch` method on a `DataLoader`:

一个`DataLoaders`包含了验证以及训练`DataLoader`的功能。`DataLoader`这个类能够一次给CPU提供很多对象。在下一章我们会学习更多有关他的内容。当你循环一个`DataLoader`类，fastai会一次默认分配给你64个对象，都堆在同一个张量里。我们可以通过在`DataLoader`里调用`show_batch`方法来查看一部分的内容：

```
dls.valid.show_batch(max_n=4, nrows=1)
```



图



By default `Resize` *crops* the images to fit a square shape of the size requested, using the full width or height. This can result in losing some important details. Alternatively, you can ask fastai to pad the images with zeros (black), or squish/stretch them:

在宽度或者高度不变的情况下，`Resize`默认将图像适应符合尺寸要求的正方形。不过这会损失掉一些重要细节。你也可以选择使用fastai来用0像素（黑色）填充图像或者挤压/拉伸它们。

```
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```



图



```
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```



图



All of these approaches seem somewhat wasteful, or problematic. If we squish or stretch the images they end up as unrealistic shapes, leading to a model that learns that things look different to how they actually are, which we would expect to result in lower accuracy. If we crop the images then we remove some of the features that allow us to perform recognition. For instance, if we were trying to recognize breeds of dog or cat, we might end up cropping out a key part of the body or the face necessary to distinguish between similar breeds. If we pad the images then we have a whole lot of empty space, which is just wasted computation for our model and results in a lower effective resolution for the part of the image we actually use.

所有这些方法看上去总有些费事，或是说有些问题。因为当我们把图像压缩或是拉伸以后它们会变成一个不真实的形状，模型用来训练的物体则和实际不同了，也就可能导致较低的准确性。而通过裁减得到的图像会损失掉一部分帮助我们识别的内容。比如当我们想要识别狗或猫的品种时，使用裁减的方式可能会丢失身体或者面部的重要部分，这些部分恰恰是用来区分相似品种的重要信息。填充0像素的方法则会使得图像大面积空白，这会导致模型算力的浪费，影响我们识别真正有用的部分。



Instead, what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch (which is one complete pass through all of our images in the dataset) we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images. It also reflects how images work in the real world: different photos of the same thing may be framed in slightly different ways.

反而在实际应用中我们通常会随机的选取图像的一部分加以裁剪。在每一次循环中（一个循环即将数据集中所有图片都访问一遍）都会随机选取不同的部分。也就是说我们的模型可以学习如何聚焦，识别图像中不同的物体。这也反映了现实世界中的图像是怎样的：同一个事物的不同那个照片总会有细微之处的差别。



In fact, an entirely untrained neural network knows nothing whatsoever about how images behave. It doesn't even recognize that when an object is rotated by one degree, it still is a picture of the same thing! So actually training the neural network with examples of images where the objects are in slightly different places and slightly different sizes helps it to understand the basic concept of what an object is, and how it can be represented in an image.

其实一个完全未经训练的神经网络对于图像的内容是一无所知的。当一个物体旋转了一度，他也无法认出那还是原来的东西。所以来训练的图片可以是有细微差别的，比如物体位置略微不同，或者物体大小略微不同。这样的话可以帮助神经网络认识到一个物体的基本概念，以及这个物体在图片中有如何的表现形式。



Here's another example where we replace `Resize` with `RandomResizedCrop`, which is the transform that provides the behavior we just described. The most important parameter to pass in is `min_scale`, which determines how much of the image to select at minimum each time:

下一个例子是我们用 `RandomResizedCrop`取代了 `Resize` ，来实现我们上述的转换方式。 `min_scale`是其中最重要的一个参数，他决定了每次截取的图像范围：

```
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```



图



We used `unique=True` to have the same image repeated with different versions of this `RandomResizedCrop` transform. This is a specific example of a more general technique, called data augmentation.

我们使用 `unique=True` 来获得 `RandomResizedCrop` 产生的不同版本的图像。下述是一个通用技术的典型案例，，被称为数据扩充。

### Data Augmentation

### 数据扩充

*Data augmentation* refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes and contrast changes. For natural photo images such as the ones we are using here, a standard set of augmentations that we have found work pretty well are provided with the `aug_transforms` function. Because our images are now all the same size, we can apply these augmentations to an entire batch of them using the GPU, which will save a lot of time. To tell fastai we want to use these transforms on a batch, we use the `batch_tfms` parameter (note that we're not using `RandomResizedCrop` in this example, so you can see the differences more clearly; we're also using double the amount of augmentation compared to the default, for the same reason):

*数据扩充* 是指随机生成输入数据的变化形式，即表现形式不同但内容相同。常见的数据扩充技术包括旋转，翻转，扭曲，亮度调节以及对比度调节。对于类似我们使用的这种自然图片，我们找到了一个由 `aug_transforms` 函数生成的标准扩充集，而且他的表现非常不错。因为我们现有的图片都为同一尺寸，我们可以用GPU对一批图片使用这些扩充方式，这会节省很多时间。我们用 `batch_tfms` 参数来让fastai对一批图片进行转换（我们没有在这里使用 `RandomResizedCrop` ，所以我们可以更清晰的看见其中的区别；为此我们也会使用默认扩充数量的两倍来处理）：

```
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```



图



Now that we have assembled our data in a format fit for model training, let's actually train an image classifier using it.

既然我们已经将所有数据转换成适合模型训练的格式，就让我们真正用它来训练一个图像分类器。



## Training Your Model, and Using It to Clean Your Data

## 训练你的模型，并用它清洗你的数据

Time to use the same lines of code as in <> to train our bear classifier.

We don't have a lot of data for our problem (150 pictures of each sort of bear at most), so to train our model, we'll use `RandomResizedCrop` with an image size of 224 px, which is fairly standard for image classification, and default `aug_transforms`:

现在该用<>里的代码来训练你的熊熊识别器了。

我们没有大量的可用图片（每个品种的熊最多只有150张），所以我们要用 `RandomResizedCrop`将图像转换成224像素。这样的图像算是图像识别的标准格式，并且默认使用`aug_transforms`:

```
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
```



We can now create our `Learner` and fine-tune it in the usual way:

现在我们可以创建 `Learner` 并对其进行微调：

```
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```



| epoch | train_loss | valid_loss | error_rate | time  |
| :---- | :--------- | :--------- | :--------- | :---- |
| 0     | 1.235733   | 0.212541   | 0.087302   | 00:05 |

| epoch | train_loss | valid_loss | error_rate | time  |
| :---- | :--------- | :--------- | :--------- | :---- |
| 0     | 0.213371   | 0.112450   | 0.023810   | 00:05 |
| 1     | 0.173855   | 0.072306   | 0.023810   | 00:06 |
| 2     | 0.147096   | 0.039068   | 0.015873   | 00:06 |
| 3     | 0.123984   | 0.026801   | 0.015873   | 00:06 |



Now let's see whether the mistakes the model is making are mainly thinking that grizzlies are teddies (that would be bad for safety!), or that grizzlies are black bears, or something else. To visualize this, we can create a *confusion matrix*:

现在我们来看模型产生的错误是否主要是把灰熊当成了泰迪熊（这可是很危险的！），还是把灰熊当成了黑熊，或者其他东西。为了清晰看到这些错误，我们可以创建一个*错误矩阵*：

```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

![](data:image/png;base64,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)



The rows represent all the black, grizzly, and teddy bears in our dataset, respectively. The columns represent the images which the model predicted as black, grizzly, and teddy bears, respectively. Therefore, the diagonal of the matrix shows the images which were classified correctly, and the off-diagonal cells represent those which were classified incorrectly. This is one of the many ways that fastai allows you to view the results of your model. It is (of course!) calculated using the validation set. With the color-coding, the goal is to have white everywhere except the diagonal, where we want dark blue. Our bear classifier isn't making many mistakes!

矩阵中的行分别代表我们数据集中所有的黑熊，灰熊，泰迪熊。列则分别代表被模型判断成的黑熊，灰熊，泰迪熊。因此，矩阵对角线的数字就是被正确判断的图像数量，非对角线的数字则代表被误判的数量。这是fastai中检查模型训练结果的方法之一。当然它是用变化形式的图片集来计算的。通过对颜色处理编码，我们将对角线设置成深蓝色，其他地方设置成白色。这样我们就可以清晰看到结果——我们的熊熊识别器并没有产生很多错误！

   

It's helpful to see where exactly our errors are occurring, to see whether they're due to a dataset problem (e.g., images that aren't bears at all, or are labeled incorrectly, etc.), or a model problem (perhaps it isn't handling images taken with unusual lighting, or from a different angle, etc.). To do this, we can sort our images by their *loss*.

我们要去了解错误到底发生在哪里，错误是否是数据集的问题造成的（比如图像上根本就不是熊，或者被标记错误等），还是模型自身的问题造成的（也许它无法应对不同光线的，不同角度的图像）。这对我们来说是一件极有帮助的事。为此我们需要以他们的丢失数来给图像分类。



The loss is a number that is higher if the model is incorrect (especially if it's also confident of its incorrect answer), or if it's correct, but not confident of its correct answer. In a couple of chapters we'll learn in depth how loss is calculated and used in the training process. For now, `plot_top_losses` shows us the images with the highest loss in our dataset. As the title of the output says, each image is labeled with four things: prediction, actual (target label), loss, and probability. The *probability* here is the confidence level, from zero to one, that the model has assigned to its prediction:

丢失数的数值较高则可能是模型不准确（它还对结果很自信），或是结果是正确的但是模型本身无法确定。在好几个章节中我们都将深入学习丢失数是如何计算以及在训练过程中使用的。现在，`plot_top_losses` 找到了我们数据集中丢失数最高的图像。正如输出的标题所示，每个图像都标记了四个指标：预测，实际（目标标记），丢失数，和概率。这里的*概率*是模型对于预测结果的自信等级，从0到1：

```
interp.plot_top_losses(5, nrows=1)
```

![]()

This output shows that the image with the highest loss is one that has been predicted as "grizzly" with high confidence. However, it's labeled (based on our Bing image search) as "black." We're not bear experts, but it sure looks to us like this label is incorrect! We should probably change its label to "grizzly."

结果显示损失最多的图片是被预测为“灰熊”的那张，且这一结果被判断为高可靠性。不过根据必应图像搜索的结果，它是被标注为“黑熊”的。我们不是研究熊的专家，但这一结果看起来是错的！我们应该吧标签改为“灰熊”。



The intuitive approach to doing data cleaning is to do it *before* you train a model. But as you've seen in this case, a model can actually help you find data issues more quickly and easily. So, we normally prefer to train a quick and simple model first, and then use it to help us with data cleaning.

直接办法是在训练模型之前清洗数据。但如此案例所示，模型其实可以帮助你更快更便捷的找到数据的问题。所以我们通常倾向于先训练一个快捷简单的模型，然后用它帮助我们清洗数据。



fastai includes a handy GUI for data cleaning called `ImageClassifierCleaner` that allows you to choose a category and the training versus validation set and view the highest-loss images (in order), along with menus to allow images to be selected for removal or relabeling:

Fastai 中包含了一个数据清洗的GUI叫做 `ImageClassifierCleaner` ，你能够选择类别以及培训与验证集，并按序查看损失最大的图像，当然你也可以选择要删除或重新标记的图像的菜单：



```
#hide_output
cleaner = ImageClassifierCleaner(learn)
cleaner
```

![]()

```
#hide
# for idx in cleaner.delete(): cleaner.fns[idx].unlink()
# for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

We can see that amongst our "black bears" is an image that contains two bears: one grizzly, one black. So, we should choose `<Delete>` in the menu under this image. `ImageClassifierCleaner` doesn't actually do the deleting or changing of labels for you; it just returns the indices of items to change. So, for instance, to delete (`unlink`) all images selected for deletion, we would run:

我们发现在“黑熊”图片中包含了两种熊：一种是灰熊，一种是黑熊。所以，我们选择图片下方的 `<Delete>` 菜单。 `ImageClassifierCleaner` 实际上不能删除或者变更标签；他只能返回图像的索引以供修改。所以假如我们要删除所有选中的图像（使用`unlink`），我们可以运行：

```
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
```

To move images for which we've selected a different category, we would run:

运行以下代码将选中的图像移动至不同的类别：



```
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

> s: Cleaning the data and getting it ready for your model are two of the biggest challenges for data scientists; they say it takes 90% of their time. The fastai library aims to provide tools that make it as easy as possible.
>
> s:清洗并准备好数据是数据科学家最大的两个挑战；他们称这占据了他们90%的时间。fastai的知识库想要提供尽可能简便的工具。



We'll be seeing more examples of model-driven data cleaning throughout this book. Once we've cleaned up our data, we can retrain our model. Try it yourself, and see if your accuracy improves!

在这本书里我们将看到更多例子清洗以模型驱动的数据。一旦我们清洗了我们的数据，我们可以重新训练我们的模型。请你自己尝试一下来看看你的准确性有没有提高！



> note: No Need for Big Data: After cleaning the dataset using these steps, we generally are seeing 100% accuracy on this task. We even see that result when we download a lot fewer images than the 150 per class we're using here. As you can see, the common complaint that *you need massive amounts of data to do deep learning* can be a very long way from the truth!
>
> 注释：没必要使用大数据：在使用这些步骤清洗数据集之后，我们通常可以得到100% 的准确率。甚至在我们只下载了比原先更少图片时，我们依旧能够得到这样的结果。如你所见，诸如“你需要大量数据来做深度学习”的说法可能并不准确！
>
> 

Now that we have trained our model, let's see how we can deploy it to be used in practice.

既然我们训练了我们的模型，就让我们看看我们能够如何使用他们。



## Turning Your Model into an Online Application

## 将你的模型变成线上应用

We are now going to look at what it takes to turn this model into a working online application. We will just go as far as creating a basic working prototype; we do not have the scope in this book to teach you all the details of web application development generally.

现在我们来看看如何将这个模型变成一个可用的线上应用。我们会搭建一个基本的工作模型；在这本书里我们没有足够的章节教授完整的应用搭建细节。



### Using the Model for Inference

### 使用推算模型



Once you've got a model you're happy with, you need to save it, so that you can then copy it over to a server where you'll use it in production. Remember that a model consists of two parts: the *architecture* and the trained *parameters*. The easiest way to save the model is to save both of these, because that way when you load a model you can be sure that you have the matching architecture and parameters. To save both parts, use the `export` method.

当你得到了一个满意的模型，你需要保存下来这样就可以复制到任何一个用于生产的服务器上。记住模型包含两部分：架构以及训练的参数。最简单的保存方法是保存这两部分，那么当你下载它时才能够保证包含了匹配的架构及参数。 `export` 可以用来保存这两部分。



This method even saves the definition of how to create your `DataLoaders`. This is important, because otherwise you would have to redefine how to transform your data in order to use your model in production. fastai automatically uses your validation set `DataLoader` for inference by default, so your data augmentation will not be applied, which is generally what you want.

这个方法甚至能够保存你如何创建 `DataLoaders`的方法。这很重要，否则当你要使用你的模型时还要重新定义如何转换你的数据。默认fastai能够自动使用有效集 `DataLoader` 进行推断，所以即便你想，你的数据扩充方法也不会被使用。



When you call `export`, fastai will save a file called "export.pkl":

当你调用 `export`, fastai会保存一个"export.pkl"文件：



```
learn.export()
```

Let's check that the file exists, by using the `ls` method that fastai adds to Python's `Path` class:

我们使用fastai添加在Python `Path` 类中的 `ls` 方法来验证文件是否存在：



```
path = Path()
path.ls(file_exts='.pkl')
```

输出结果： (#1) [Path('export.pkl')]

You'll need this file wherever you deploy your app to. For now, let's try to create a simple app within our notebook.

不论你把应用部署在哪儿，你都会用到这个文件。现在，让我们尝试在笔记本里创建一个简单的应用。



When we use a model for getting predictions, instead of training, we call it *inference*. To create our inference learner from the exported file, we use `load_learner` (in this case, this isn't really necessary, since we already have a working `Learner` in our notebook; we're just doing it here so you can see the whole process end-to-end):

我们用模型来得到预测结果的这个过程被称作推算而不是训练。我们使用`load_learner` 来为生成的文件创建一个推算学习器（这并不是必须的，因为书已经有了一个可用的 `Learner` ；在这里我们只是为了让你能够看到实现的全过程）：



```
learn_inf = load_learner(path/'export.pkl')
```

When we're doing inference, we're generally just getting predictions for one image at a time. To do this, pass a filename to `predict`:

当我们在推算时，通常每次只能得到一副图像的预测结果。所以我们给 `predict`一个文件名：



输出结果： learn_inf.predict('images/grizzly.jpg')



```
('grizzly', tensor(1), tensor([9.0767e-06, 9.9999e-01, 1.5748e-07]))
```

This has returned three things: the predicted category in the same format you originally provided (in this case that's a string), the index of the predicted category, and the probabilities of each category. The last two are based on the order of categories in the *vocab* of the `DataLoaders`; that is, the stored list of all possible categories. At inference time, you can access the `DataLoaders` as an attribute of the `Learner`:

此处将返回三个结果：根据你给出的格式生成的预测类目（此处是一个字符串），预测类目的指针，以及每个类目的几率。后两者是基于类目名称在`DataLoaders`中词汇的排序；也就是说，其中包含了存储的所有可能的类目。在推算时，你能够获得一个 `DataLoaders` ，它是 `Learner`的其中一个属性：



```
learn_inf.dls.vocab
```



输出结果： (#3) ['black','grizzly','teddy']

We can see here that if we index into the vocab with the integer returned by `predict` then we get back "grizzly," as expected. Also, note that if we index into the list of probabilities, we see a nearly 1.00 probability that this is a grizzly.

由此可见如果我们指向一个词，其中包含 `predict` 函数返回的整数，我们就会如愿得到“灰熊”的结果。同时请注意如果我们指向几率列表，我们会看见结果是灰熊的几率近乎为1.00。



We know how to make predictions from our saved model, so we have everything we need to start building our app. We can do it directly in a Jupyter notebook.

我们已经知道了如何用我们保存的模型生成预测结果，所以，万事俱备，只差开始搭建我们的应用了。我们可以直接在Jupyter记事本里开始写。



### Creating a Notebook App from the Model

### 用模型搭建一个文本应用

To use our model in an application, we can simply treat the `predict` method as a regular function. Therefore, creating an app from the model can be done using any of the myriad of frameworks and techniques available to application developers.

要在应用中使用我们的模型，我们可以直接把 `predict` 方法当做一个常用函数。因此我们可以使用任何一个可用架构和技术来创建这个应用。



However, most data scientists are not familiar with the world of web application development. So let's try using something that you do, at this point, know: it turns out that we can create a complete working web application using nothing but Jupyter notebooks! The two things we need to make this happen are:

不过大多数数据科学家并不熟悉网页应用的开发。所以我们试着用一些你熟悉的东西，或者说你只是知道而已：事实证明我们仅需要用Jupyter记事本就可以搭建一个完整的网页应用！我们要用到的两个东西是：



- IPython widgets (ipywidgets)
- Voilà

*IPython widgets* are GUI components that bring together JavaScript and Python functionality in a web browser, and can be created and used within a Jupyter notebook. For instance, the image cleaner that we saw earlier in this chapter is entirely written with IPython widgets. However, we don't want to require users of our application to run Jupyter themselves.

*IPython widgets* 是GUI组件，他们能够将JavaScript和Python的功能放在同一个浏览器页面中使用，并且它们可以在Jupyter记事本中实现。比如上文中提到的图像清洗器就完全是用*IPython widgets* 写的。不过我们不想要求我们应用的使用者自己运行Jupyter。



That is why *Voilà* exists. It is a system for making applications consisting of IPython widgets available to end users, without them having to use Jupyter at all. Voilà is taking advantage of the fact that a notebook *already is* a kind of web application, just a rather complex one that depends on another web application: Jupyter itself. Essentially, it helps us automatically convert the complex web application we've already implicitly made (the notebook) into a simpler, easier-to-deploy web application, which functions like a normal web application rather than like a notebook.

所以出现了 *Voilà* 。这一系统使得终端用户能够使用包含IPython widgets的应用，它就比Jupyter本身稍微复杂一点点。它将我们创建的网页（基于记事本的）应用自动转化得更加简单，更容易部署，所以和记事本相比它本身更像是一种网页应用。



But we still have the advantage of developing in a notebook, so with ipywidgets, we can build up our GUI step by step. We will use this approach to create a simple image classifier. First, we need a file upload widget:

用记事本开发依旧是有好处的，所以我们可以利用*IPython widgets* 一步一步搭建我们的GUI。我们将使用这种方法穿件一个简单的图像识别器。首先我们需要一个文件来上传这些小工具：



```
#hide_output
btn_upload = widgets.FileUpload()
btn_upload
```

![An upload button](images/att_00008.png)

Now we can grab the image:

现在我们可以抓取图像了：



```
#hide
# For the book, we can't actually click an upload button, so we fake it
btn_upload = SimpleNamespace(data = ['images/grizzly.jpg'])
```

```
img = PILImage.create(btn_upload.data[-1])
```

![Output widget representing the image](images/att_00009.png)

We can use an `Output` widget to display it:

我们用 `Output` 工具来运行：



```
#hide_output
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl
```

![Output widget representing the image](images/att_00009.png)

Then we can get our predictions:

这样我们就可以得到预测结果了：



```
pred,pred_idx,probs = learn_inf.predict(img)
```

and use a `Label` to display them:

用 `Label` 运行：



```
#hide_output
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred
Prediction: grizzly; Probability: 1.0000
```

We'll need a button to do the classification. It looks exactly like the upload button:

我们会用到一个按钮来执行分类。他看上去很想上传按钮：



```
#hide_output
btn_run = widgets.Button(description='Classify')
btn_run
```

We'll also need a *click event handler*; that is, a function that will be called when it's pressed. We can just copy over the lines of code from above:

我们还需要一个*点击处理*；当按钮被点击时这个功能就会被唤起。我们可以直接复制上述的代码行：

```
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)
```



You can test the button now by pressing it, and you should see the image and predictions update automatically!

We can now put them all in a vertical box (`VBox`) to complete our GUI:

现在你可以点击这个按钮加以测试，你会看见图片和预测结果自动上传了！

现在将他们都放进一个下拉框 (`VBox`) 里，就可以完成你的GUI了：



```
#hide
#Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()
```

```
#hide_output
VBox([widgets.Label('Select your bear!'), 
      btn_upload, btn_run, out_pl, lbl_pred])
```

![The whole widget](images/att_00011.png)

We have written all the code necessary for our app. The next step is to convert it into something we can deploy.

我们已经完成了我们的应用的最关键代码。下一步是将它们转换成我们可以不部署的形式。



### Turning Your Notebook into a Real App

### 将你的记事本变成一个真正的应用



```
#hide
# !pip install voila
# !jupyter serverextension enable voila —sys-prefix
```

Now that we have everything working in this Jupyter notebook, we can create our application. To do this, start a new notebook and add to it only the code needed to create and show the widgets that you need, and markdown for any text that you want to appear. Have a look at the *bear_classifier* notebook in the book's repo to see the simple notebook application we created.

既然Jupyter里所有的东西都可以正常工作了，我们就可以搭建我们的应用了。首先新建一个记事本，只添加搭建和显示工具需要的代码，并且标记所有你想要展示的内容。请看一下书中关于 *bear_classifier* 的内容来了解我们穿兼得简单应用。



Next, install Voilà if you haven't already, by copying these lines into a notebook cell and executing it:

接下来，如果还没安装Voilà ，请复制下述代码行到记事本中并且运行它：



```
!pip install voila
!jupyter serverextension enable voila —sys-prefix
```

Cells that begin with a `!` do not contain Python code, but instead contain code that is passed to your shell (bash, Windows PowerShell, etc.). If you are comfortable using the command line, which we'll discuss more later in this book, you can of course simply type these two lines (without the `!` prefix) directly into your terminal. In this case, the first line installs the `voila` library and application, and the second connects it to your existing Jupyter notebook.

带有 `!` 的命令不包含Python代码，但是包含复制到shell上的代码 (bash, Windows PowerShell等)。如果你习惯用命令行（我们将在后面的章节谈论命令行），你仍旧可以将不含 `!` 的两行代码敲进你的命令提示符，第一行用来安装 `voila` 和应用，第二行和你的Jupyter记事本连接。



Voilà runs Jupyter notebooks just like the Jupyter notebook server you are using now does, but it also does something very important: it removes all of the cell inputs, and only shows output (including ipywidgets), along with your markdown cells. So what's left is a web application! To view your notebook as a Voilà web application, replace the word "notebooks" in your browser's URL with: "voila/render". You will see the same content as your notebook, but without any of the code cells.

用Voilà运行Jupyter记事本就和Jupyter记事本服务器一样，但它也做了一些非常重要的是：它移除了所有单元的输入，只显示输出（包括*IPython widgets*）和标记的单元。所以，剩下的就是网页应用！将你的记事本看作是Voilà网页应用，将“notebooks”的文字换成网页链接"voila/render"。你会看见和记事本中一样的内容，只不过没有任何的代码单元。

Of course, you don't need to use Voilà or ipywidgets. Your model is just a function you can call (`pred,pred_idx,probs = learn.predict(img)`), so you can use it with any framework, hosted on any platform. And you can take something you've prototyped in ipywidgets and Voilà and later convert it into a regular web application. We're showing you this approach in the book because we think it's a great way for data scientists and other folks that aren't web development experts to create applications from their models.

当然你不是必须要用Voilà或者ipywidgets。你可以把你的模型当做一个方法(`pred,pred_idx,probs = learn.predict(img)`)，所以你可以在任何架构任何平台上使用它。

We have our app, now let's deploy it!

现在应用有了，那就来部署它吧！



### Deploying your app

### 部署你的应用

As you now know, you need a GPU to train nearly any useful deep learning model. So, do you need a GPU to use that model in production? No! You almost certainly *do not need a GPU to serve your model in production*. There are a few reasons for this:

现在你知道了你需要一个GUI来训练差不多所有有用的深度学习模型。所以在实际使用中你是否需要一个GUI来使用这个模型呢？不！你基本不需要在实际使用中用一个GUI来服务你的模型。理由如下：



- As we've seen, GPUs are only useful when they do lots of identical work in parallel. If you're doing (say) image classification, then you'll normally be classifying just one user's image at a time, and there isn't normally enough work to do in a single image to keep a GPU busy for long enough for it to be very efficient. So, a CPU will often be more cost-effective.

- 已知GPU在大量识别的工作同时发生时很有用。如果你在做我们说的图像区分，通常来说一次只会识别一个用户的图像，而对于单张图片的识别工作并不能够让GPU工作足够长的时间，这不足以使GUI明显增加他的效率。

  

- An alternative could be to wait for a few users to submit their images, and then batch them up and process them all at once on a GPU. But then you're asking your users to wait, rather than getting answers straight away! And you need a high-volume site for this to be workable. If you do need this functionality, you can use a tool such as Microsoft's [ONNX Runtime](https://github.com/microsoft/onnxruntime), or [AWS Sagemaker](https://aws.amazon.com/sagemaker/)

- 一个选择是等待多个用户提交他们的图像，然后批量上传并在GPU中一次计算。但你得让你的用户等待一段时间才能得到答案！而且你需要一个大容量的网站使他能够工作。如果你确实需要这个功能，你可以使用诸如Microsoft的 [ONNX Runtime](https://github.com/microsoft/onnxruntime), 或者 [AWS Sagemaker](https://aws.amazon.com/sagemaker/)等工具。

  

- The complexities of dealing with GPU inference are significant. In particular, the GPU's memory will need careful manual management, and you'll need a careful queueing system to ensure you only process one batch at a time.

- 处理GPU推算的复杂性是非常重要的。尤其是需要非常仔细的手动管理GPU的存储，并且你需要一个惊喜的队列机制来确保一次处理一个推送。

  

- There's a lot more market competition in CPU than GPU servers, as a result of which there are much cheaper options available for CPU servers.

- 在CPU服务器的行业中存在着比GPU行业更多的市场竞争，导致有更多便宜的CPU服务器以供选择。

  

Because of the complexity of GPU serving, many systems have sprung up to try to automate this. However, managing and running these systems is also complex, and generally requires compiling your model into a different form that's specialized for that system. It's typically preferable to avoid dealing with this complexity until/unless your app gets popular enough that it makes clear financial sense for you to do so.

由于GPU服务的复杂性，诞生了许多系统尝试自动运行。但这些系统的管理和运行也十分复杂，通常还需要将你的模型编译成适合系统的形式。这样能够更好地避免因复杂性导致的问题，当然等到你的应用足够畅销时，为了多赚钱你还是要处理这些问题。



For at least the initial prototype of your application, and for any hobby projects that you want to show off, you can easily host them for free. The best place and the best way to do this will vary over time, so check the [book's website](https://book.fast.ai/) for the most up-to-date recommendations. As we're writing this book in early 2020 the simplest (and free!) approach is to use [Binder](https://mybinder.org/). To publish your web app on Binder, you follow these steps:

至少对于初版的原型，你可以融入任何你喜欢的元素，反正是免费的。当然这么做最好的时机取决于时间的不同，所以你可以在 [book's website](https://book.fast.ai/) 上查看最新的推荐你。我们是在2020年初编写这本书的，所以这时候最方便（且免费）的选择就是使用 [Binder](https://mybinder.org/)了。你可以使用如下步骤将你的网页应用发布在Binder上：



2. Paste the URL of that repo into Binder's URL, as shown in <>.
3. Change the File dropdown to instead select URL.
4. In the "URL to open" field, enter `/voila/render/name.ipynb` (replacing `name` with the name of for your notebook).
5. Click the clickboard button at the bottom right to copyt the URL and paste it somewhere safe.
6. Click Launch.

1. 将文中的链接粘贴至Binder中。

2. 更换文件下拉框以选中链接。

3. 在“"URL to open”文件中输入 `/voila/render/name.ipynb` （将 `name` 替换成你的文本的名字）。

4. 点击右下角的点击按钮复制链接，然后把它粘贴至某处保存起来。

5. 点击运行。

   

The first time you do this, Binder will take around 5 minutes to build your site. Behind the scenes, it is finding a virtual machine that can run your app, allocating storage, collecting the files needed for Jupyter, for your notebook, and for presenting your notebook as a web application.

第一次执行时，Binder要花大约5分钟建立你的站点。他会在后台寻找一台虚拟机运行你的应用，分配存储空间，找到相应的文件来运行Jupyter，运行你的记事本，并将你的记事本以网页应用的形式加以展现。



Finally, once it has started the app running, it will navigate your browser to your new web app. You can share the URL you copied to allow others to access your app as well.

最终，当开始运行你的应用时，他将你的浏览器换成你的新应用。你可以将你复制的链接分享给别人，这样他们也可以有权是用你的应用。



For other (both free and paid) options for deploying your web app, be sure to take a look at the [book's website](https://book.fast.ai/).

请一定要看 [book's website](https://book.fast.ai/)，这里还有很多其他选择能让你部署你的应用（有免费的，也有付费的）。



You may well want to deploy your application onto mobile devices, or edge devices such as a Raspberry Pi. There are a lot of libraries and frameworks that allow you to integrate a model directly into a mobile application. However, these approaches tend to require a lot of extra steps and boilerplate, and do not always support all the PyTorch and fastai layers that your model might use. In addition, the work you do will depend on what kind of mobile devices you are targeting for deployment—you might need to do some work to run on iOS devices, different work to run on newer Android devices, different work for older Android devices, etc. Instead, we recommend wherever possible that you deploy the model itself to a server, and have your mobile or edge application connect to it as a web service.

你可能想在手机上安装你的应用，或者诸如Raspberry Pi之类的终端系统。有许多的知识库和架构能帮助你将一个模型直接集成为一个手机应用。但这些方法有很大可能性会要做很多额外的步骤和模板，也无法完全支持所有你用到的PyTorch 和 fastai。另外，你的工作量还取决于你部署的设备—在iOS设备上运行时你可能需要一些额外的步骤，在新的和旧的安卓设备上使用方法也不同，等等。所以我们推荐你将他部署在服务器上，然后连接你的手机或者移动终端，把他当做一个线上服务来使用。



There are quite a few upsides to this approach. The initial installation is easier, because you only have to deploy a small GUI application, which connects to the server to do all the heavy lifting. More importantly perhaps, upgrades of that core logic can happen on your server, rather than needing to be distributed to all of your users. Your server will have a lot more memory and processing capacity than most edge devices, and it is far easier to scale those resources if your model becomes more demanding. The hardware that you will have on a server is also going to be more standard and more easily supported by fastai and PyTorch, so you don't have to compile your model into a different form.

这个方法有很多好处。最初的安装比较简单，因为你只要安装一个很小的GUI应用，由他来连接到服务器做其他的活。或许更重要的是，比起升级所有用户端，你只需要升级服务器的核心逻辑。你的服务器会有大量的内存，运行能力也比大部分终端设备高，当你的模型要求变高时，在服务器上更容易规划这些资源。且服务器上的硬件会更加标准，更容易被fastai和PyTorch支持，所以你也不用转换格式了。



There are downsides too, of course. Your application will require a network connection, and there will be some latency each time the model is called. (It takes a while for a neural network model to run anyway, so this additional network latency may not make a big difference to your users in practice. In fact, since you can use better hardware on the server, the overall latency may even be less than if it were running locally!) Also, if your application uses sensitive data then your users may be concerned about an approach which sends that data to a remote server, so sometimes privacy considerations will mean that you need to run the model on the edge device (it may be possible to avoid this by having an *on-premise* server, such as inside a company's firewall). Managing the complexity and scaling the server can create additional overhead too, whereas if your model runs on the edge devices then each user is bringing their own compute resources, which leads to easier scaling with an increasing number of users (also known as *horizontal scaling*).

当然也有劣势。你的应用会需要网络连接，每次请求时你的模型都会有延迟。（神经网络的运行需要时间，所以额外的网络延迟并不会对个人用户造成很大影响。事实上因为你在服务器上可以使用更好的硬件，延迟时间可能比你在终端运行时间还短！）另外如果你的应用包含了敏感数据的话，用户可能会担心这些数据被传输到一个远程的服务器上是否安全，所以有时隐私问题意味着你不得不在终端设备运行你的模型（使用加密的服务器可能会避免这个问题，比如说企业内网）。当用户在终端设备上运行模型而不是使用你的算力，对复杂性的管理以及对于服务器规模的扩大都会造成额外的算力浪费。这一行为也更容易让用户数增长（也就是横向拓展）。



> A: I've had a chance to see up close how the mobile ML landscape is changing in my work. We offer an iPhone app that depends on computer vision, and for years we ran our own computer vision models in the cloud. This was the only way to do it then since those models needed significant memory and compute resources and took minutes to process inputs. This approach required building not only the models (fun!) but also the infrastructure to ensure a certain number of "compute worker machines" were absolutely always running (scary), that more machines would automatically come online if traffic increased, that there was stable storage for large inputs and outputs, that the iOS app could know and tell the user how their job was doing, etc. Nowadays Apple provides APIs for converting models to run efficiently on device and most iOS devices have dedicated ML hardware, so that's the strategy we use for our newer models. It's still not easy but in our case it's worth it, for a faster user experience and to worry less about servers. What works for you will depend, realistically, on the user experience you're trying to create and what you personally find is easy to do. If you really know how to run servers, do it. If you really know how to build native mobile apps, do that. There are many roads up the hill.
>
> A: 在我的工作中，我有幸见证了移动ML景观的变迁。我们提供一个基于计算机视觉的iPhone应用，而多年来我们在云上运行我们的计算机视觉模型。这些模型需要很强的内存和算力，还需要数分钟来写入，因此这是唯一的使用方法。这个方法不仅需要搭建模型（有趣！），还需要搭建一个确保一定数量的计算机能够运行的架构（可怕），一旦流量增加，更多机器能够自动及时上线确保有足够的稳定存储空间来应付大量输入输出，那么iOS应用可以知道运行状况如何并告知用户。现在苹果公司提供了相关的接口来保证模型能够在设备上运行顺畅，而大部分的iOS设备都有专用的ML硬件，所以这也是我们我们在更新的模型上使用的策略。这种方式依旧不算轻松，但在我们的例子中，是值得一做的，这能让我们拥有更快的用户体验，也不用过多担心服务器。实际上这些工作取决于你想要打造什么样的用户体验，以及你自己觉得怎么样做更加简单。如果你真的知道怎么运行服务器，就去做。如果你擅长建造本地手机应用，就去做。得到最终结果的方法不止一个。



Overall, we'd recommend using a simple CPU-based server approach where possible, for as long as you can get away with it. If you're lucky enough to have a very successful application, then you'll be able to justify the investment in more complex deployment approaches at that time.

总之，只要可以再做替换，我们推荐你使用一个简单的CPU服务器。如果你很幸运的拥有了一个成功的应用，那时候再去考虑投资更为复杂的部署。



Congratulations, you have successfully built a deep learning model and deployed it! Now is a good time to take a pause and think about what could go wrong.

恭喜你成功搭建并部署了一个深度学习模型。现在应该停一停，想想哪里可能会出错。



## How to Avoid Disaster

## 如何避免灾难

In practice, a deep learning model will be just one piece of a much bigger system. As we discussed at the start of this chapter, a data product requires thinking about the entire end-to-end process, from conception to use in production. In this book, we can't hope to cover all the complexity of managing deployed data products, such as managing multiple versions of models, A/B testing, canarying, refreshing the data (should we just grow and grow our datasets all the time, or should we regularly remove some of the old data?), handling data labeling, monitoring all this, detecting model rot, and so forth. In this section we will give an overview of some of the most important issues to consider; for a more detailed discussion of deployment issues we refer to you to the excellent [Building Machine Learning Powered Applications](http://shop.oreilly.com/product/0636920215912.do) by Emmanuel Ameisen (O'Reilly)

事实上深度学习模型只是更大系统的一部分。正如我们在开头谈到的，一个数据产品需要考虑从概念到生产的整个端到端的过程。我们不能在本书中包含所有部署数据产品的难点，比如多版本模型的管理，A/B测试，冒泡测试，更新数据（我们是不是让我们的数据集不断增长，还是定期删除旧的数据？），处理数据标签，并且监控这一切，检测模型漏洞，等等。在这一小节我们来看看一字儿最重要的问题；我们向你推荐Emmanuel Ameisen (O'Reilly)的 [Building Machine Learning Powered Applications](http://shop.oreilly.com/product/0636920215912.do) 来针对部署问题进行深入探讨。



One of the biggest issues to consider is that understanding and testing the behavior of a deep learning model is much more difficult than with most other code you write. With normal software development you can analyze the exact steps that the software is taking, and carefully study which of these steps match the desired behavior that you are trying to create. But with a neural network the behavior emerges from the model's attempt to match the training data, rather than being exactly defined.

最大的问题之一就是理解和测试深度学习模型的行为要比你写的其他代码复杂得多。通常在软件开发中你可以确切分析软件运行的每一步，并且细心学习哪些步骤匹配你想要的行为。但在神经网络中的行为不能被确切定义，而是由模型决定于训练数据的判断。



This can result in disaster! For instance, let's say we really were rolling out a bear detection system that will be attached to video cameras around campsites in national parks, and will warn campers of incoming bears. If we used a model trained with the dataset we downloaded there would be all kinds of problems in practice, such as:

这会造成极大的灾难。比如我们想要实现一个熊熊识别器并连接国家公园露营地旁的录像机，在熊靠近时能够警示露营者。如果我们用的是经由下载的数据集训练的模型，将会有一大堆的问题，比如：



- Working with video data instead of images

- Handling nighttime images, which may not appear in this dataset

- Dealing with low-resolution camera images

- Ensuring results are returned fast enough to be useful in practice

- Recognizing bears in positions that are rarely seen in photos that people post online (for example from behind, partially covered by bushes, or when a long way away from the camera)

- 这一场景使用的是录像数据而不是图像

- 要处理夜间画面，而这没有出现在数据集中

- 要处理低分辨率照片

- 要确保使用中结果回传足够快速

- 要识别处在不常见的位置中的熊（比如从后面，被灌木部分遮挡，或者离相机很远）

  

A big part of the issue is that the kinds of photos that people are most likely to upload to the internet are the kinds of photos that do a good job of clearly and artistically displaying their subject matter—which isn't the kind of input this system is going to be getting. So, we may need to do a lot of our own data collection and labelling to create a useful system.

大部分的问题就处在人们上传的大部分图像都是取景绝佳，效果极好的作品—和这个系统真正得到的输入相差极大。所以，我们需要大量收集并标记数据以建造一个实用的系统。



This is just one example of the more general problem of *out-of-domain* data. That is to say, there may be data that our model sees in production which is very different to what it saw during training. There isn't really a complete technical solution to this problem; instead, we have to be careful about our approach to rolling out the technology.

这只是因范围之外的数据而产生的问题之一。也就是说，生产环境中得到的数据可能和训练环境中的差别巨大。针对这个问题还没有一个科学办法能够解决；我们需要仔细思考如何实现这项技术。



There are other reasons we need to be careful too. One very common problem is *domain shift*, where the type of data that our model sees changes over time. For instance, an insurance company may use a deep learning model as part of its pricing and risk algorithm, but over time the types of customers that the company attracts, and the types of risks they represent, may change so much that the original training data is no longer relevant.

还有一些你应该仔细思考的原因。一个非常常见的就是范围转换，因为我们的模型看到的数据类型一直在变。比如一家保险公司会在他们的定价算法和风险预测算法中使用深度学习模型，但是公司吸引的客户，他们代表的风险种类都会和最初的数据相差巨大甚至毫不相干。



Out-of-domain data and domain shift are examples of a larger problem: that you can never fully understand the entire behaviour of your neural network. They have far too many parameters to be able to analytically understand all of their possible behaviors. This is the natural downside of their best feature—their flexibility, which enables them to solve complex problems where we may not even be able to fully specify our preferred solution approaches. The good news, however, is that there are ways to mitigate these risks using a carefully thought-out process. The details of this will vary depending on the details of the problem you are solving, but we will attempt to lay out here a high-level approach, summarized in <>, which we hope will provide useful guidance.

范围外的数据和范围转换代表了一个巨大的问题：你无法完全理解你的神经网络的行为。他们距离能够辩证的理解所有可能行为还有很长一段路。他们最大的特点也带来了不足—他们的灵活性，这种灵活性能够让他们在我们无法选择最佳方案时处理复杂问题。好消息是通过仔细思考，还有很多减轻这种风险的方法。细节会随着你处理的问题不同而改变，但是在这里我们尝试着给出一个高级的方法来提供有用的指导。



![Deployment process](images/att_00061.png)

Where possible, the first step is to use an entirely manual process, with your deep learning model approach running in parallel but not being used directly to drive any actions. The humans involved in the manual process should look at the deep learning outputs and check whether they make sense. For instance, with our bear classifier a park ranger could have a screen displaying video feeds from all the cameras, with any possible bear sightings simply highlighted in red. The park ranger would still be expected to be just as alert as before the model was deployed; the model is simply helping to check for problems at this point.

如果可能的话，第一步就是使用一个完全手动的过程，同时你的深度学习模型也在运行但是不会直接驱动任何动作。负责手动的人需要查看深度学习的输出结果并且检查他们是否是有意义的结果。比如在公园使用熊熊识别器时，就需要有人监控所有录像来查看是否有被标记为红色的可疑目标。在这个模型被部署使用之前，公园看守人就需要在预测到熊靠近时发出警示；在这种情况下模型只是简单的用来发现问题。



The second step is to try to limit the scope of the model, and have it carefully supervised by people. For instance, do a small geographically and time-constrained trial of the model-driven approach. Rather than rolling our bear classifier out in every national park throughout the country, we could pick a single observation post, for a one-week period, and have a park ranger check each alert before it goes out.

第二步是试着限制模型范围，并且让人们能够监控。比如在地理条件和时间条件限制的情况下对模型驱动的方法做一个测试。比起在所有国家公园使用我们的熊熊识别器，我们可以以一周时间为限挑选一个观测点，让公园守卫在发布预警前复核每个警告。



Then, gradually increase the scope of your rollout. As you do so, ensure that you have really good reporting systems in place, to make sure that you are aware of any significant changes to the actions being taken compared to your manual process. For instance, if the number of bear alerts doubles or halves after rollout of the new system in some location, we should be very concerned. Try to think about all the ways in which your system could go wrong, and then think about what measure or report or picture could reflect that problem, and ensure that your regular reporting includes that information.

接着不断扩大使用范围。这时请确保你有非常良好的汇报机制来保证你能够察觉到这一过程和手动过程相比的变化。比如在一些新的地点使用新系统后熊出没的警告翻倍了或是减半了，我们要额外小心了。试着排查所有系统出错的可能原因，然后寻找什么方法或报告或图片能够反映这个问题，并确保你的常规报告中包含这些信息。



> J: I started a company 20 years ago called *Optimal Decisions* that used machine learning and optimization to help giant insurance companies set their pricing, impacting tens of billions of dollars of risks. We used the approaches described here to manage the potential downsides of something going wrong. Also, before we worked with our clients to put anything in production, we tried to simulate the impact by testing the end-to-end system on their previous year's data. It was always quite a nerve-wracking process, putting these new algorithms into production, but every rollout was successful.
>
> J：我在20年前创办了我的公司 *Optimal Decisions* 。我们利用机器学习和优选法帮助大型保险公司定价，影响到数亿美元的风险。我们用这个方法管控一些潜在的缺点。同时，在我们将客户的任何东西投入进来之前我们都会试着用历年数据加以测试来模拟影响。将这些新的算法投产是一个令人头疼的过程，但好在每次都成功了。



### Unforeseen Consequences and Feedback Loops

### 不可预见的情况和反馈机制



One of the biggest challenges in rolling out a model is that your model may change the behaviour of the system it is a part of. For instance, consider a "predictive policing" algorithm that predicts more crime in certain neighborhoods, causing more police officers to be sent to those neighborhoods, which can result in more crimes being recorded in those neighborhoods, and so on. In the Royal Statistical Society paper ["To Predict and Serve?"](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x), Kristian Lum and William Isaac observe that: "predictive policing is aptly named: it is predicting future policing, not future crime."

使用一个模型最大的挑战就是你的模型会改变他所属系统的行为。假如一个“警务预测”算法在一个特定的街区预测出更多的犯罪事件，那么更对的警力被送到这里，又会造成这个街区有更多的犯罪记录，等等。在一份官方的统计学论文 ["为了预测和服务?"](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x)，Kristian Lum 和 William Isaac 说到：警务预测这个名字非常准确：它在预测警力，而不是预测犯罪。“



Part of the issue in this case is that in the presence of bias (which we'll discuss in depth in the next chapter), *feedback loops* can result in negative implications of that bias getting worse and worse. For instance, there are concerns that this is already happening in the US, where there is significant bias in arrest rates on racial grounds. [According to the ACLU](https://www.aclu.org/issues/smart-justice/sentencing-reform/war-marijuana-black-and-white), "despite roughly equal usage rates, Blacks are 3.73 times more likely than whites to be arrested for marijuana." The impact of this bias, along with the rollout of predictive policing algorithms in many parts of the US, led Bärí Williams to [write in the *New York Times*](https://www.nytimes.com/2017/12/02/opinion/sunday/intelligent-policing-and-my-innocent-children.html): "The same technology that’s the source of so much excitement in my career is being used in law enforcement in ways that could mean that in the coming years, my son, who is 7 now, is more likely to be profiled or arrested—or worse—for no reason other than his race and where we live."

这个例子中的问题在于偏见的存在（我们将在下一章节深入探讨），反馈机制也会对这些偏见造成负面的影响。在美国已经产生了这样的担忧，不同种族的逮捕率正在产生种族偏见。根据[ACLU](https://www.aclu.org/issues/smart-justice/sentencing-reform/war-marijuana-black-and-white)所述， “尽管大家做了同样的事，黑人因为大麻被逮捕的比例是白人的3.73倍。”随着警务预测算法在美国使用而造成的偏见让 Bärí Williams 在 [*New York Times*](https://www.nytimes.com/2017/12/02/opinion/sunday/intelligent-policing-and-my-innocent-children.html) 中写道：“这个在我职业生涯中带来诸多兴奋的科技已经使用在执法工作中，这意味着在以后的日子里，我7岁的儿子很可能仅仅因为他的种族和他生活的地方而被捕。”



A helpful exercise prior to rolling out a significant machine learning system is to consider this question: "What would happen if it went really, really well?" In other words, what if the predictive power was extremely high, and its ability to influence behavior was extremely significant? In that case, who would be most impacted? What would the most extreme results potentially look like? How would you know what was really going on?

在使用机器学习系统前的一个很好的练习就是思考一下这个问题：“如果真的大量使用，会发生什么？”或者说是，如果潜在的影响非常非常大，能影响的行为非常重要，会发生什么？那么谁会受到最大的影响？最极端的情况可能会是什么样子的？你要如何知道真正会发生的是什么？



Such a thought exercise might help you to construct a more careful rollout plan, with ongoing monitoring systems and human oversight. Of course, human oversight isn't useful if it isn't listened to, so make sure that there are reliable and resilient communication channels so that the right people will be aware of issues, and will have the power to fix them.

这样的思考或许能够帮助你在现有监管机制和人的监视中制定一个更加靠谱的使用计划。当然，人的监管要被遵守才有用，所以请确保你有可靠的沟通渠道让正确的人知道问题并加以解决。



## Get Writing!

## 开始写作！



One of the things our students have found most helpful to solidify their understanding of this material is to write it down. There is no better test of your understanding of a topic than attempting to teach it to somebody else. This is helpful even if you never show your writing to anybody—but it's even better if you share it! So we recommend that, if you haven't already, you start a blog. Now that you've completed Chapter 2 and have learned how to train and deploy models, you're well placed to write your first blog post about your deep learning journey. What's surprised you? What opportunities do you see for deep learning in your field? What obstacles do you see?

对于学生来说最有用的就是牢记这些材料最好的方式就是写下来。而测试你理解程度最好的方式就是试着教授另一个人。而更好的方式是向他人分享你写作的成果。所以如果你还没有的话，我们推荐你开始写博客。既然你已经完成了 第二章节的学习并且已经学习了如何训练部署模型，你可以开始写关于你的深度学习之旅的的第一篇博文了。你可以写写让你惊讶的东西，在你的领域你看到了什么深度学习的应用机会，或者你遇到的障碍。



Rachel Thomas, cofounder of fast.ai, wrote in the article ["Why You (Yes, You) Should Blog"](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045):

fast.ai的联合创始人Rachel Thomas在文章 ["你为什么要写博客（对，说的就是你）"](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045)当中写道：



```
asciidoc
____
The top advice I would give my younger self would be to start blogging sooner. Here are some reasons to blog:

* It’s like a resume, only better. I know of a few people who have had blog posts lead to job offers!
* Helps you learn. Organizing knowledge always helps me synthesize my own ideas. One of the tests of whether you understand something is whether you can explain it to someone else. A blog post is a great way to do that.
* I’ve gotten invitations to conferences and invitations to speak from my blog posts. I was invited to the TensorFlow Dev Summit (which was awesome!) for writing a blog post about how I don’t like TensorFlow.
* Meet new people. I’ve met several people who have responded to blog posts I wrote.
* Saves time. Any time you answer a question multiple times through email, you should turn it into a blog post, which makes it easier for you to share the next time someone asks.
如果能给年轻的我一个建议，那一定会是赶快开始写博客。原因如下：
*博客就像是你的简历，并且比简历更好的展示你。我知道不少人因为发布的博文拿到了工作邀请！
*博客帮助你学习。整理你的知识总是能够帮你整合自己的想法。检验你是否理解一个东西的方法就是你是否能够将他解释给别人听。写博客就是很好的一个途径。
*通过发表博客，我收到过很多会议邀请和演讲邀请。我还曾因为在博客吐槽 TensorFlow 而受邀参加 TensorFlow 开发峰会（牛逼！）。
*遇见新的人。我认识了很多回复我的博客的人。
*节约时间。每次你回复无数份邮件都在回答同一个问题时，你可以直接发一个博客回应，下次有人再问你就可以很方便的把博文分享给他。

____
```

Perhaps her most important tip is this:

也许她最重要的意见就是：



> : You are best positioned to help people one step behind you. The material is still fresh in your mind. Many experts have forgotten what it was like to be a beginner (or an intermediate) and have forgotten why the topic is hard to understand when you first hear it. The context of your particular background, your particular style, and your knowledge level will give a different twist to what you’re writing about.
>
> ：对于距你一步之遥的人来是，你是帮助他们的最佳人选。你脑海中的知识还热乎着。很多专家都忘了做一个初学者（或是刚入门的人）的感受，也忘了当你第一次听到这些概念时是多么难以理解。因为你特殊的背景，特殊的风格以及你的知识面，你将写出独一无二的东西。



We've provided full details on how to set up a blog in <>. If you don't have a blog already, take a look at that now, because we've got a really great approach set up for you to start blogging for free, with no ads—and you can even use Jupyter Notebook!

 在文中我们描述了创建博客的所有细节。如果你还没有博客，快去看看吧，因为我们为你准备了一个绝佳且免费的写博客方法，没有广告哦~你还可以直接用Jupyter记事本！



## Questionnaire

## 练习题



1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
2. Where do text models currently have a major deficiency?
3. What are possible negative societal implications of text generation models?
4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
5. What kind of tabular data is deep learning particularly good at?
6. What's a key downside of directly using a deep learning model for recommendation systems?
7. What are the steps of the Drivetrain Approach?
8. How do the steps of the Drivetrain Approach map to a recommendation system?
9. Create an image recognition model using data you curate, and deploy it on the web.
10. What is `DataLoaders`?
11. What four things do we need to tell fastai to create `DataLoaders`?
12. What does the `splitter` parameter to `DataBlock` do?
13. How do we ensure a random split always gives the same validation set?
14. What letters are often used to signify the independent and dependent variables?
15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
16. What is data augmentation? Why is it needed?
17. What is the difference between `item_tfms` and `batch_tfms`?
18. What is a confusion matrix?
19. What does `export` save?
20. What is it called when we use a model for getting predictions, instead of training?
21. What are IPython widgets?
22. When might you want to use CPU for deployment? When might GPU be better?
23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
25. What is "out-of-domain data"?
26. What is "domain shift"?
27. What are the three steps in the deployment process?

1. 基于训练数据结构和风格的不同，举例说明熊熊识别器在哪个使用场景中效果比较差。
2. 文本模型现在的缺点在哪里？
3. 文本模型可能产生哪些负面的社会影响？
4. 当模型出错且可能是有害的情况下，还有什么好的选择来进行自动化流程？
5. 深度学习特别擅长哪种表格数据？
6. 在推荐系统中直接使用深度学习模型有什么主要缺点？
7. 传动系方法的步骤有哪些？
8. 这些步骤如何构建一个推荐系统的？
9. 利用你收集的数据搭建一个图像识别模型，在网站上部署。
10. 什么是 `DataLoaders`？
11. 我们要告诉fastai哪四件事来创建`DataLoaders`？
12.  `splitter` 参数在 `DataBlock` 中有什么作用？
13. 如何确保随机分割总是能得到一样的变量集？
14. 通常使用哪些字母来标记独立的和非独立的变量？
15. 裁剪、填充和压缩调整大小的方法都有什么不同？你如何从中选择一个合适的方法？
16. 什么是数据扩充？作用是什么？
17.  `item_tfms` 和 `batch_tfms`的差别是什么？
18. 什么是混淆矩阵？
19.  `export` 保存什么？
20. 我们用模型获得预测而不是训练的过程叫什么？
21. IPython widgets是什么？
22. 什么时候你会选择使用CPU部署？什么时候选择GPU？
23. 和手机，个人电脑之类的用户设备（或是终端设备）相比，将应用部署在服务器上的弊端是什么？
24. 举例说出在应用熊熊警告系统时会出现的三个问题？
25. 什么是“范围外数据”？
26. 什么是“范围变化”？
27. 部署过程的三个步骤是什么？



### Further Research

### 进一步研究



1. Consider how the Drivetrain Approach maps to a project or problem you're interested in.
2. When might it be best to avoid certain types of data augmentation?
3. For a project you're interested in applying deep learning to, consider the thought experiment "What would happen if it went really, really well?"
4. Start a blog, and write your first blog post. For instance, write about what you think deep learning might be useful for in a domain you're interested in.

1. 思考传动系方法是如何指导你感兴趣的项目或是问题的。
2. 什么时候避免特定的数据扩充是最好的？
3. 将深度学习应用在你感兴趣的项目中，思考这个问题“如果他被广泛使用，将会发生什么？”
4. 开通你的博客，并开始第一次写作。比如深度学习会在哪个你感兴趣的领域中产生作用。