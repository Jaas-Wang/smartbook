# Data Ethics

# 数据伦理学

### Sidebar: Acknowledgement: Dr. Rachel Thomas

### 侧边栏：致谢：雷切尔·托马斯博士

This chapter was co-authored by Dr. Rachel Thomas, the cofounder of fast.ai, and founding director of the Center for Applied Data Ethics at the University of San Francisco. It largely follows a subset of the syllabus she developed for the [Introduction to Data Ethics](https://ethics.fast.ai/) course.

本章节由雷切尔·托马斯博士联合编写，他是 fast.ai 的联合创始人，也是旧金山大学应用数据伦理学中心的创始负责人。它很大程度上沿用了她开发的[数据伦理概论](https://ethics.fast.ai/) 课程的大纲的子集。

### End sidebar

### 侧边栏结束

As we discussed in Chapters 1 and 2, sometimes machine learning models can go wrong. They can have bugs. They can be presented with data that they haven't seen before, and behave in ways we don't expect. Or they could work exactly as designed, but be used for something that we would much prefer they were never, ever used for.

正好我们在第一章和第二章所讨论的，有时候机器学习模型会犯错。他们会有缺陷。对于它们以前没有见过的数据它们能呈现出我们不希望的行为。或者它们能够完全按照设计工作，但我们更喜欢把它们用于那些从来没用过的事情上。

Because deep learning is such a powerful tool and can be used for so many things, it becomes particularly important that we consider the consequences of our choices. The philosophical study of *ethics* is the study of right and wrong, including how we can define those terms, recognize right and wrong actions, and understand the connection between actions and consequences. The field of *data ethics* has been around for a long time, and there are many academics focused on this field. It is being used to help define policy in many jurisdictions; it is being used in companies big and small to consider how best to ensure good societal outcomes from product development; and it is being used by researchers who want to make sure that the work they are doing is used for good, and not for bad.

因为深度学习是一个强大的工作并能被用于很多事情上，我们考虑选择的后果变的尤为重要。*伦理学*的哲学研究是一门对和错的研究，包含了我们能够如何定义那些术语，识别对和错的行为，以及理解行为和结果间的关系。数据伦理学领域已经存在了很久了，并且很多学者关注这一领域。它用于帮助很多司法体系制定政策；在很多大和小的公司它被用于去思考如何最大限度的确保产品开发带来好的社会结果；它被研究人员用于思考确保他们的工作用于好的方面而不是坏的方面。

As a deep learning practitioner, therefore, it is likely that at some point you are going to be put in a situation where you need to consider data ethics. So what is data ethics? It's a subfield of ethics, so let's start there.

所以做为深度学习参与者，可能有些时候你处理思考数据伦理的情况。那么什么是数据伦理？它是伦理学的一个子领域，所以让我从这里开始吧。

> J: At university, philosophy of ethics was my main thing (it would have been the topic of my thesis, if I'd finished it, instead of dropping out to join the real world). Based on the years I spent studying ethics, I can tell you this: no one really agrees on what right and wrong are, whether they exist, how to spot them, which people are good, and which bad, or pretty much anything else. So don't expect too much from the theory! We're going to focus on examples and thought starters here, not theory.
>
> 杰：在大学，伦理哲学是我的主要工作（要不是我退学走入社会，如果我完成它，可能它会是我论文的主题）。基于我花费那些年研究伦理，我可以告诉你这些信息：没人真正同意什么是对什么是错，他们是否存在，怎样发现他们，那些人是好的和那些人是坏的，或别的任何东西。所以不要对理论报有太多期望！在这里我们将聚焦示例和思想先驱，而不是理论。

In answering the question ["What Is Ethics"](https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/what-is-ethics/), The Markkula Center for Applied Ethics says that the term refers to:

在回答[“什么是伦理”](https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/what-is-ethics/)这个问题中，应用伦理马克库拉中心认为这个术语要参照：

- Well-founded standards of right and wrong that prescribe what humans ought to do
- The study and development of one's ethical standards.
- 理由充分的对错标准，规定人类应该什么
- 研究和发展个人伦理标准。

There is no list of right answers. There is no list of do and don't. Ethics is complicated, and context-dependent. It involves the perspectives of many stakeholders. Ethics is a muscle that you have to develop and practice. In this chapter, our goal is to provide some signposts to help you on that journey.

这里没有正确答案列表。没有做和不做的列表。伦理是复杂的，并依赖环境。它涉及许多利益相关者的观点。伦理是一种你不得不发展和实践的力量。在这一章节，我们的目标是提供一些路标以帮助你完成这一旅程。

Spotting ethical issues is best to do as part of a collaborative team. This is the only way you can really incorporate different perspectives. Different people's backgrounds will help them to see things which may not be obvious to you. Working with a team is helpful for many "muscle-building" activities, including this one.

发现伦理问题的方法是最好做为合作团队的一部分。这是你能实际吸纳不同观点的唯一方法。不同的人员背景会帮助他们看到对你来说可能并不明显的事情。同一个团队工作有帮与那些“能力塑造”活动，也包括这个。

This chapter is certainly not the only part of the book where we talk about data ethics, but it's good to have a place where we focus on it for a while. To get oriented, it's perhaps easiest to look at a few examples. So, we picked out three that we think illustrate effectively some of the key topics.

本章当然不是本书唯一部分讲解数据伦理，但它很好的提供了让我们暂时可以聚焦它的地方。明确定位，看一些示例可能是最容易理解的。所我们挑选出了三个我们认为能够有效说明关键主题的一些东西。

## Key Examples for Data Ethics

## 数据伦理的关键示例

We are going to start with three specific examples that illustrate three common ethical issues in tech:

我们会从三个特定例子开始，这些示例说明了在技术中三个常见的伦理问题：

1. *Recourse processes*—Arkansas's buggy healthcare algorithms left patients stranded.
2. *Feedback loops*—YouTube's recommendation system helped unleash a conspiracy theory boom.
3. *Bias*—When a traditionally African-American name is searched for on Google, it displays ads for criminal background checks.

1. *追索程序* — 阿肯色州不成熟的医疗保健算法让病人陷入困境。
2. *反馈循环* — YouTube 的推荐系统助长了阴谋论浪潮。
3. *偏见* — 在谷歌上搜索一个传统非裔美国人时，会显示犯罪背景调查的广告。

In fact, for every concept that we introduce in this chapter, we are going to provide at least one specific example. For each one, think about what you could have done in this situation, and what kinds of obstructions there might have been to you getting that done. How would you deal with them? What would you look out for?

实际上，对于在本章节我们介绍每个概念，我们将会提供至少一个特定例子。对于每个人，思考在这种情况下你能做什么，及你完成这个工作可能会有什么样的障碍。你应该如何处理他们？你应该注意什么？

### Bugs and Recourse: Buggy Algorithm Used for Healthcare Benefits

### 缺陷和追索：不成熟算法被用于医疗保健福利

The Verge investigated software used in over half of the US states to determine how much healthcare people receive, and documented their findings in the article ["What Happens When an Algorithm Cuts Your Healthcare"](https://www.theverge.com/2018/3/21/17144260/healthcare-medicaid-algorithm-arkansas-cerebral-palsy). After implementation of the algorithm in Arkansas, hundreds of people (many with severe disabilities) had their healthcare drastically cut. For instance, Tammy Dobbs, a woman with cerebral palsy who needs an aid to help her to get out of bed, to go to the bathroom, to get food, and more, had her hours of help suddenly reduced by 20 hours a week. She couldn’t get any explanation for why her healthcare was cut. Eventually, a court case revealed that there were mistakes in the software implementation of the algorithm, negatively impacting people with diabetes or cerebral palsy. However, Dobbs and many other people reliant on these healthcare benefits live in fear that their benefits could again be cut suddenly and inexplicably.

 Verge 调查超过美国一半的州所使用的软件，以查明人们接受了多少医疗保健，并在文章[“当算法削减你的医疗保健时会发生什么”](https://www.theverge.com/2018/3/21/17144260/healthcare-medicaid-algorithm-arkansas-cerebral-palsy)中记录了他们的调查。在阿肯色州实施了算法后，数百人（一些有严重的残疾）的医疗保健被大幅削减。例如，塔米·多布斯，一名患有大脑性麻痹的女士，她需要一项救助以帮助她离开床、进入浴室、进餐，及更多的帮助，然而她每周的救助时间突然被缩减了20 小时。她无法获得任何关于医疗保健时间被缩减的解释。最终，一个司法案例揭露了在算法软件实施中所存在的错误。它对于糖尿病和大脑性麻痹人群有负面影响。不管怎样，多布斯和一些其他依赖医疗保健福利的人，生活在他们的福利被突然再次缩减而没有解释的恐惧中。

### Feedback Loops: YouTube's Recommendation System

### 反馈循环：YouTube 的推荐系统

Feedback loops can occur when your model is controlling the next round of data you get. The data that is returned quickly becomes flawed by the software itself.

反馈循环在你的模型正在控制你所获取的下一轮数据时会发生。通过软件自身使得快速返回的数据具有缺陷。

For instance, YouTube has 1.9 billion users, who watch over 1 billion hours of YouTube videos a day. Its recommendation algorithm (built by Google), which was designed to optimize watch time, is responsible for around 70% of the content that is watched. But there was a problem: it led to out-of-control feedback loops, leading the *New York Times* to run the headline ["YouTube Unleashed a Conspiracy Theory Boom. Can It Be Contained?"](https://www.nytimes.com/2019/02/19/technology/youtube-conspiracy-stars.html). Ostensibly recommendation systems are predicting what content people will like, but they also have a lot of power in determining what content people even see.

例如，YouTube 有 19 亿用户，一天观看YouTube 视频总时长超过 10 亿小时。由谷歌构建的YouTube 推荐算法被设计的目的是优化观看时间，此算法对观看的大约 70%的内容负有责任。但这里有一个问题：它依赖于失去控制的反馈循环。以至于*纽约时报*头版头条发布了[“YouTube助长了阴谋伦浪潮。它能包含吗？”](https://www.nytimes.com/2019/02/19/technology/youtube-conspiracy-stars.html)。表面上推荐系统预测人们会喜欢什么内容，但他们也有很大的权利决定什么内容人们可以看。

### Bias: Professor Lantanya Sweeney "Arrested"

Dr. Latanya Sweeney is a professor at Harvard and director of the university's data privacy lab. In the paper ["Discrimination in Online Ad Delivery"](https://arxiv.org/abs/1301.6822) (see <lantanya_arrested>) she describes her discovery that Googling her name resulted in advertisements saying "Latanya Sweeney, arrested?" even though she is the only known Latanya Sweeney and has never been arrested. However when she Googled other names, such as "Kirsten Lindquist," she got more neutral ads, even though Kirsten Lindquist has been arrested three times.

兰塔尼亚·斯威尼博士是哈佛大学教授及大学的数据隐私实验室负责人。在论文[“在线广告投放中的歧视”](https://arxiv.org/abs/1301.6822)（下图谷歌搜索引擎显示了兰塔尼亚·斯威尼教授的拘捕记录）中她描述了她的发现，尽管她是唯一已知的兰塔尼亚·斯威尼并从来没有被拘捕过，通过谷歌搜索她的名字，在广告中的结果是“兰塔尼亚·斯威尼，被拘捕？”。然而当她用谷歌搜索其它人的名字（例如“克里斯汀•林奎斯特”），即使克里斯汀•林奎斯特已经被拘捕 3 次，但她却收到了更加中立的广告信息。



<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image1.png" id="lantanya_arrested" caption="Google search showing ads about Professor Lantanya Sweeney's arrest record" alt="Screenshot of google search showing ads about Professor Lantanya Sweeney's arrest record" width="400">
  </p>
  <p align="center">图：谷歌搜索引擎显示了兰塔尼亚·斯威尼教授的拘捕记录</p>
</div>

Being a computer scientist, she studied this systematically, and looked at over 2000 names. She found a clear pattern where historically Black names received advertisements suggesting that the person had a criminal record, whereas, white names had more neutral advertisements.

做为一名计算机科学家，她有计划的做了研究，并查看了超过 2000 个人名。她发现了一个很清晰的模式：历史上黑人名会收到这个人已经有犯罪记录的广告建议，然而白人名则会有更加中立的广告。

This is an example of bias. It can make a big difference to people's lives—for instance, if a job applicant is Googled it may appear that they have a criminal record when they do not.

这是一个偏见的示例。它能给人的生活制造巨大的麻烦。例如，如果应聘者在谷歌上被搜索显示他们有犯罪记录，实际上他们并没有时。

### Why Does This Matter?

### 为什么这么重要？

One very natural reaction to considering these issues is: "So what? What's that got to do with me? I'm a data scientist, not a politician. I'm not one of the senior executives at my company who make the decisions about what we do. I'm just trying to build the most predictive model I can."

思考这个问题的一个很正常的反应：“所以呢？那和我有什么关系？我是只一句数据科学家，并不是政治家。在公司我不是一名决策我们做什么的资深经营者。我只是尽我所有去构建最有预见性的模型”

These are very reasonable questions. But we're going to try to convince you that the answer is that everybody who is training models absolutely needs to consider how their models will be used, and consider how to best ensure that they are used as positively as possible. There are things you can do. And if you don't do them, then things can go pretty badly.

这些是很合理的问题。但我们要尝试说服你，答案是每个训练模型的人绝对需要去考虑他们模型的用途，并考虑如何最大限度确保他们尽可能用于正面的方向。这些事情你能够做的，如果你不做，事情会变的非常糟糕。

One particularly hideous example of what happens when technologists focus on technology at all costs is the story of IBM and Nazi Germany. In 2001, a Swiss judge ruled that it was not unreasonable "to deduce that IBM's technical assistance facilitated the tasks of the Nazis in the commission of their crimes against humanity, acts also involving accountancy and classification by IBM machines and utilized in the concentration camps themselves."

当工程师不惜一切代价聚焦于技术时发生了一件尤为骇人听闻的事情：IBM 和德国纳粹的故事。在 2001 年，一名法官的裁决不无道理“推定 IBM 的技术加快了纳粹反人类委员会的任务，行为也涉及到通过 IBM 的机器进行会计记录和分类，并利用在他们自己的集中营”

IBM, you see, supplied the Nazis with data tabulation products necessary to track the extermination of Jews and other groups on a massive scale. This was driven from the top of the company, with marketing to Hitler and his leadership team. Company President Thomas Watson personally approved the 1939 release of special IBM alphabetizing machines to help organize the deportation of Polish Jews. Pictured in <meeting> is Adolf Hitler (far left) meeting with IBM CEO Tom Watson Sr. (second from left), shortly before Hitler awarded Watson a special “Service to the Reich” medal in 1937.

正如你看到的，IBM 提供给纳粹数据表格产品，它是追踪大规模犹太人和其它团队灭绝所必须的。这是由公司高层推动的营销希特勒和他的领导团队。公司主席汤姆·沃森个人批准 1939 年发布的特殊IBM 字母拼写机以帮助组织波兰犹太人驱逐出境。下图拍摄的照片是阿道夫·希特勒（左一）会见IBM首席执行官汤姆·沃森（左二）。之前早些时候，在 1937 年阿道夫给沃森颁布了一个特别的“服务帝国”勋章。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image2.png" id="meeting" caption="IBM CEO Tom Watson Sr. meeting with Adolf Hitler" alt="A picture of IBM CEO Tom Watson Sr. meeting with Adolf Hitler" width="400">
  </p>
  <p align="center">图：IBM首席执行官汤姆·沃森会见阿道夫·希特勒</p>
</div>

But this was not an isolated incident—the organization's involvement was extensive. IBM and its subsidiaries provided regular training and maintenance onsite at the concentration camps: printing off cards, configuring machines, and repairing them as they broke frequently. IBM set up categorizations on its punch card system for the way that each person was killed, which group they were assigned to, and the logistical information necessary to track them through the vast Holocaust system. IBM's code for Jews in the concentration camps was 8: some 6,000,000 were killed. Its code for Romanis was 12 (they were labeled by the Nazis as "asocials," with over 300,000 killed in the *Zigeunerlager*, or “Gypsy camp”). General executions were coded as 4, death in the gas chambers as 6.

但这并不是一个孤立事件，而是有组织的广泛参与。IBM 和它的子公司提供定期训练和在集中营的现场运维：打印卡片、配置机器，及在机器频繁故障时进行维修。IBM 在它的打孔卡系统上建立了分类用于区分每一个人被杀的方法，他们被分配到那个团体，和通过大屠杀系统跟踪他们所必须的物流信息。在集中营犹太人的 IBM 编码是 8：大约六百万人被杀。吉普赛人的代码是 12（他们被纳粹标注为“自私的”，超过三十万人在*Zigeunerlager*或“吉普赛人集中营”被杀）。普通的处决代码是 4，在毒气室被杀是 6。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image3.jpeg" id="punch_card" caption="A punch card used by IBM in concentration camps" alt="Picture of a punch card used by IBM in concentration camps" width="600">
  </p>
  <p align="center">图： 在集中营 IBM 所使用的打孔卡</p>
</div>