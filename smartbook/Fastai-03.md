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

Of course, the project managers and engineers and technicians involved were just living their ordinary lives. Caring for their families, going to the church on Sunday, doing their jobs the best they could. Following orders. The marketers were just doing what they could to meet their business development goals. As Edwin Black, author of *IBM and the Holocaust* (Dialog Press) observed: "To the blind technocrat, the means were more important than the ends. The destruction of the Jewish people became even less important because the invigorating nature of IBM's technical achievement was only heightened by the fantastical profits to be made at a time when bread lines stretched across the world."

当然，参与的项目经理、工程师和技师只是过他们的普通生活。照顾他们的家人，周日去教堂，尽他们所能做他们的工作。按部就班。市场人员只是实现他们的商业目标。埃德温·布莱克，*IBM 和大屠杀*（Dialog 发行）的作者注意到：“犹太人的毁灭甚至变的不那么重要，因为 IBM 的技术成就令人愉悦本质只是当面包线延伸到全世界时通过每一次创造的奇幻利润被提升了”

Step back for a moment and consider: How would you feel if you discovered that you had been part of a system that ended up hurting society? Would you be open to finding out? How can you help make sure this doesn't happen? We have described the most extreme situation here, but there are many negative societal consequences linked to AI and machine learning being observed today, some of which we'll describe in this chapter.

退后一步并考虑：如果你发现你所完成的系统部分最终伤害了社会，你会任何感想？你愿意去发现吗？你会提供什么帮助以确保不会发现？我们在这里已经描述了最极端的情况，但现如今会发现有很多的负面社会后果与人工智能和机器学习相关，我们在本章节会谈论其中的一些内容。

It's not just a moral burden, either. Sometimes technologists pay very directly for their actions. For instance, the first person who was jailed as a result of the Volkswagen scandal, where the car company was revealed to have cheated on its diesel emissions tests, was not the manager that oversaw the project, or an executive at the helm of the company. It was one of the engineers, James Liang, who just did what he was told.

这也不仅仅是一个道德负担。有时技术专家要为他们的行为直接付出代价。例如，大众汽车公司被揭露在柴油排放测试上存在欺诈行为，詹姆斯·梁成为了大众丑闻入狱第一人，他不是监管项目的管理人员，也不是主导公司的经理，他只是众多工程师的一员，做了被告知的工作。

Of course, it's not all bad—if a project you are involved in turns out to make a huge positive impact on even one person, this is going to make you feel pretty great!

当然，也不全是坏的方面：如果你参与的一个项目对即使一个人产生了巨大的正面影响，这也会让你感觉非常非常的好。

Okay, so hopefully we have convinced you that you ought to care. But what should you do? As data scientists, we're naturally inclined to focus on making our models better by optimizing some metric or other. But optimizing that metric may not actually lead to better outcomes. And even if it *does* help create better outcomes, it almost certainly won't be the only thing that matters. Consider the pipeline of steps that occurs between the development of a model or an algorithm by a researcher or practitioner, and the point at which this work is actually used to make some decision. This entire pipeline needs to be considered *as a whole* if we're to have a hope of getting the kinds of outcomes we want.

好吧，希望我们已经说服你应该关心这些事情。但是你应该做什么呢？做为一名数据科学家，我们很自然的倾向关注通过优化指标或其它参数让我们的模型更优。但优化指标实际上不可能引出更好的结果。即使它帮助创建了更好的结果，几乎可以确定的是这也不是唯一要紧的事情。在通过研究人员或参与者对模型或算法的发展和这一工作实际用途决策的点之间思考发生的所有步骤通道。如果我们希望想要取得好的结果，这个通道需要被视为*一个整体*

Normally there is a very long chain from one end to the other. This is especially true if you are a researcher, where you might not even know if your research will ever get used for anything, or if you're involved in data collection, which is even earlier in the pipeline. But no one is better placed to inform everyone involved in this chain about the capabilities, constraints, and details of your work than you are. Although there's no "silver bullet" that can ensure your work is used the right way, by getting involved in the process, and asking the right questions, you can at the very least ensure that the right issues are being considered.

从一端到另一端正常来说这是很长的链条。如果你是一名研究人员这尤为真实，你可能甚至不知道你的研究是否有用，或在管道初期你是否参与了数据收集。但与你相比没人更合适去通知在这一链条上的每一个人关于你工作的能力、限制和细节。虽然没有“银弹”能确保你的工作被说于正确的用途，通过涉及的过程问正确的问题，你至少能够确保正确的问题会被思考。

Sometimes, the right response to being asked to do a piece of work is to just say "no." Often, however, the response we hear is, "If I don’t do it, someone else will." But consider this: if you’ve been picked for the job, you’re the best person they’ve found to do it—so if you don’t do it, the best person isn’t working on that project. If the first five people they ask all say no too, even better!

有时候，正好反馈被要求做的工作内容只需要说“不”。然而通常你听到的反馈是，“如果你不做，其它人会做。”但请考虑这名话：如果你已经获得了这个工作，你就是他们所能找到最适合做这个工作的人。所以如果你不做，在这个项目上没人更合适的人做。如果他们要求的头五个人也都说不的话，那就再好不过了！

## Integrating Machine Learning with Product Design

## 集成了机器学习的产品设计

Presumably the reason you're doing this work is because you hope it will be used for something. Otherwise, you're just wasting your time. So, let's start with the assumption that your work will end up somewhere. Now, as you are collecting your data and developing your model, you are making lots of decisions. What level of aggregation will you store your data at? What loss function should you use? What validation and training sets should you use? Should you focus on simplicity of implementation, speed of inference, or accuracy of the model? How will your model handle out-of-domain data items? Can it be fine-tuned, or must it be retrained from scratch over time?

做这一工作的大概原因是因为你希望它会用到某些地方。否则就只是在浪费你的时间。所以，让我们从你的工作成果最终会被使用的假设开始。现在，你正在搜集数据和开始你的模型，你正在做一些决策。你的数据存储到什么聚集级别？你应该采用什么损失函数？你应该使用什么样的验证集和训练集？你应该聚焦在简单实施，快速推理或模型的精度？你的模型怎么处理域外数据？它能被微调吗，或随着时间的推移它必定要被再训练吗？

These are not just algorithm questions. They are data product design questions. But the product managers, executives, judges, journalists, doctors… whoever ends up developing and using the system of which your model is a part will not be well-placed to understand the decisions that you made, let alone change them.

这里没有任何算法问题。他们只是一些数据产品设计问题。但产品经理，经营决策者，法官，新闻记者，医生...无论谁最终开发和使用的系统涉及到你的模型，他们所处的位置都不可能很好的理解你所做的决策，更何况改变他们。

For instance, two studies found that Amazon’s facial recognition software produced [inaccurate](https://www.nytimes.com/2018/07/26/technology/amazon-aclu-facial-recognition-congress.html) and [racially biased](https://www.theverge.com/2019/1/25/18197137/amazon-rekognition-facial-recognition-bias-race-gender) results. Amazon claimed that the researchers should have changed the default parameters, without explaining how this would have changed the biased results. Furthermore, it turned out that [Amazon was not instructing police departments](https://gizmodo.com/defense-of-amazons-face-recognition-tool-undermined-by-1832238149) that used its software to do this either. There was, presumably, a big distance between the researchers that developed these algorithms and the Amazon documentation staff that wrote the guidelines provided to the police. A lack of tight integration led to serious problems for society at large, the police, and Amazon themselves. It turned out that their system erroneously matched 28 members of congress to criminal mugshots!  (And the Congresspeople wrongly matched to criminal mugshots were disproportionately people of color, as seen in <congressmen>.)

例如，两名学者发现亚马逊公司的面部识别软件会产生不精准和种族偏见的结果。亚马逊声称研究人员应该修改默认参数，但没有解释如何改变偏见的结果。此外，结果是[亚马逊没有指导警察部门](https://gizmodo.com/defense-of-amazons-face-recognition-tool-undermined-by-1832238149) 用这个软件时也要做这个事情。可能在研究人员开发算法和亚马逊文书人员编写提供给警察的指引之间还有很大的距离。缺乏紧密的整合对于整个社会、警察和亚马逊自身会导致一系列问题。结果是他们的系统把 28 名国会议员错误的匹配为罪犯的面部照片！（把国会议员错误的匹配到罪犯的面部照片大多是有色人种，见下图<国会议员>）

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image4.png" id="congressmen" caption="Congresspeople matched to criminal mugshots by Amazon software" alt="Picture of the congresspeople matched to criminal mugshots by Amazon software, they are disproportionatedly people of color" width="500">
  </p>
  <p align="center">图： 国会议员</p>
</div>

Data scientists need to be part of a cross-disciplinary team. And researchers need to work closely with the kinds of people who will end up using their research. Better still is if the domain experts themselves have learned enough to be able to train and debug some models themselves—hopefully there are a few of you reading this book right now!

数据学科家需要做为跨学科团队的一份子。并且研究人员需要和那些将最终使用他们模型的人紧密工作。最好的是如果领域专家学到到足够的知识，能够自己去训练和调试一些模型，希望你们中一些人现在正在阅读本书。

The modern workplace is a very specialized place. Everybody tends to have well-defined jobs to perform. Especially in large companies, it can be hard to know what all the pieces of the puzzle are. Sometimes companies even intentionally obscure the overall project goals that are being worked on, if they know that their employees are not going to like the answers. This is sometimes done by compartmentalising pieces as much as possible.

现代的工作场所是一个非常特殊的地方。很一个人都倾向执行有明确定义的工作。尤其在大公司，几乎很难知道工作任务的所有部分。如果公司知道他们的员工有可能不喜欢某个工作目标，有时候公司甚至会故意模糊化正在工作中的项目总体目标。有时候通过尽可能的碎片化来完成项目。

In other words, we're not saying that any of this is easy. It's hard. It's really hard. We all have to do our best. And we have often seen that the people who do get involved in the higher-level context of these projects, and attempt to develop cross-disciplinary capabilities and teams, become some of the most important and well rewarded members of their organizations. It's the kind of work that tends to be highly appreciated by senior executives, even if it is sometimes considered rather uncomfortable by middle management.

从别一方面说，我们不是说这是一件很容易的事情。它很难，真的很难。我们不得不尽全力做到最好。我们经常会看到一些项目高级别背景的干系人，尝试发展交叉学科能够和团队，变成他们组织中最重要且好评的成员。这类工作即使有时中层管理人员认为很不舒服，然而却能获得高级管理层的高度赞赏。

## Topics in Data Ethics

## 数据伦理的几个主题

Data ethics is a big field, and we can't cover everything. Instead, we're going to pick a few topics that we think are particularly relevant:

数据伦理是一个巨大的领域，我们无法覆盖每一件事情。所以我们会挑出一些我们认为高度相关的主题：

- The need for recourse and accountability
- Feedback loops
- Bias
- Disinformation
- 需要追索和问责
- 反馈循环
- 偏见
- 虚假信息

Let's look at each in turn.

让我们依次看每一部分。

### Recourse and Accountability

### 追索和问责

In a complex system, it is easy for no one person to feel responsible for outcomes. While this is understandable, it does not lead to good results. In the earlier example of the Arkansas healthcare system in which a bug led to people with cerebral palsy losing access to needed care, the creator of the algorithm blamed government officials, and government officials blamed those who implemented the software. NYU professor [Danah Boyd](https://www.youtube.com/watch?v=NTl0yyPqf3E) described this phenomenon: "Bureaucracy has often been used to shift or evade responsibility... Today's algorithmic systems are extending bureaucracy."

在一个复杂系统中，没有人感觉对结果负责是很容易的。虽然这是可以理解的，但它不会带来好的结果。在早先阿肯色州医疗保健系统，因系统缺陷导致具有大脑性麻痹人群失去必要看护的案例中，算法的创建人谴责政府官员，政府官员谴责那些实施软件的人。纽约大学教授[Danah Boyd](https://www.youtube.com/watch?v=NTl0yyPqf3E)这样描述这一现象：“官僚主义经常习惯于推卸和逃避责任... 今天的算法系统是官僚管理的延伸。”

An additional reason why recourse is so necessary is because data often contains errors. Mechanisms for audits and error correction are crucial. A database of suspected gang members maintained by California law enforcement officials was found to be full of errors, including 42 babies who had been added to the database when they were less than 1 year old (28 of whom were marked as “admitting to being gang members”). In this case, there was no process in place for correcting mistakes or removing people once they’d been added. Another example is the US credit report system: in a large-scale study of credit reports by the Federal Trade Commission (FTC) in 2012, it was found that 26% of consumers had at least one mistake in their files, and 5% had errors that could be devastating. Yet, the process of getting such errors corrected is incredibly slow and opaque. When public radio reporter [Bobby Allyn](https://www.washingtonpost.com/posteverything/wp/2016/09/08/how-the-careless-errors-of-credit-reporting-agencies-are-ruining-peoples-lives/) discovered that he was erroneously listed as having a firearms conviction, it took him "more than a dozen phone calls, the handiwork of a county court clerk and six weeks to solve the problem. And that was only after I contacted the company’s communications department as a journalist."

一个额外的原因，为什么追索是如何必须的，因为数据经常包含错误。审核及错误纠正机制非常关键。一个由加利福尼亚司法部门官员维护的推测帮派成员数据库里，发现其充满了错误，里面包含了42名婴儿，当他们不满一岁的时候被添加进了数据库（他们中的28人被 标记为“承认是帮派成员”）。在这一案例中，一旦被添加没有纠正错误或移除的流程。那一个是美国信用报告系统的示例：在2012年联邦贸易委员会（FTC）开展了大规模的信用报告研究，在26%的消费者文件中发现至少存在一个错误，并且5%的人含有摧毁性的错误。然而，纠正错误的过程是难以置信的慢和不透明。当公共广播记者[鲍比·艾琳](https://www.washingtonpost.com/posteverything/wp/2016/09/08/how-the-careless-errors-of-credit-reporting-agencies-are-ruining-peoples-lives/)发现，他被错误的列为拥有枪只定罪时，这让他“打了超过一打的电话，一个县法庭职员的杰作和六周的时间去解决这一问题。这一切还是我做为一个记者联系了公司的传播部门。”

As machine learning practitioners, we do not always think of it as our responsibility to understand how our algorithms end up being implemented in practice. But we need to.

做为机器学习参与者，我们并不是一直认为了解我们的算法在实践中最终被如何实施是我们的责任。但是我们需要去做。

### Feedback Loops

### 反馈循环

We explained in <chapter_intro> how an algorithm can interact with its enviromnent to create a feedback loop, making predictions that reinforce actions taken in the real world, which lead to predictions even more pronounced in the same direction. 
As an example, let's again consider YouTube's recommendation system. A couple of years ago the Google team talked about how they had introduced reinforcement learning (closely related to deep learning, but where your loss function represents a result potentially a long time after an action occurs) to improve YouTube's recommendation system. They described how they used an algorithm that made recommendations such that watch time would be optimized.

在<概述>这一章中我们解释了一个算法如何和它的环境交互以创建一个反馈循环，进行预测以增强在真实世界中的行为，从而在同一方向上使得预测更准确。做为一个实例，让我们再次思考YouTube的推荐系统。两年前谷歌团队宣城他们如何引入了增强学习（与深度学习很接近，只是在增强学习上你的损失函数代表的是一个行为发生后很长时间的潜在结果）以改善YouTube的推荐系统。他们描述如何利用一个算法做出推荐这样观看时间会获得优化。

However, human beings tend to be drawn to controversial content. This meant that videos about things like conspiracy theories started to get recommended more and more by the recommendation system. Furthermore, it turns out that the kinds of people that are interested in conspiracy theories are also people that watch a lot of online videos! So, they started to get drawn more and more toward YouTube. The increasing number of conspiracy theorists watching videos on YouTube resulted in the algorithm recommending more and more conspiracy theory and other extremist content, which resulted in more extremists watching videos on YouTube, and more people watching YouTube developing extremist views, which led to the algorithm recommending more extremist content... The system was spiraling out of control.

然而，人类倾向被有争议的内容所吸引。这意味着像阴谋论这种视频开始获得推荐系统越来越多的推荐。而且，事实证明对阴谋论有兴趣的那些人也是大量看在线视频的人。在YouTube上阴谋论观看视频数的增加导致算法推荐越来越多的阴谋念经和其它极端内容，这就导致在YouTube上更多的极端视频被观看，并且更多观看YouTube的人形成了极端的观点，这使的算法推荐更多的极端内容... 系统呈螺旋状的失去控制。

And this phenomenon was not contained to this particular type of content. In June 2019 the *New York Times* published an article on YouTube's recommendation system, titled ["On YouTube’s Digital Playground, an Open Gate for Pedophiles"](https://www.nytimes.com/2019/06/03/world/americas/youtube-pedophiles.html). The article started with this chilling story:

并且这一现象并不包含这一特定类型的内容。在2019年6月，*纽约时报*对于YouTube的推荐系统发表了一篇文章，题为[YouTube的游乐场，一扇对恋童癖敞开的大门](https://www.nytimes.com/2019/06/03/world/americas/youtube-pedophiles.html)。文章以这样信人心寒的故事开始：

> : Christiane C. didn’t think anything of it when her 10-year-old daughter and a friend uploaded a video of themselves playing in a backyard pool… A few days later… the video had thousands of views. Before long, it had ticked up to 400,000... “I saw the video again and I got scared by the number of views,” Christiane said. She had reason to be. YouTube’s automated recommendation system… had begun showing the video to users who watched other videos of prepubescent, partially clothed children, a team of researchers has found.
>
> ：当克里斯蒂安妮·C. 10岁的女儿和她的朋友上传了在后院游泳池玩耍的视频时时，克里斯蒂安妮·C.没感觉有什么特别的。几天后...这个视频已经有了数千的观看量。前不久，它已经上升到了40万...“我再看视频，我被观看数量吓到了”，克里斯蒂安妮说道。她有理由害怕。一个研究团队发现，YouTube的自动推荐系统...开始向已经观看其它少儿视频的用户展示这一视频，尤其是穿了衣服的孩子。

> : On its own, each video might be perfectly innocent, a home movie, say, made by a child. Any revealing frames are fleeting and appear accidental. But, grouped together, their shared features become unmistakable.
>
> ：每个视频可能都有任何问题，只是孩子们拍摄的家庭录像。任何展示的帧是短暂且显的偶然。但是把它们聚合在一起，他们共同的特征就显露无疑了。

YouTube's recommendation algorithm had begun curating playlists for pedophiles, picking out innocent home videos that happened to contain prepubescent, partially clothed children.

YouTube的推荐算法开始为恋童癖制作播放列表，挑选出包含少儿（尤其是穿衣儿童）无辜家庭的视频。

No one at Google planned to create a system that turned family videos into porn for pedophiles. So what happened?

在谷歌没有人计划建立一个把家庭视频转为儿童色情的系统。那到底发生了什么？

Part of the problem here is the centrality of metrics in driving a financially important system. When an algorithm has a metric to optimize, as you have seen, it will do everything it can to optimize that number. This tends to lead to all kinds of edge cases, and humans interacting with a system will search for, find, and exploit these edge cases and feedback loops for their advantage.

这里的部分问题是因为在推动一个经济性重要系统中指标是核心。下如你看到的，当一个算法有个指标要优化时，它将作出所有努力以能够优化那数值。这个倾向导致各种极端情况，及人与系统的交互行为都会被搜索、发现，为了他们利益并利用这些极端情况并反馈循环。

There are signs that this is exactly what has happened with YouTube's recommendation system. *The Guardian* ran an article called ["How an ex-YouTube Insider Investigated its Secret Algorithm"](https://www.theguardian.com/technology/2018/feb/02/youtube-algorithm-election-clinton-trump-guillaume-chaslot) about Guillaume Chaslot, an ex-YouTube engineer who created AlgoTransparency, which tracks these issues. Chaslot published the chart in <ethics_yt_rt>, following the release of Robert Mueller's "Report on the Investigation Into Russian Interference in the 2016 Presidential Election."

有一些迹象表明，YouTube的推荐系统发生了什么。*卫报*刊登了关于Guillaume Chaslot一篇名为["前YouTube内部人如何调查它的秘密算法"](https://www.theguardian.com/technology/2018/feb/02/youtube-algorithm-election-clinton-trump-guillaume-chaslot)，这名前YouTube工程师创建了人工智能透明化公司，用以追踪这些问题。罗伯特·穆勒发表了“对于俄罗斯干涉2016年总统大选的调查报告”，Chaslotd在<穆勒报告>中发布了图表。如下图所示。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image18.jpeg" id="ethics_yt_rt" caption="Coverage of the Mueller report" alt="Coverage of the Mueller report" width="500">
  </p>
  <p align="center">图：米勒报告</p>
</div>

Russia Today's coverage of the Mueller report was an extreme outlier in terms of how many channels were recommending it. This suggests the possibility that Russia Today, a state-owned Russia media outlet, has been successful in gaming YouTube's recommendation algorithm. Unfortunately, the lack of transparency of systems like this makes it hard to uncover the kinds of problems that we're discussing.

今日俄罗斯对穆勒报告的报道在推荐它的频道数量上是及其异常的。这表明今日俄罗斯这一俄罗斯国属新闻媒体在博弈YouTube的推荐系统方面已经取得成功。不幸的是，就想这样系统透明度的缺乏使得它很难揭露我们正在讨论这些问题。

One of our reviewers for this book, Aurélien Géron, led YouTube's video classification team from 2013 to 2016 (well before the events discussed here). He pointed out that it's not just feedback loops involving humans that are a problem. There can also be feedback loops without humans! He told us about an example from YouTube:

这本书的一个评论员奥雷利安·杰龙，他从2013年到2016年带领了YouTube的视频分类团队（这个事情以前发生的）。他指出，问题是它不仅仅涉及人的反馈循环。也有无人的反馈循环！他告诉了我们一个来自YouTube的例子：

> : One important signal to classify the main topic of a video is the channel it comes from. For example, a video uploaded to a cooking channel is very likely to be a cooking video. But how do we know what topic a channel is about? Well… in part by looking at the topics of the videos it contains! Do you see the loop? For example, many videos have a description which indicates what camera was used to shoot the video. As a result, some of these videos might get classified as videos about “photography.” If a channel has such a misclassified video, it might be classified as a “photography” channel, making it even more likely for future videos on this channel to be wrongly classified as “photography.” This could even lead to runaway virus-like classifications! One way to break this feedback loop is to classify videos with and without the channel signal. Then when classifying the channels, you can only use the classes obtained without the channel signal. This way, the feedback loop is broken.
>
> ：对视频的主题进行分类的一个重要标志是它来自的频道。例如，上传到烹饪频道的视频非常可能是一个烹饪视频。但是我们怎么知道频道的主题是什么？很好...依据看到的视频主题包含的内容！你看到循环了吗？例如，很多视频有用的是是什么品牌的摄像机拍摄的这个视频的描述。作为结果，这些视频是的一些可能会分类为“摄影”的视频。如果一个频道有这样一个错误分类视频，它可能被分类为“摄影”频道，使得它更可能对未来上传到本频道的视频错误的分类为“摄影”。这甚至能导致病毒样本分类失控！打破这一反馈循环的方法是通过无频道表示进行视频分类。然后正在分类频道时，你只能看到获取无频道标识的分类。这样，反馈循环就被打破了。

There are positive examples of people and organizations attempting to combat these problems. Evan Estola, lead machine learning engineer at Meetup, [discussed the example](https://www.youtube.com/watch?v=MqoRzNhrTnQ) of men expressing more interest than women in tech meetups. taking gender into account could therefore cause Meetup’s algorithm to recommend fewer tech meetups to women, and as a result, fewer women would find out about and attend tech meetups, which could cause the algorithm to suggest even fewer tech meetups to women, and so on in a self-reinforcing feedback loop. So, Evan and his team made the ethical decision for their recommendation algorithm to not create such a feedback loop, by explicitly not using gender for that part of their model. It is encouraging to see a company not just unthinkingly optimize a metric, but consider its impact. According to Evan, "You need to decide which feature not to use in your algorithm... the most optimal algorithm is perhaps not the best one to launch into production."

这里有一些人和组织尝试同这些问题战斗的例子。埃文·埃斯托拉带领机器学习工程师在聚会项目上，讨论男性表达出对于技术聚会比女性更有兴趣的[例子](https://www.youtube.com/watch?v=MqoRzNhrTnQ)。把性别做为考虑因素能够导致聚会算法推荐更少的技术聚会给女性，这就导致更少的女性找到和参加技术聚会，这就引发算法推荐更少的技术聚会给女性，以及诸如此的类自强化循环反馈。所以埃文和他的团队对于他们的推荐算法做出一个伦理决策，不去创建此类的循环反馈，明确不用性别做为他们模型的一部分。令人高兴的看到一个公司不仅仅只是欠考虑的去优化一个指标，而是去考虑它的影响。埃文说到，“你需要决定在你的算法中什么特征不能用...一个最优的算法可能不是在产品中所启用的最好的算法。”

While Meetup chose to avoid such an outcome, Facebook provides an example of allowing a runaway feedback loop to run wild. Like YouTube, it tends to radicalize users interested in one conspiracy theory by introducing them to more. As Renee DiResta, a researcher on proliferation of disinformation, [writes](https://www.fastcompany.com/3059742/social-network-algorithms-are-distorting-reality-by-boosting-conspiracy-theories):

当聚会公司选择避免这种结果的时候，脸书提供了一个允许失控循环反馈暴涨的例子。像YouTube一样，它倾向通过介绍更多人一种阴谋论以激发用户对它的兴趣。正如对虚假信息激增进行研究的研究员[蕾妮·迪瑞斯塔所写的](https://www.fastcompany.com/3059742/social-network-algorithms-are-distorting-reality-by-boosting-conspiracy-theories)：

> : Once people join a single conspiracy-minded [Facebook] group, they are algorithmically routed to a plethora of others. Join an anti-vaccine group, and your suggestions will include anti-GMO, chemtrail watch, flat Earther (yes, really), and "curing cancer naturally groups. Rather than pulling a user out of the rabbit hole, the recommendation engine pushes them further in."
>
> ：一旦用户加入一个阴谋思想团队[脸书]，他们的算法会发送其它很多的内容。加入一个反疫苗团队，然后你会看到包含反转基因、化学制品的手表、地球是平的（真是这样），和“癌症自愈团体”。不是把用户脱离困境，算法引擎而是把用户推的更远。

It is extremely important to keep in mind that this kind of behavior can happen, and to either anticipate a feedback loop or take positive action to break it when you see the first signs of it in your own projects. Another thing to keep in mind is *bias*, which, as we discussed briefly in the previous chapter, can interact with feedback loops in very troublesome ways.

记住这种行为能够发生非常重要，在你的项目中预料循环反馈或当你看到有类似迹象的第一时刻采取积极的行动去打破它。要记住的另一个事情是*偏见*，我们在之前的章节中已经讨论过，它会以非常麻烦的方式与循环反馈进行交互。

### Bias

### 偏见

Discussions of bias online tend to get pretty confusing pretty fast. The word "bias" means so many different things. Statisticians often think when data ethicists are talking about bias that they're talking about the statistical definition of the term bias. But they're not. And they're certainly not talking about the biases that appear in the weights and biases which are the parameters of your model!

在线偏见讨论趋于更混乱更快。“偏见”一词意味着很多不同的事情。统计学家经常认为当数据伦理谈论偏见时，他们正在谈的是一个偏移术语的统计定义。但是他们不是。确实不是讨论关于显示在你模型参数权重和偏移中的那些偏移！

What they're talking about is the social science concept of bias. In ["A Framework for Understanding Unintended Consequences of Machine Learning"](https://arxiv.org/abs/1901.10002) MIT's Harini Suresh and John Guttag describe six types of bias in machine learning, summarized in <bias> from their paper.

他们正在讨论的是社会科学偏见概念。在["一个理解非故意机器学习后果框架"](https://arxiv.org/abs/1901.10002)中，麻省理工大学的哈里尼·苏雷什和约翰·古塔格描述了机器学习中的六种偏见，下<偏见>总结图来自他们的论文。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/pipeline_diagram.svg" id="bias" caption="Bias in machine learning can come from multiple sources (courtesy of Harini Suresh and John V. Guttag)" alt="A diagram showing all sources where bias can appear in machine learning" width="650">
  </p>
  <p align="center">偏见</p>
</div>

We'll discuss four of these types of bias, those that we've found most helpful in our own work (see the paper for details on the others).

我们会讨论这些类型中的四个，我们认为这几个对我们的工作帮助最大（其它细节请看论文）。

#### Historical bias

#### 历史上的偏见

*Historical bias* comes from the fact that people are biased, processes are biased, and society is biased. Suresh and Guttag say: "Historical bias is a fundamental, structural issue with the first step of the data generation process and can exist even given perfect sampling and feature selection."

历史上的偏见来自人们是偏见的、过程是偏见的、社会也是偏见的这些因素。苏雷什和古塔格说：“对于数据生成过程的第一步历史偏见是一个基础的框架性问题，即使给出完美的样本和特征选择也能够存在”

For instance, here are a few examples of historical *race bias* in the US, from the *New York Times* article ["Racial Bias, Even When We Have Good Intentions"](https://www.nytimes.com/2015/01/04/upshot/the-measuring-sticks-of-racial-bias-.html) by the University of Chicago's Sendhil Mullainathan:

例如，这里有几个历史上美国*种族偏见*的例子，来自*纽约时报*由芝加哥大学森提尔·穆兰纳森发表的题为["种族偏见，及时当我们有好的意图的时候"](https://www.nytimes.com/2015/01/04/upshot/the-measuring-sticks-of-racial-bias-.html)的文章：

- When doctors were shown identical files, they were much less likely to recommend cardiac catheterization (a helpful procedure) to Black patients.
- When bargaining for a used car, Black people were offered initial prices $700 higher and received far smaller concessions.
- Responding to apartment rental ads on Craigslist with a Black name elicited fewer responses than with a white name.
- An all-white jury was 16 percentage points more likely to convict a Black defendant than a white one, but when a jury had one Black member it convicted both at the same rate.
- 当医生看到相同的档案，他们更不太可能给黑人推荐心脏插管术。
- 对一个二手车谈协议时，黑人会得到高于700美元的出价，并获得很小的让步。
- 在克雷格列表网站上有内容名字的公寓出租广告答复要远少于白人名字的答复。
- 一个全是白人的陪审团更有可能判黑人被告有罪，比白人被告高出16%，但是当陪审团中有一个黑人成员那它的有罪审判比率是相同的。

The COMPAS algorithm, widely used for sentencing and bail decisions in the US, is an example of an important algorithm that, when tested by [ProPublica](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing), showed clear racial bias in practice (<bail_algorithm>).

在美国广泛用于审判和保释决策的COMPAS算法是一个很重要的算法例子，当[ProPublica](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)测试这一算法时发现它在实践中有明显的种族偏见(下图<COMPAS算法结果>)

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image6.png" id="bail_algorithm" caption="Results of the COMPAS algorithm" alt="Table showing the COMPAS algorithm is more likely to give bail to white people, even if they re-offend more" width="600">
  </p>
  <p align="center">COMPAS算法结果</p>
</div>

Any dataset involving humans can have this kind of bias: medical data, sales data, housing data, political data, and so on. Because underlying bias is so pervasive, bias in datasets is very pervasive. Racial bias even turns up in computer vision, as shown in the example of autocategorized photos shared on Twitter by a Google Photos user shown in <<google_photos>>.

任何涉及到人的数据集都能有这种偏见：医学数据、销售数据、住宅数据、政治数据等等。因为偏见是如此广泛，在数据集中的偏见也是非常广泛。种族偏见甚至发生在计算机视觉中，推特上分享了一个谷歌照片用户在照片自动分类的例子中显示分类错误有点离谱，如下图<谷歌离谱的错误标注>。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image7.png" id="google_photos" caption="One of these labels is very wrong..." alt="Screenshot of the use of Google photos labeling a black user and her friend as gorillas" width="450">
  </p>
  <p align="center">谷歌离谱的错误标注</p>
</div>

Yes, that is showing what you think it is: Google Photos classified a Black user's photo with their friend as "gorillas"! This algorithmic misstep got a lot of attention in the media. “We’re appalled and genuinely sorry that this happened,” a company spokeswoman said. “There is still clearly a lot of work to do with automatic image labeling, and we’re looking at how we can prevent these types of mistakes from happening in the future.”

是的，它展示了你认为它是什么：谷歌照片一个黑人用户及他们的朋友分类为“猩猩”！这个算法错误在媒体上获得了广泛关注。“发生这样的事情我们是惊骇的并诚挚道歉，”一个公司发言人说到。“很显然对于自动图像标注还有很多工作要做，我们正在寻找能够如何阻止今后此类错误的发生。”

Unfortunately, fixing problems in machine learning systems when the input data has problems is hard. Google's first attempt didn't inspire confidence, as coverage by *The Guardian* suggested (<<gorilla-ban>>).

不幸的是，当输入数据有问题时，机器学习系统修复这些问题是很困难的。谷歌的首次尝试并没有激发人的信息，*卫报*的新闻报道如下图(<谷歌对问题的初次尝试>)

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image8.png" id="gorilla-ban" caption="Google's first response to the problem" alt="Pictures of a headlines from the Guardian, showing Google removed gorillas and other moneys from the possible labels of its algorithm" width="500">
  </p>
  <p align="center">谷歌对问题的初次尝试</p>
</div>

These kinds of problems are certainly not limited to just Google. MIT researchers studied the most popular online computer vision APIs to see how accurate they were. But they didn't just calculate a single accuracy number—instead, they looked at the accuracy across four different groups, as illustrated in <<face_recognition>>.

这些类型的错误不仅仅局限于谷歌。麻省理工大学研究人员对大多数主流在线计算机视觉应用程序接口进行了研究，去看他们的精度如何。但是他们没有计算单一的精度数值，而是通过四个不同的团队看了这个精度，正如下图<各种人脸识别系统对每一性别和种族的错误率>如示。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image9.jpeg" id="face_recognition" caption="Error rate per gender and race for various facial recognition systems" alt="Table showing how various facial recognition systems perform way worse on darker shades of skin and females" width="600">
  </p>
  <p align="center">各种人脸识别系统对每一性别和种族的错误率</p>
</div>

IBM's system, for instance, had a 34.7% error rate for darker females, versus 0.3% for lighter males—over 100 times more errors! Some people incorrectly reacted to these experiments by claiming that the difference was simply because darker skin is harder for computers to recognize. However, what actually happened was that, after the negative publicity that this result created, all of the companies in question dramatically improved their models for darker skin, such that one year later they were nearly as good as for lighter skin. So what this actually showed is that the developers failed to utilize datasets containing enough darker faces, or test their product with darker faces.

例如，IBM的系统对深肤色女性的错误率在34.7%，相对比浅肤色男性只有0.3%，错误对比超过了100多倍！很多人对这些实验有不正确的反应，只是简单的认为这种差异是因为深肤色对于计算机来说更难识别。然而，实际上发生了什么呢，在负面的公共媒体结果生成后，所有的这一问题的公司的模型对深色肤色有了戏剧化的改进，因此一年后他们几乎和浅肤色一样好了。所以这实际上反应的是开发者未能利用包含了足够深肤色面部的数据集，或者用深肤色面部对测试了他们的产品。

One of the MIT researchers, Joy Buolamwini, warned: "We have entered the age of automation overconfident yet underprepared. If we fail to make ethical and inclusive artificial intelligence, we risk losing gains made in civil rights and gender equity under the guise of machine neutrality."

一名麻省理工大学研究员乔·布兰维尼警告道：“我们已经进入了还没有准备好的过度自信自动化时代。如果我们未能建立伦理和不排斥人工智能，我们在机器中立的伪装之下会面临损失在人权和性别平衡方面所产生的收益的风险”

Part of the issue appears to be a systematic imbalance in the makeup of popular datasets used for training models. The abstract to the paper ["No Classification Without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World"](https://arxiv.org/abs/1711.08536) by Shreya Shankar et al. states, "We analyze two large, publicly available image data sets to assess geo-diversity and find that these data sets appear to exhibit an observable amerocentric and eurocentric representation bias. Further, we analyze classifiers trained on these data sets to assess the impact of these training distributions and find strong differences in the relative performance on images from different locales." <<image_provenance>> shows one of the charts from the paper, showing the geographic makeup of what was, at the time (and still are, as this book is being written) the two most important image datasets for training models.

部分问题显示出了在对用于模型训练的流行数据集的制作上是一个系统性的不平衡。在由什里娅·香卡等人编写的论文["无代表无分类：对于发展中国家评估开放数据集上的地理多样性问题"](https://arxiv.org/abs/1711.08536)摘要中描述，“我们分析了两个大型的公开获得的图像数据集以评估地理多样性，发现这些数据集显示了可观察到的大气中心和欧洲中心为代表的偏差。进而，我们分析了在这些数据集上训练的分类器以评估这些训练分布的影响，发现在不同地区在图片相关表现有巨大的差异。”论文中的一个插图<流行训练集图像来源>显示了地理构成是什么样子，在那时（在本书编写时还一直这个样子）两最重要的图像数据集用于训练模型。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image10.png" id="image_provenance" caption="Image provenance in popular training sets" alt="Graphs showing how the vast majority of images in popular training datasets come from the US or Western Europe" width="800">
  </p>
  <p align="center">流行训练集图像来源</p>
</div>

The vast majority of the images are from the United States and other Western countries, leading to models trained on ImageNet performing worse on scenes from other countries and cultures. For instance, research found that such models are worse at identifying household items (such as soap, spices, sofas, or beds) from lower-income countries. <<object_detect>> shows an image from the paper, ["Does Object Recognition Work for Everyone?"](https://arxiv.org/pdf/1906.02659.pdf) by Terrance DeVries et al. of Facebook AI Research that illustrates this point.

图像主要来自美国和其它西方国家，导致在ImageNet上训练的模型对于来自其它国家和文化的当地物品表现很糟糕。例如，研究发现这些模型在识别来自低收入国家家用物品（如肥皂、调料、沙发或床）方面很糟糕。由脸书人工智能研究院泰伦斯·德弗里斯等人编写的["对每个人目标识别做了什么工作？"](https://arxiv.org/pdf/1906.02659.pdf)论文插图说明了这一点，下图<目标识别>展示的内容来自这一论文。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image17.png" id="object_detect" caption="Object detection in action" alt="Figure showing an object detection algorithm performing better on western products" width=500 >
  </p>
  <p align="center">目标识别</p>
</div>

In this example, we can see that the lower-income soap example is a very long way away from being accurate, with every commercial image recognition service predicting "food" as the most likely answer!

在这个例子里，我们能够看到低收入肥皂的精度差距很大，每一个商业图像识别服务把“食品”作为最有可能的答案！

As we will discuss shortly, in addition, the vast majority of AI researchers and developers are young white men. Most projects that we have seen do most user testing using friends and families of the immediate product development group. Given this, the kinds of problems we just discussed should not be surprising.

此外，我们将会简短的讨论一下，绝大多数人工智能研究人员和开发人员是年青的白人男性。绝大多数我们看到的项目做的用户测试是利用的即使开发小组的朋友和家人成员。因此，我们刚刚讨论的这些类型错误就不应该感到惊讶了。

Similar historical bias is found in the texts used as data for natural language processing models. This crops up in downstream machine learning tasks in many ways. For instance, it [was widely reported](https://nypost.com/2017/11/30/google-translates-algorithm-has-a-gender-bias/) that until last year Google Translate showed systematic bias in how it translated the Turkish gender-neutral pronoun "o" into English: when applied to jobs which are often associated with males it used "he," and when applied to jobs which are often associated with females it used "she" (<<turkish_gender>>).

类似的历史偏见也在用于自然语言处理模型的文本数据里发现了。以多种方式出现在了机器学习下流任务里。例如，一个广为报道的事例，直到去年谷歌翻译在翻译土耳其的中性性别代词“o”为英文时出现了系统偏见：当应用的工作经常与男性关联时它会用“他”，当经常与女性关联的工作时它会用“她”。如下图<在文本数据集里的性别偏见>

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image11.png" id="turkish_gender" caption="Gender bias in text data sets" alt="Figure showing gender bias in data sets used to train language models showing up in translations" width="600" >
  </p>
  <p align="center">在文本数据集里的性别偏见</p>
</div>

We also see this kind of bias in online advertisements. For instance, a [study](https://arxiv.org/abs/1904.02095) in 2019 by Muhammad Ali et al. found that even when the person placing the ad does not intentionally discriminate, Facebook will show ads to very different audiences based on race and gender. Housing ads with the same text, but picture either a white or a Black family, were shown to racially different audiences.

在线广告上我们也看到这类偏见。例如，在2019年由默罕默德·阿里等人开展的一项研究发现，即使在人们放置广告时并无有意识的歧视，脸书基于种族和性别把广告展示给非常不同的受众。具有相同文字的但是白人或黑人家庭照片的房产广告会展示给不同种族的受众。

#### Measurement bias

#### 计量偏见

In the paper ["Does Machine Learning Automate Moral Hazard and Error"](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) in *American Economic Review*, Sendhil Mullainathan and Ziad Obermeyer look at a model that tries to answer the question: using historical electronic health record (EHR) data, what factors are most predictive of stroke? These are the top predictors from the model:

在*美国经济观察*上的文章["机器学习会自动化道德危机和错误吗"](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) ，森提尔·穆兰纳森和齐亚德·奥伯迈耶查看一个模型尝试回答一个问题：利用历史电子健康记录（EHR）数据什么因素最能预测中风？这里有来自模型的顶级预测因素：

- Prior stroke
- Cardiovascular disease
- Accidental injury
- Benign breast lump
- Colonoscopy
- Sinusitis
- 之前中风
- 心血管疾病
- 事故伤害
- 良性乳房肿瘤
- 结肠镜
-  鼻窦炎

However, only the top two have anything to do with a stroke! Based on what we've studied so far, you can probably guess why. We haven’t really measured *stroke*, which occurs when a region of the brain is denied oxygen due to an interruption in the blood supply. What we’ve measured is who had symptoms, went to a doctor, got the appropriate tests, *and* received a diagnosis of stroke. Actually having a stroke is not the only thing correlated with this complete list—it's also correlated with being the kind of person who actually goes to the doctor (which is influenced by who has access to healthcare, can afford their co-pay, doesn't experience racial or gender-based medical discrimination, and more)! If you are likely to go to the doctor for an *accidental injury*, then you are likely to also go the doctor when you are having a stroke.

然而，只有头两个会引发中风！基于迄今为止我们的研究内容，我们大概能够猜出原因。我们没有真正的计量*中风*，当大脑的一个区域由于中止血液供应无法获得氧气就会发生中风。我们计量的是有了相关症状，去看医生，进行合适的检查并收到中风诊断。事实上患有中风不仅仅不是关联这一全部列表的唯一事情，它也关联实际上去看医生的那类人（影响谁获得理疗保健，能影响他们的共同支付，没有种族和基于性别的医疗歧视经历，等等）！如果因为一次*意外伤害*你可能去看医生，然后当你有中风的时候你也可能去看医生。

This is an example of *measurement bias*. It occurs when our models make mistakes because we are measuring the wrong thing, or measuring it in the wrong way, or incorporating that measurement into the model inappropriately.

这是一个*计量偏见*的例子。当我们的模型生发差错时它会发生，因为我们计量了错误的事情，或用了错误的方式计量它，或计量不合适的纳入了模型。

#### Aggregation bias

#### 聚集性偏见

*Aggregation bias* occurs when models do not aggregate data in a way that incorporates all of the appropriate factors, or when a model does not include the necessary interaction terms, nonlinearities, or so forth. This can particularly occur in medical settings. For instance, the way diabetes is treated is often based on simple univariate statistics and studies involving small groups of heterogeneous people. Analysis of results is often done in a way that does not take account of different ethnicities or genders. However, it turns out that diabetes patients have [different complications across ethnicities](https://www.ncbi.nlm.nih.gov/pubmed/24037313), and HbA1c levels (widely used to diagnose and monitor diabetes) [differ in complex ways across ethnicities and genders](https://www.ncbi.nlm.nih.gov/pubmed/22238408). This can result in people being misdiagnosed or incorrectly treated because medical decisions are based on a model that does not include these important variables and interactions.

当模型采纳所有恰当的因素在某种程度上没有集料数据时，或当一个模型未包含必要的交互项、非线性等诸如此类情况的时候就会发生*聚集偏见*。这特别会发生在医疗环境中。例如，糖尿病的治疗通常基于简单的相关混合小团体人群的单一变量统计和研究。某种意义上来说分析结果的完成通常不会考虑区分种族和性别，然而，事实证明糖尿病患者[不同的种族会有不同的并发症](https://www.ncbi.nlm.nih.gov/pubmed/24037313)，并且糖化血红蛋白水平（广泛用于诊断和监控糖尿病）[在不同的种族和性别间存在复杂的差异](https://www.ncbi.nlm.nih.gov/pubmed/22238408)。基于一个没有包含重要变量和交互项的模型进行医疗判断，这导致的结果是对病人会误诊或不正确的治疗。

#### Representation bias

#### 表示性偏见

The abstract of the paper ["Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting"](https://arxiv.org/abs/1901.09451) by Maria De-Arteaga et al. notes that there is gender imbalance in occupations (e.g., females are more likely to be nurses, and males are more likely to be pastors), and says that: "differences in true positive rates between genders are correlated with existing gender imbalances in occupations, which may compound these imbalances."

玛丽亚·德·阿尔泰加等人编写的论文["生物偏见：在高风险环境中语意表示偏见的一个实证研究"](https://arxiv.org/abs/1901.09451)摘要中提到职业方面的性别不平等（例如，女性更可能成为护士，而男性更可能做牧师），并认为：“真正确定的性别间比率差异与在职业中现存的性别不平等是有关联的，可能会加剧这些不平等。”

In other words, the researchers noticed that models predicting occupation did not only *reflect* the actual gender imbalance in the underlying population, but actually *amplified* it! This type of *representation bias* is quite common, particularly for simple models. When there is some clear, easy-to-see underlying relationship, a simple model will often simply assume that this relationship holds all the time. As <<representation_bias>> from the paper shows, for occupations that had a higher percentage of females, the model tended to overestimate the prevalence of that occupation.

另一方面，研究人员注意到，模型预测职业不仅仅*反映*了基于人口的真实性别不平等，实际上而是*放大*了它！这类*表示性偏见*非常普遍，尤其对于简单的模型。当有一些清晰的、显而易见的基础关系，一个简单的模型通常会假设所有的情况下都会保持这些关系。正如论文中显示的<表示性偏见>图，对于女性占比很高的职业，模型倾向高估职业的流行性。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image12.png" id="representation_bias" caption="Model error in predicting occupation plotted against percentage of women in said occupation" alt="Graph showing how model predictions overamplify existing bias" width="500">
  </p>
  <p align="center">表示性偏见</p>
</div>

For example, in the training dataset 14.6% of surgeons were women, yet in the model predictions only 11.6% of the true positives were women. The model is thus amplifying the bias existing in the training set.

例如，在训练集中外科女性医生有14.6%，而在模型预测中只有11.6%是真实确定是女性。因而模型放大了现在于训练集中的偏见。

Now that we've seen that those biases exist, what can we do to mitigate them?

现在，我们已经看了那些存在的偏见，那我们能做什么来缓解它们呢？

### Addressing different types of bias

### 处理不同类型的偏见

Different types of bias require different approaches for mitigation. While gathering a more diverse dataset can address representation bias, this would not help with historical bias or measurement bias. All datasets contain bias. There is no such thing as a completely debiased dataset. Many researchers in the field have been converging on a set of proposals to enable better documentation of the decisions, context, and specifics about how and why a particular dataset was created, what scenarios it is appropriate to use in, and what the limitations are. This way, those using a particular dataset will not be caught off guard by its biases and limitations.

不同类型的偏见需要用不同方法来缓解。在收集更多多样化的数据集的时候能够处理表示性偏见，这不会对历史偏见和计量偏见有帮助。所有的数据集都包含偏见。这并不是说这是一个完全低质量的数据集。在这一领域的很多研究人员已经汇聚在一起提出一系列建议，以更好的记录关于怎样和为什么一个特定数据集被建立的决策、背影和细节，什么样的场景适合采纳，限制的是什么。用这种方法，那些使用这一特点数据集的人就不会被数据集的偏见和限制搞的措手不及。

We often hear the question—"Humans are biased, so does algorithmic bias even matter?" This comes up so often, there must be some reasoning that makes sense to the people that ask it, but it doesn't seem very logically sound to us! Independently of whether this is logically sound, it's important to realize that algorithms (particularly machine learning algorithms!) and people are different. Consider these points about machine learning algorithms:

我们经常会听到这个问题—“人类有偏见，那么算法有偏见算个事吗？” 这如此常见，一定有一些说的通的原因以导致人们问这个问题，但它对我们来说并没有听起来那么符合逻辑！是否符合逻辑这外，认识到算法（特指机器学习算法）和人类有差异是很重要的。思考这些关于机器学习算法的点：

- *Machine learning can create feedback loops*:: Small amounts of bias can rapidly increase exponentially due to feedback loops.
- *Machine learning can amplify bias*:: Human bias can lead to larger amounts of machine learning bias.
- *Algorithms & humans are used differently*:: Human decision makers and algorithmic decision makers are not used in a plug-and-play interchangeable way in practice.
- *Technology is power*:: And with that comes responsibility.
- 机器学习能够创建反馈循环：由于反馈循环少量的偏见能呈几何倍数增长。
- 机器学习能够放大偏见：人类偏见能导致更大数据的机器学习偏见。
- 算法和人类用法不同：在实践中人类决策者和算法决策者并不是使用即插即用的交互方法。
- 技术就是力量：并且随之而来的是责任。

As the Arkansas healthcare example showed, machine learning is often implemented in practice not because it leads to better outcomes, but because it is cheaper and more efficient. Cathy O'Neill, in her book *Weapons of Math Destruction* (Crown), described the pattern of how the privileged are processed by people, whereas the poor are processed by algorithms. This is just one of a number of ways that algorithms are used differently than human decision makers. Others include:

正如阿肯色州医疗保健事例，机器学习经常在实践中被实施，不是因为它会有更好的结果，而是因为它更便宜及更有效率。在凯茜·奥尼尔的书中*数学破坏性武器*（皇冠出版）描述了特权是怎样通过人处理，而穷人是通过算法处理的模式。这只是许多算法使用不同与人类决策者的方法中的一种。其它还包括：

- People are more likely to assume algorithms are objective or error-free (even if they’re given the option of a human override).
- Algorithms are more likely to be implemented with no appeals process in place.
- Algorithms are often used at scale.
- Algorithmic systems are cheap.
- 人类更可能假设算法是公平或无错误的（即使他们给出的是人类控制的选择）。
- 算法更可能被实施在无吸引力过程的地方。
- 算法通常会被大规模使用。
- 算法系统是廉价的。

Even in the absence of bias, algorithms (and deep learning especially, since it is such an effective and scalable algorithm) can lead to negative societal problems, such as when used for *disinformation*.

即使不存在偏见，算法（尤其是深度学习，因为它是一个如此有效率和可扩展的算法）能够导致负面的社会问题，如果当使用虚假信息的时候。

### Disinformation

### 虚假信息

*Disinformation* has a history stretching back hundreds or even thousands of years. It is not necessarily about getting someone to believe something false, but rather often used to sow disharmony and uncertainty, and to get people to give up on seeking the truth. Receiving conflicting accounts can lead people to assume that they can never know whom or what to trust.

*虚假信息*可回溯数百甚至数千年的历史。它不必让人们相信事情是假的，而只需要经常去传播不和谐和不确定的信息，并让人们放弃寻找真相。接收冲突信息能够让人们假设他们永远都无法知道哪些人和什么事才是真的。

Some people think disinformation is primarily about false information or *fake news*, but in reality, disinformation can often contain seeds of truth, or half-truths taken out of context. Ladislav Bittman was an intelligence officer in the USSR who later defected to the US and wrote some books in the 1970s and 1980s on the role of disinformation in Soviet propaganda operations. In *The KGB and Soviet Disinformation* (Pergamon) he wrote, "Most campaigns are a carefully designed mixture of facts, half-truths, exaggerations, and deliberate lies."

一些人认为虚假信息主要是假信息或*假新闻*，但事实上，虚假信息经常能够包含真相的种子，或脱离背影的半真实信息。拉迪斯拉夫·比特曼曾是苏维埃社会主义共和国联盟的一名情报机构官员，后来叛逃到美国并基于在二十世纪70和80年代作为苏联鼓动宣传操作中虚假信息的作用编写了一些著作。在*克格勃和苏联虚假信息*（培格曼出版）一书中他写道，“大多数运动是精心设计的各种因素的混合体，半真实的、夸大的信息，及认真编造的谎言。”

In the US this has hit close to home in recent years, with the FBI detailing a massive disinformation campaign linked to Russia in the 2016 election. Understanding the disinformation that was used in this campaign is very educational. For instance, the FBI found that the Russian disinformation campaign often organized two separate fake "grass roots" protests, one for each side of an issue, and got them to protest at the same time! The [*Houston Chronicle*](https://www.houstonchronicle.com/local/gray-matters/article/A-Houston-protest-organized-by-Russian-trolls-12625481.php) reported on one of these odd events (<<texas>>).

近几年这已经影响到了美国家庭，美国联邦调查局公布一大量在2016年总统大选时与俄罗斯相关联的虚假信息运动。理解虚假信息被用于这样的运动是非常有教育意义的。例如，联邦调查局发现俄罗斯虚假信息运动通常会组织两个独立的虚假“平民”抗议，一方反对另一方的议题，并让他们在同一时间抗议！[休斯敦纪事报](https://www.houstonchronicle.com/local/gray-matters/article/A-Houston-protest-organized-by-Russian-trolls-12625481.php)报道了众多奇怪事件中的一个（如下图<德克萨斯之心团体组织的事件>）。

> : A group that called itself the "Heart of Texas" had organized it on social media—a protest, they said, against the "Islamization" of Texas. On one side of Travis Street, I found about 10 protesters. On the other side, I found around 50 counterprotesters. But I couldn't find the rally organizers. No "Heart of Texas." I thought that was odd, and mentioned it in the article: What kind of group is a no-show at its own event? Now I know why. Apparently, the rally's organizers were in Saint Petersburg, Russia, at the time. "Heart of Texas" is one of the internet troll groups cited in Special Prosecutor Robert Mueller's recent indictment of Russians attempting to tamper with the U.S. presidential election.
>
> ：一个自称“德克萨斯之心”的团地在社交媒体上组织了一个抗议游行。他们说反对德克萨斯的“伊斯兰化”。在特拉维斯街一侧我发现有大约10名抗议者。在街道另一侧有大约50名反对示威者。但是我没有发现集会组织者。没有“德克萨斯之心”。我认为这是奇怪的，并在文正中提到：什么类型的团队没有现身在他们自己的活动中？现在我知道为什么了。很显然，与此同时集会组织者在俄罗斯的圣彼得堡。“德克萨斯之心”是网络巨魔集团中的一个，特别检察官罗伯特·穆勒最近在起诉俄罗斯尝试篡改美国总统大选中引述了它。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/image13.png" id="texas" caption="Event organized by the group Heart of Texas" alt="Screenshot of an event organized by the group Heart of Texas" width="300">
  </p>
  <p align="center">德克萨斯之心团体组织的事件</p>
</div>

Disinformation often involves coordinated campaigns of inauthentic behavior. For instance, fraudulent accounts may try to make it seem like many people hold a particular viewpoint. While most of us like to think of ourselves as independent-minded, in reality we evolved to be influenced by others in our in-group, and in opposition to those in our out-group. Online discussions can influence our viewpoints, or alter the range of what we consider acceptable viewpoints. Humans are social animals, and as social animals we are extremely influenced by the people around us. Increasingly, radicalization occurs in online environments; influence is coming from people in the virtual space of online forums and social networks.

虚假信息通常涉及协调不真实行为的运动。例如，那些欺诈账户可以尝试让它看似一些人保持一个特定的观点。与此同时我们中的大多数认为我们自己是一个独立思考者，事实上逐步被在我们内部群体的其它人所影响，并反对那些我们群体之外的人。在线讨论能够影响我们的观点，或改变一系列我们认为能够同意的观点。人是社会型动物，做为社会型动物我们能够被我们周边的人极大的影响。进而，激进事件在线环境中发生。影响力来自那些在线论坛和社交网络的虚拟空间中的人。

Disinformation through autogenerated text is a particularly significant issue, due to the greatly increased capability provided by deep learning. We discuss this issue in depth when we delve into creating language models, in <<chapter_nlp>>.

由于通过深度学习提供的大幅增加的能力，通过自动生成文本生成的虚假信息是尤为重大的问题。当我们在<自然语言处理>章节中研究创建语言模型时我们深入讨论这一问题。

One proposed approach is to develop some form of digital signature, to implement it in a seamless way, and to create norms that we should only trust content that has been verified. The head of the Allen Institute on AI, Oren Etzioni, wrote such a proposal in an article titled ["How Will We Prevent AI-Based Forgery?"](https://hbr.org/2019/03/how-will-we-prevent-ai-based-forgery): "AI is poised to make high-fidelity forgery inexpensive and automated, leading to potentially disastrous consequences for democracy, security, and society. The specter of AI forgery means that we need to act to make digital signatures de rigueur as a means of authentication of digital content."

一个建议方法是开发一些数字签名形式，以一种连续的方法去实施它，并创建一个我们应该只相信那些被验证过的内容基准。艾伦人工智能研究院负责人奥伦·埃齐奥尼在一篇题为[“我们将如何阻止基于人工智能的伪造”](https://hbr.org/2019/03/how-will-we-prevent-ai-based-forgery)中写下这么一个建议：“人工智能有望既不昂贵又自动化的进行高保真伪造，致使对于民主、安全和社会造成潜在的灾难性后果。人工智能伪造的幽灵意味着我们需要行动起来，制造出必要的数字签名作为数字内容的一种认证手段。”

Whilst we can't hope to discuss all the ethical issues that deep learning, and algorithms more generally, brings up, hopefully this brief introduction has been a useful starting point you can build on. We'll now move on to the questions of how to identify ethical issues, and what to do about them.

同时我们不能寄希望可以讨论深度学习所有的伦理问题，算法更加通用，希望这个简短介绍的提出对于你起点的建立有所帮助。我们现在进入如果识别伦理的问题，并对他们应该做些什么。

## Identifying and Addressing Ethical Issues

## 识别及处理伦理问题

Mistakes happen. Finding out about them, and dealing with them, needs to be part of the design of any system that includes machine learning (and many other systems too). The issues raised within data ethics are often complex and interdisciplinary, but it is crucial that we work to address them.

错误发生了。找出并处理他们，这需要成为包括机器学习（一些其它系统也需要如此）在内的任何系统设计的一部分。数据伦理问题提出通过是复杂和跨学科，但我们努力处理他们是至关重要的。

So what can we do? This is a big topic, but a few steps towards addressing ethical issues are:

那么我们能做什么呢？这是一个很大的话题，但几个对于处理伦理问题的步骤是：

- Analyze a project you are working on.
- Implement processes at your company to find and address ethical risks.
- Support good policy.
- Increase diversity.
- 分析你正在工作的项目。
- 在你的公司实施一些发现并处理伦理风险的过程。
- 提供好的政策。
- 增加多样性。

Let's walk through each of these steps, starting with analyzing a project you are working on.

让我们对上述每一个步骤都过一下，开始分析你正在进行中的一个项目。

### Analyze a Project You Are Working On

### 分析一个你正在做的项目

It's easy to miss important issues when considering ethical implications of your work. One thing that helps enormously is simply asking the right questions. Rachel Thomas recommends considering the following questions throughout the development of a data project:

当思考你的工作所产生的伦理后果时容易忽视一些重要问题。简单问几个正确的问题能帮助放大一个事情。雷切尔·托马斯建议在开发一个数据项目的时候考虑下述几个问题：

- Should we even be doing this?
- What bias is in the data?
- Can the code and data be audited?
- What are the error rates for different sub-groups?
- What is the accuracy of a simple rule-based alternative?
- What processes are in place to handle appeals or mistakes?
- How diverse is the team that built it?
- 我们到底应该做这个事吗？
- 在数据中的偏见是什么？
- 不同的子团体的错误率是什么？
- 一个简单的基于规则的替代物的精确度是什么？
- 正确处理上述或错误的过程是什么？
- 如果多样化创造它的团队？

These questions may be able to help you identify outstanding issues, and possible alternatives that are easier to understand and control. In addition to asking the right questions, it's also important to consider practices and processes to implement.

这些问题可以帮助你识别显著的问题，并且对可能的替代方案更加容易理解和控制。此外问一些正确的问题，对于考虑实践和实施过程也是很重要的。

One thing to consider at this stage is what data you are collecting and storing. Data often ends up being used for different purposes than what it was originally collected for. For instance, IBM began selling to Nazi Germany well before the Holocaust, including helping with Germany’s 1933 census conducted by Adolf Hitler, which was effective at identifying far more Jewish people than had previously been recognized in Germany. Similarly, US census data was used to round up Japanese-Americans (who were US citizens) for internment during World War II. It is important to recognize how data and images collected can be weaponized later. Columbia professor [Tim Wu wrote](https://www.nytimes.com/2019/04/10/opinion/sunday/privacy-capitalism.html) that “You must assume that any personal data that Facebook or Android keeps are data that governments around the world will try to get or that thieves will try to steal.”

在这个阶段思考一个问题：你正在收集和存储什么数据。数据通常在最终用于不同于开始搜集它的目的。例如，IBM在早于大屠杀之前就开始给纳粹德国销售产品了，包括帮助德国1933年由阿道夫·希特勒提出的人口统计，相比之前德国识别的结果，有效率的识别出更多犹太人。类似的，二次世界大战期间，美国人口统计数据被用于围捕拘留日裔美国人（他们是美国公民）。识别收集怎么的数据和图片以后会被武器化是很重要的。哥伦比亚大学教授吴修铭写道“你必须假设任何脸书或安卓所拥有的个人数据，世界上的各国政府将会尝试获得或窃贼会尝试盗取。”

### Processes to Implement

### 实施过程

The Markkula Center has released [An Ethical Toolkit for Engineering/Design Practice](https://www.scu.edu/ethics-in-technology-practice/ethical-toolkit/) that includes some concrete practices to implement at your company, including regularly scheduled sweeps to proactively search for ethical risks (in a manner similar to cybersecurity penetration testing), expanding the ethical circle to include the perspectives of a variety of stakeholders, and considering the terrible people (how could bad actors abuse, steal, misinterpret, hack, destroy, or weaponize what you are building?).

马库拉中心发布了[一个面向工程/设计实践的伦理工具包](https://www.scu.edu/ethics-in-technology-practice/ethical-toolkit/) ，它包含了在你公司实施的具体作法，其中包括定期主动排查搜索伦理风险（类似于网络案例渗透测试方法），扩展伦理范围包括各种利益关联方的观点，及思考那些糟糕的人（对于你正在创建的内容，这些坏份子能够如何滥用，盗窃，错误的阐述，黑客行为，破坏，武器化？）。

Even if you don't have a diverse team, you can still try to pro-actively include the perspectives of a wider group, considering questions such as these (provided by the Markkula Center):

即使你没有一个多样化的团队，你能够持续尝试涵盖广泛团体的观点，例如考虑如下这些问题（由马库拉中心提供）：

- Whose interests, desires, skills, experiences, and values have we simply assumed, rather than actually consulted?
- Who are all the stakeholders who will be directly affected by our product? How have their interests been protected? How do we know what their interests really are—have we asked?
- Who/which groups and individuals will be indirectly affected in significant ways?
- Who might use this product that we didn’t expect to use it, or for purposes we didn’t initially intend?
- 你只是简单的假设了那些人的兴趣、欲望、技能、经验和价值，而不是实际的咨询过？
- 将直接影响我们产品的所有利益关联方都是谁？他们的利益怎么被保护？我们怎么知道他们真实的利益是什么，我们问过吗？
- 哪些团体和个人将会间接的产品显著影响？
- 谁可能使用这一产品，而我们并不希望他们使用，或我们最初没有想到的使用目的？

#### Ethical lenses

#### 伦理视角

Another useful resource from the Markkula Center is its [Conceptual Frameworks in Technology and Engineering Practice](https://www.scu.edu/ethics-in-technology-practice/conceptual-frameworks/). This considers how different foundational ethical lenses can help identify concrete issues, and lays out the following approaches and key questions:

别一个来自马库拉中心可利用的资源是[在技术和工程实施中的概念框架](https://www.scu.edu/ethics-in-technology-practice/conceptual-frameworks/)。它考虑的是不同基础伦理镜头能够怎样帮助识别具体的问题，并列示出下面的方法与关键问题：

- The rights approach:: Which option best respects the rights of all who have a stake?
- The justice approach:: Which option treats people equally or proportionately?
- The utilitarian approach:: Which option will produce the most good and do the least harm?
- The common good approach:: Which option best serves the community as a whole, not just some members?
- The virtue approach:: Which option leads me to act as the sort of person I want to be?
- 权力途径：哪种选择最能尊重所有利益相关者的权力？
- 公平途径：哪种选择威胁到人类平均或适合的比例？
- 实用途径：哪种选择会产生最好和做到最小伤害？
- 通用好方法：哪种选择最好的服务与整个团队，而不仅仅是少数？
- 道德途径：哪种选择使得我做出我想成为的那类人的行为？

Markkula's recommendations include a deeper dive into each of these perspectives, including looking at a project through the lenses of its *consequences*:

马库拉的建议潜入到每一个观点的更深层，包括通过它们的*结果*视角看一个项目：

- Who will be directly affected by this project? Who will be indirectly affected?
- Will the effects in aggregate likely create more good than harm, and what types of good and harm?
- Are we thinking about all relevant types of harm/benefit (psychological, political, environmental, moral, cognitive, emotional, institutional, cultural)?
- How might future generations be affected by this project?
- Do the risks of harm from this project fall disproportionately on the least powerful in society? Will the benefits go disproportionately to the well-off?
- Have we adequately considered "dual-use"?
- 哪些人会被这个项目直接影响？哪些人会被间接影响？
- 整体上可能产生大于伤害的更好结果吗，什么类型的好结果和伤害？
- 你会思虑所有相关的伤害/收益类型吗（心理、政治、环境、道德、认知、情绪、制度、文化）？
- 这个项目可能会怎样影响后代？
- 社会中的低层人民会不成比例的遭受来自这个项目的伤害风险吗？不成比例的收益会进入上层阶级吗？
- 我们充分考虑“双重用途”了吗？

The alternative lens to this is the *deontological* perspective, which focuses on basic concepts of *right* and *wrong*:

可选择视角是*责任*视点，聚焦于*对*和*错*的基础概念：

- What rights of others and duties to others must we respect?
- How might the dignity and autonomy of each stakeholder be impacted by this project?
- What considerations of trust and of justice are relevant to this design/project?
- Does this project involve any conflicting moral duties to others, or conflicting stakeholder rights? How can we prioritize these?
- 他人的什么权力和对他人的责任我们必须尊重？
- 通过这个项目每个利益相关者的尊严和自治可能被怎样影响？
- 信任和公正的考虑是怎么关联到这个设计/项目？
- 这个项目涉及哪些对他人的道德责任冲突，或对利益相关者的权利产生冲突？我们怎么确定优先级？

One of the best ways to help come up with complete and thoughtful answers to questions like these is to ensure that the people asking the questions are *diverse*.

帮助想出完整及详尽答案的一个最佳方法是像这样问一些问题，确保问问题的人是*多样化*的。

### The Power of Diversity

### 多样化的力量

Currently, less than 12% of AI researchers are women, according to [a study from Element AI](https://medium.com/element-ai-research-lab/estimating-the-gender-ratio-of-ai-researchers-around-the-world-81d2b8dbe9c3). The statistics are similarly dire when it comes to race and age. When everybody on a team has similar backgrounds, they are likely to have similar blindspots around ethical risks. The *Harvard Business Review* (HBR) has published a number of studies showing many benefits of diverse teams, including:

来自[元素人工智能的一项研究](https://medium.com/element-ai-research-lab/estimating-the-gender-ratio-of-ai-researchers-around-the-world-81d2b8dbe9c3)，当前人工智能研究员女性占比要少于12%。当看到种族和年龄时统计结果惊人的相似。当在团队里的每个人的背影类似时，他们可能对于伦理风险有类似的盲点。*哈佛商业评论*发表的许多研究显示了多样化团队很多收益，包括：

- ["How Diversity Can Drive Innovation"](https://hbr.org/2013/12/how-diversity-can-drive-innovation) 
- ["Teams Solve Problems Faster When They’re More Cognitively Diverse"](https://hbr.org/2017/03/teams-solve-problems-faster-when-theyre-more-cognitively-diverse) 
- ["Why Diverse Teams Are Smarter"](https://hbr.org/2016/11/why-diverse-teams-are-smarter), and 
- ["Defend Your Research: What Makes a Team Smarter? More Women"](https://hbr.org/2011/06/defend-your-research-what-makes-a-team-smarter-more-women) 
- [“如何多样化能够驱动创新”](https://hbr.org/2013/12/how-diversity-can-drive-innovation) 
- [“当团队认知更加多样化时他们解决问题会更快”](https://hbr.org/2017/03/teams-solve-problems-faster-when-theyre-more-cognitively-diverse) 
- [“为什么多样化的团队更加聪明”](https://hbr.org/2016/11/why-diverse-teams-are-smarter)，以及 
- [“保卫你的研究：什么使得一个团队更加聪明？更多女性”](https://hbr.org/2011/06/defend-your-research-what-makes-a-team-smarter-more-women) 

Diversity can lead to problems being identified earlier, and a wider range of solutions being considered. For instance, Tracy Chou was an early engineer at Quora. She [wrote of her experiences](https://qz.com/1016900/tracy-chou-leading-silicon-valley-engineer-explains-why-every-tech-worker-needs-a-humanities-education/), describing how she advocated internally for adding a feature that would allow trolls and other bad actors to be blocked. Chou recounts, “I was eager to work on the feature because I personally felt antagonized and abused on the site (gender isn’t an unlikely reason as to why)... But if I hadn’t had that personal perspective, it’s possible that the Quora team wouldn’t have prioritized building a block button so early in its existence.” Harassment often drives people from marginalized groups off online platforms, so this functionality has been important for maintaining the health of Quora's community.

多样化能够使得更容易识别问题，并更广范围的思考解决方案。例如，在Quora公司的早期工程师特蕾西·周[写下了她的经验](https://qz.com/1016900/tracy-chou-leading-silicon-valley-engineer-explains-why-every-tech-worker-needs-a-humanities-education/)，描述了她如何在内部倡议增加一个允许拉黑挑衅和其它坏份子的功能。周讲述到，“我渴望实现这一功能，因为我个人感觉在网站上被对立和虐待（至于为什么，性别是一个可能的原因）...但如果我没有个人的观点，可能Quora团队不会在诞生初期就优先创建一个拉黑按钮。”骚扰经常驱使边缘团体的人离开在线平台，所以这个功能对于维护Quora的社区健康已经很重要了。

A crucial aspect to understand is that women leave the tech industry at over twice the rate that men do, according to the [*Harvard Business Review*](https://www.researchgate.net/publication/268325574_By_RESEARCH_REPORT_The_Athena_Factor_Reversing_the_Brain_Drain_in_Science_Engineering_and_Technology) (41% of women working in tech leave, compared to 17% of men). An analysis of over 200 books, white papers, and articles found that the reason they leave is that “they’re treated unfairly; underpaid, less likely to be fast-tracked than their male colleagues, and unable to advance.”

一个关键因素要去理解，根据[哈佛商业评论](https://www.researchgate.net/publication/268325574_By_RESEARCH_REPORT_The_Athena_Factor_Reversing_the_Brain_Drain_in_Science_Engineering_and_Technology) （41%的女性在技术性休假，而男性是17%）女性离开技术产业的比率是男性的两倍多。分析了超过200本著作，白皮书和文章发现了她们离开的原因是“他们被不平等的对待：薪酬低，相比男同事她们不太可能被快速跟踪，不能够晋升。”

Studies have confirmed a number of the factors that make it harder for women to advance in the workplace. Women receive more vague feedback and personality criticism in performance evaluations, whereas men receive actionable advice tied to business outcomes (which is more useful). Women frequently experience being excluded from more creative and innovative roles, and not receiving high-visibility “stretch” assignments that are helpful in getting promoted. One study found that men’s voices are perceived as more persuasive, fact-based, and logical than women’s voices, even when reading identical scripts.

研究已经确认了许多因素使得女性在职场更难晋升。在表现评估方面女性获得更模糊的反馈和人格品论，然而男性收到与商业成果相关联的更有用的可行建议。女性经常经历被拒绝更有创造性和创新性的角色，且没有得到有助于获得晋升的清晰的“延伸”性工作分配。一项研究发现即使在阅读相同的脚本，男性的声音相比女性被认为更有说服力、更基于事实和更合乎逻辑。

Receiving mentorship has been statistically shown to help men advance, but not women. The reason behind this is that when women receive mentorship, it’s advice on how they should change and gain more self-knowledge. When men receive mentorship, it’s public endorsement of their authority. Guess which is more useful in getting promoted?

统计显示接收的指导有助于男性晋升，但是女性并不会。背后的原因是当女性接受指导时，给出的建议是她们应该改变和更有自知之明。当男性获得指导时，它是公开的赞同和他们的权威。试想哪种更能有益于获得晋升？

As long as qualified women keep dropping out of tech, teaching more girls to code will not solve the diversity issues plaguing the field. Diversity initiatives often end up focusing primarily on white women, even though women of color face many additional barriers. In [interviews](https://worklifelaw.org/publications/Double-Jeopardy-Report_v6_full_web-sm.pdf) with 60 women of color who work in STEM research, 100% had experienced discrimination.

当越来越多的优秀女性持续离开技术领域，教育更多女孩子编码将不能解决困扰这一领域的多样化问题。多样化举措通常最终会主要集中在白人女性，而有色人种女性会面临更多障碍。[访谈](https://worklifelaw.org/publications/Double-Jeopardy-Report_v6_full_web-sm.pdf)了60名从事科学、技术、工程与数学研究的有色人种女性，她们所有人都经历过歧视。

The hiring process is particularly broken in tech. One study indicative of the disfunction comes from Triplebyte, a company that helps place software engineers in companies, conducting a standardized technical interview as part of this process. They have a fascinating dataset: the results of how over 300 engineers did on their exam, coupled with the results of how those engineers did during the interview process for a variety of companies. The number one finding from [Triplebyte’s research](https://triplebyte.com/blog/who-y-combinator-companies-want) is that “the types of programmers that each company looks for often have little to do with what the company needs or does. Rather, they reflect company culture and the backgrounds of the founders.”

雇佣过程在技术领域尤为让人崩溃。一项来自三倍字节的研究指出了异常，这家公司帮助安置一些公司中的软件工程师，实施一个标准化的访谈是这一过程的一部分。他们有一个丰富的数据库：超过300名工程的测试成绩，还有在对各类公司访谈期间那些工程师的成绩。[三位字节的研究](https://triplebyte.com/blog/who-y-combinator-companies-want)发现排名第一的是“每个公司所寻找的一些类型程序员与公司需要的或所做的关系不大。从而，这反映出公司的文化和创始人的背影。”

This is a challenge for those trying to break into the world of deep learning, since most companies' deep learning groups today were founded by academics. These groups tend to look for people "like them"—that is, people that can solve complex math problems and understand dense jargon. They don't always know how to spot people who are actually good at solving real problems using deep learning.

对于那些尝试进入深度学习领域的人来说这是一个挑战，因为现今大多数公司的深度学习团队是通过学术派创建的。这些团队倾向去寻找“类似他们”一样的人：这些人能够解决复杂的数学问题和理解难懂的术语。他们并不能总会知道如何找出真的擅长利用深度学习解决实际问题的人。

This leaves a big opportunity for companies that are ready to look beyond status and pedigree, and focus on results!

这对于那些准备超越地位和血统并关注结果的公司留下了巨大的机会！