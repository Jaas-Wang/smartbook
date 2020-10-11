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