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

