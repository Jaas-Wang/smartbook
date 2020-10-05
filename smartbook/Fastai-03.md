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
> 杰：在大学，伦理哲学是我的主要工作（要不是我退学走入社会，如果我完成它，可能它会是我论文的主题）。基于我花费那些年研究伦理，我可以告诉你这些信息：没人真正同意什么是对什么是错，他们是否存在，提出他们，那些人是好的和那些人是坏的，或更多的其它东西。所以不要对理论报有太多期望！在这里我们将聚焦示例和思想先驱，而不是理论。

