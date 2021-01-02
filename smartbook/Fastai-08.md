# Collaborative Filtering Deep Dive

# 协同过滤深度研究

One very common problem to solve is when you have a number of users and a number of products, and you want to recommend which products are most likely to be useful for which users. There are many variations of this: for example, recommending movies (such as on Netflix), figuring out what to highlight for a user on a home page, deciding what stories to show in a social media feed, and so forth. There is a general solution to this problem, called *collaborative filtering*, which works like this: look at what products the current user has used or liked, find other users that have used or liked similar products, and then recommend other products that those users have used or liked.

一个要解决的常见问题是，当你有一定数量的用户和一定数量的产品，你希望推荐哪些产品最可能对哪些用户有用。对于这个问题有很多变化：例如，推荐电影（如网飞），计算出用户的网页上要高亮显示哪些内容，决定社交媒体中显示哪些故事，等等。对于这个问题这有一个常用解决方案称为*协同过滤*，它的工作方式：看当前用户已经使用或喜爱了哪些产品，寻找其它用户已经使用或喜爱的类似产品，然后推荐其它那些用户已经使用或喜爱的产品。

For example, on Netflix you may have watched lots of movies that are science fiction, full of action, and were made in the 1970s. Netflix may not know these particular properties of the films you have watched, but it will be able to see that other people that have watched the same movies that you watched also tended to watch other movies that are science fiction, full of action, and were made in the 1970s. In other words, to use this approach we don't necessarily need to know anything about the movies, except who like to watch them.

例如，在网飞上你可能已经看了很多充满动作的科幻电影，这些电影在1970年代制作的。网飞可能不知道这些你看过的电影的特定属性，但是它将能够查看和你观看相同电影的其它人，倾向看其它的充满动作的在1970年代制作的科幻电影。换句话说，用这个方法除了人们喜欢观看这些电影，我们不需要知道关于电影的任何信息，

There is actually a more general class of problems that this approach can solve, not necessarily involving users and products. Indeed, for collaborative filtering we more commonly refer to *items*, rather than *products*. Items could be links that people click on, diagnoses that are selected for patients, and so forth.



The key foundational idea is that of *latent factors*. In the Netflix example, we started with the assumption that you like old, action-packed sci-fi movies. But you never actually told Netflix that you like these kinds of movies. And Netflix never actually needed to add columns to its movies table saying which movies are of these types. Still, there must be some underlying concept of sci-fi, action, and movie age, and these concepts must be relevant for at least some people's movie watching decisions.



For this chapter we are going to work on this movie recommendation problem. We'll start by getting some data suitable for a collaborative filtering model.