# NLP Deep Dive: RNNs

# 自然语言处理深潜：RNNs

In <chapter_intro> we saw that deep learning can be used to get great results with natural language datasets. Our example relied on using a pretrained language model and fine-tuning it to classify reviews. That example highlighted a difference between transfer learning in NLP and computer vision: in general in NLP the pretrained model is trained on a different task.

在<章节：概述>中，我们学习了使用自然语言数据集深度学习能够用来获取很好的结果。我们例子依赖于使用一个预训练语言模型并微调它来分类评论。那个例子强调了自然语言处理和计算机视觉迁移学习间的不同：通常在自然语言处理中预训练模型是在不同的任务上训练的。

What we call a language model is a model that has been trained to guess what the next word in a text is (having read the ones before). This kind of task is called *self-supervised learning*: we do not need to give labels to our model, just feed it lots and lots of texts. It has a process to automatically get labels from the data, and this task isn't trivial: to properly guess the next word in a sentence, the model will have to develop an understanding of the English (or other) language. Self-supervised learning can also be used in other domains; for instance, see ["Self-Supervised Learning and Computer Vision"](https://www.fast.ai/2020/01/13/self_supervised/) for an introduction to vision applications. Self-supervised learning is not usually used for the model that is trained directly, but instead is used for pretraining a model used for transfer learning.

我们调用的语言模型是一个已经被训练用于预测文本中下个词的内容的模型（之前阅读了很多文本）。这类任务被称为*自监督学习*：我们不需要给我们的模型提供标注，只需要喂给它很多很多的文本。它有一个从数据中自动获取标签的过程，这一任务不重要的是：正确猜测句子中的下个词，模型不得不会开发一个英文（或其它）语言的理解能力。自监督学习也能够被用于其它领域。例如，查看[自监督学习和计算机视觉](https://www.fast.ai/2020/01/13/self_supervised/)介绍了视觉方面的应用。自监督学习不常用于直接训练模型，相反用于预训练用于迁移学习。

> jargon: Self-supervised learning: Training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

> 术语：自监督学习：使用自变量中嵌入的标签训练一个模型，而不需要额外的标签。例如，训练一个模型来预测文本中的下个词。

The language model we used in <chapter_intro> to classify IMDb reviews was pretrained on Wikipedia. We got great results by directly fine-tuning this language model to a movie review classifier, but with one extra step, we can do even better. The Wikipedia English is slightly different from the IMDb English, so instead of jumping directly to the classifier, we could fine-tune our pretrained language model to the IMDb corpus and then use *that* as the base for our classifier.

我们在<章节：概述>中使用来分类IMDb评论的语言模型是在Wikipedia上预训练的。通过直接微调这个语言模型为一个影视评论分类器我们取得了很好的结果，但是有一个额外的步骤，我们甚至能够做的更好。Wikipedia英文与IMDb英文稍微有点差别，所以直接跳到了分类器，我们能够微调我们的预训练语言模型到IMDb语料库，然后用*这个*作为我们分类器的基础。

Even if our language model knows the basics of the language we are using in the task (e.g., our pretrained model is in English), it helps to get used to the style of the corpus we are targeting. It may be more informal language, or more technical, with new words to learn or different ways of composing sentences. In the case of the IMDb dataset, there will be lots of names of movie directors and actors, and often a less formal style of language than that seen in Wikipedia.

即使我们的语言模型知道我们在任务中使用的基础语言（例如，我们的预训练模型用的英文），它有助于习惯我们目标预料库的风格。这个目标库可能是更非正式的语言，或更多专业，有很多新词需要学习或不同句子的组合方式。在IMDb数据集中，有很多电影导演和演员的句子，且通常比在Wikipedia中看到的语言缺少那么一点正式性。

We already saw that with fastai, we can download a pretrained English language model and use it to get state-of-the-art results for NLP classification. (We expect pretrained models in many more languages to be available soon—they might well be available by the time you are reading this book, in fact.) So, why are we learning how to train a language model in detail?

我们已经用fastai看到了，我们能够下载一个预训练英文语言模型并用它取得自然语言处理分类的最先进的结果。（我们希望立刻获取更多语言的预训练模型。事实上，在你阅读本书期间，它们能够有效的获得。）那么，为什么我们学习如何训练一个语言模型的细节呢？

One reason, of course, is that it is helpful to understand the foundations of the models that you are using. But there is another very practical reason, which is that you get even better results if you fine-tune the (sequence-based) language model prior to fine-tuning the classification model. For instance, for the IMDb sentiment analysis task, the dataset includes 50,000 additional movie reviews that do not have any positive or negative labels attached. Since there are 25,000 labeled reviews in the training set and 25,000 in the validation set, that makes 100,000 movie reviews altogether. We can use all of these reviews to fine-tune the pretrained language model, which was trained only on Wikipedia articles; this will result in a language model that is particularly good at predicting the next word of a movie review.

当然，其中一个原因是它有助于理解我们正在使用的模型的基础。但还有另外一个特别的原因是，如果你在调整分类模型之前来微调（基于序列）语言模型你甚至会获得更好的结果。例如，对于IMDb情绪分析任务，数据集包含了 50,000 条附加的电影评论，其没有附上任何正面名负面的标签。因此有 25,000 条被标注的在训练集，25,000 在验证集，从而制作了总共 100,000 电影评论。我们能够这些所有的评论来微调预训练语言模型，这个预训练模型只在Wikipedia文章上做了训练。结果是语言模型在预测电影评论的下一个词上尤其的好。

This is known as the Universal Language Model Fine-tuning (ULMFit) approach. The [paper](https://arxiv.org/abs/1801.06146) showed that this extra stage of fine-tuning of the language model, prior to transfer learning to a classification task, resulted in significantly better predictions. Using this approach, we have three stages for transfer learning in NLP, as summarized in <ulmfit_process>.

这被称为通用语言模型微调（ULMFit）方法。[论文](https://arxiv.org/abs/1801.06146) 展示了这一额外语言模型微调的步骤，在迁移学习到一个分类任务前，会产生明显更好的预测。使用这一方法，我们对于自然语言处理中的迁移学习有三个步骤阶段，见下图<通用语言模型微调过程>所总结。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00027.png" alt="Diagram of the ULMFiT process" width="700" caption="The ULMFiT process" id="ulmfit_process" />
  </p>
  <p align="center">图：通用语言模型微调过程</p>
</div>

We'll now explore how to apply a neural network to this language modeling problem, using the concepts introduced in the last two chapters. But before reading further, pause and think about how *you* would approach this.

使用在前两章节介绍的概念，现在我们将要探索如何应用神经网络到这一语言建模问题上。但是在进一步阅读前，暂停并思考*你*会如何处理这个事情。

## Text Preprocessing

## 文本预处理

It's not at all obvious how we're going to use what we've learned so far to build a language model. Sentences can be different lengths, and documents can be very long. So, how can we predict the next word of a sentence using a neural network? Let's find out!

这不是毫无信息的我们将如何使用到目前为止我们已经学习的来创建一个言语模型。句子有不同的长度，文档可能是非常的长。所以，我们如何使用神经网络预测句子的下个词呢？让我们揭示出来吧！

We've already seen how categorical variables can be used as independent variables for a neural network. The approach we took for a single categorical variable was to:

 我们已经学习了分类变量如何能够以自变量被用于神经网络。我们认为对单一分类变量的这个方法是：

1. Make a list of all possible levels of that categorical variable (we'll call this list the *vocab*).
2. Replace each level with its index in the vocab.
3. Create an embedding matrix for this containing a row for each level (i.e., for each item of the vocab).
4. Use this embedding matrix as the first layer of a neural network. (A dedicated embedding matrix can take as inputs the raw vocab indexes created in step 2; this is equivalent to but faster and more efficient than a matrix that takes as input one-hot-encoded vectors representing the indexes.)

1. 生成一个分类变量所有可能等级的列表（我们称这个列表为*词汇表*）。
2. 在词汇表中用它的索引替换每个等级。
3. 为其创建一个嵌入矩阵，每个等级包含一行（即，对词汇表的每个项目）。
4. 使用这个嵌入矩阵作为神经网络的第一层。（一个专用嵌入矩阵能够把在第二步所创建的原始词汇表索引当作输入。这是等价的，但它比把独热编码矢量代表的索引当做输入的矩阵更快且更有效率。）

We can do nearly the same thing with text! What is new is the idea of a sequence. First we concatenate all of the documents in our dataset into one big long string and split it into words, giving us a very long list of words (or "tokens"). Our independent variable will be the sequence of words starting with the first word in our very long list and ending with the second to last, and our dependent variable will be the sequence of words starting with the second word and ending with the last word.

我们能够对文本做几乎相同的事情！新的内容是序列的想法。首先我们串联我们的数据集中所有文档为一个很长的字符串，并分割为词，这提供给我们一个很长的词列表（或“标记”）。我们的自变量是从我们长列表第一个词开始的序列词，且末尾是长列表倒数第二个词，我们的因变量是从列表第二个词开始，且结尾是列表的最后一个词。

Our vocab will consist of a mix of common words that are already in the vocabulary of our pretrained model and new words specific to our corpus (cinematographic terms or actors names, for instance). Our embedding matrix will be built accordingly: for words that are in the vocabulary of our pretrained model, we will take the corresponding row in the embedding matrix of the pretrained model; but for new words we won't have anything, so we will just initialize the corresponding row with a random vector.

我们的词汇表会是普通词的混合的组合，这些普通词已经在我们预训练模型的词汇中，且新词具体对我们的预料库（例如，电影拍摄名词或演员姓名）。我们的嵌入矩阵会相应的创建：对于我们预训练模型的词汇中的词，我们会在预训练模型的嵌入矩阵中包含相应的行，但是对于新词我们不会做任何事情，所以我们只是用随机向量初始化相应的行。

Each of the steps necessary to create a language model has jargon associated with it from the world of natural language processing, and fastai and PyTorch classes available to help. The steps are:

对于自然语言处理世界创建语言模型的每个必须的步骤有与它相关的术语，且fastai和PyTorch类可以提供帮助。这些步骤是：

- Tokenization:: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)
- Numericalization:: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab
- Language model data loader creation:: fastai provides an `LMDataLoader` class which automatically handles creating a dependent variable that is offset from the independent variable by one token. It also handles some important details, such as how to shuffle the training data in such a way that the dependent and independent variables maintain their structure as required
- Language model creation:: We need a special kind of model that does something we haven't seen before: handles input lists which could be arbitrarily big or small. There are a number of ways to do this; in this chapter we will be using a *recurrent neural network* (RNN). We will get to the details of these RNNs in the <chapter_nlp_dive>, but for now, you can think of it as just another deep neural network.
- 标记化：把文本转换为一个词列表（依据你的模型粒度，或字符，或子串）
- 数值化：生成出现（词汇表）的所有唯一词的列表，并转换每个词为一个数值，在词汇表中通过它的索引进行查找
- 语言模型数据加载器创建：fastai提供了一个`LMDataLoader`类，其自动的处理创建一个因变量，这个变量是自变量一个标记的偏置量。这也处理一个重要细节，如以因变量和自变量按需维护它们的结构方式来混洗训练数据
- 语言模型创建：我们需要一个特定的模型，它能够处理我们之前没有见的内容：处理任意大小的输入列表。有很多方法可以做这个操作。在本章节我们会使用*递归神经网络*（RNN）。我们会在<章节：自然语言处理深潜>中接触到这些递归神经网络的细节，但现在，你只需要把它视为另一种深度神经网络。

Let's take a look at how each step works in detail.

现在让我们看一下每个步骤在细节上是怎样处理的。

### Tokenization

### 标记化

When we said "convert the text into a list of words," we left out a lot of details. For instance, what do we do with punctuation? How do we deal with a word like "don't"? Is it one word, or two? What about long medical or chemical words? Should they be split into their separate pieces of meaning? How about hyphenated words? What about languages like German and Polish where we can create really long words from many, many pieces? What about languages like Japanese and Chinese that don't use bases at all, and don't really have a well-defined idea of *word*?

当我们说“转换文本为一个词列表”时，我们忽略了很多细节。例如，我们对标点符号做了什么？我们如何处想“不做”（don't）这样的缩写词？它是一个词，还是两个？那么长的医学和化学词呢？它们应该分割它们分开的含义吗？有连字符的词呢？像德语和波兰语我们能够用很多很多词创建很长的词呢？像日语和中文完全不使用基础语言，实际上没有明确定义的*词义* 呢？

Because there is no one correct answer to these questions, there is no one approach to tokenization. There are three main approaches:

因为对于这些问题没有一个正确的答案，没有一个方法来标记化。这里有三个主要的方法：

- Word-based:: Split a sentence on spaces, as well as applying language-specific rules to try to separate parts of meaning even when there are no spaces (such as turning "don't" into "do n't"). Generally, punctuation marks are also split into separate tokens.
- Subword based:: Split words into smaller parts, based on the most commonly occurring substrings. For instance, "occasion" might be tokenized as "o c ca sion."
- Character-based:: Split a sentence into its individual characters.
- 基于词：通过句子的空格分割，应用特定语言规则，即使没有空格的时候，也尝试分割有意思的部分（例如转换“don't”为“do n't”）。通常，标点符号也分割为单独的标记。
- 基于子词：基于常见的子字符串，把词分割为更小的部分。例如，“occasion”可能被标记化为“o c ca sion。”
- 基于字符：把句子分割为单个字符。

We'll be looking at word and subword tokenization here, and we'll leave character-based tokenization for you to implement in the questionnaire at the end of this chapter.

> jargon: token: One element of a list created by the tokenization process. It could be a word, part of a word (a *subword*), or a single character.

在这里我将会学习词和子词的标记化，在本章节的末尾练习题部分，我们会为你留下基于字符的实现。

> 术语：标记：通过标记化过程创建的一个列表元素。它可以是一个词，一个词的部分（一个子词），或一个单个字符。

### Word Tokenization with fastai

### 用fastai做单词标记化

Rather than providing its own tokenizers, fastai instead provides a consistent interface to a range of tokenizers in external libraries. Tokenization is an active field of research, and new and improved tokenizers are coming out all the time, so the defaults that fastai uses change too. However, the API and options shouldn't change too much, since fastai tries to maintain a consistent API even as the underlying technology changes.

fastai不是提供它自己的标记器，相反在外部库它提供了针对一系列标记器的一致的接口。标记化是一个很活跃的研究领域，新的和改善的标记器总是在出现，所以fastai默认下也要有使用变化。然而，API和操作不应该变化太多，因此即使底层技术变化，fastai也尝试维护一个一致的API。

Let's try it out with the IMDb dataset that we used in <chapter_intro>:

让我们用在<章节：概述>中我们所使用的IMDb数据做实验：

```
from fastai.text.all import *
path = untar_data(URLs.IMDB)
```

We'll need to grab the text files in order to try out a tokenizer. Just like `get_image_files`, which we've used many times already, gets all the image files in a path, `get_text_files` gets all the text files in a path. We can also optionally pass `folders` to restrict the search to a particular list of subfolders:

我们需要抓取文本文件，以便实验一个标记器。就像我们已经用了它很多次的`get_image_files`一样，它能够获取路径下的所有图像文件，`get_text_files`能够获取路径下的所有文本。我们也能够操作传递`文件夹`来限制搜索一个特定的子文件夹列表：

```
files = get_text_files(path, folders = ['train', 'test', 'unsup'])
```

Here's a review that we'll tokenize (we'll just print the start of it here to save space):

这是一个评论，我们会做标记化（为了节省空间我们只会输出开头部分）：

```
txt = files[0].open().read(); txt[:75]
```

Out: 'This movie, which I just discovered at the video store, has apparently sit '

As we write this book, the default English word tokenizer for fastai uses a library called *spaCy*. It has a sophisticated rules engine with special rules for URLs, individual special English words, and much more. Rather than directly using `SpacyTokenizer`, however, we'll use `WordTokenizer`, since that will always point to fastai's current default word tokenizer (which may not necessarily be spaCy, depending when you're reading this).

在我们写这本书的时候，fastai默认英文标记器使用的是称为*spaCy*库。它有一个对URL、单个特定英文单词及更多方面的特定规则的成熟规则引擎。然而，我们会使用`wordTokenizer`而不是`SpacyTokenizer`，由于它会一直指向fastai当前默认单词标记器（它可能不一定是spaCy，这依赖你正在阅读这本书的时间）。

Let's try it out. We'll use fastai's `coll_repr(collection, n)` function to display the results. This displays the first *`n`* items of *`collection`*, along with the full size—it's what `L` uses by default. Note that fastai's tokenizers take a collection of documents to tokenize, so we have to wrap `txt` in a list:

让我们实验一下。我们会使用fastai的`coll_repr(collection, n)`函数来显示结果。这个显示了全尺寸（这是默认使用的`L`）的*`collection`* 头 *`n`* 个数据项。注意，fastai的标记器收集文档来做标记化，所以我们必须在列表中打包`txt`：

```
spacy = WordTokenizer()
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
```

Out: (#201) ['This' , 'movie' , ' , ' , 'which' , 'I' , 'just' , 'discovered' , 'at' , 'the' , 'video' , 'store' , ' , ' , 'has' , 'apparently' , 'sit' , 'around' , 'for' , 'a' , 'couple' , 'of' , 'years' , 'without' , 'a' , 'distributor' , ' . ' , 'It' , "'s" , 'easy' , 'to' , 'see' ... ]

As you see, spaCy has mainly just separated out the words and punctuation. But it does something else here too: it has split "it's" into "it" and "'s". That makes intuitive sense; these are separate words, really. Tokenization is a surprisingly subtle task, when you think about all the little details that have to be handled. Fortunately, spaCy handles these pretty well for us—for instance, here we see that "." is separated when it terminates a sentence, but not in an acronym or number:

 正如你看到的，spaCy主要分割了单词和标点符号。但它也做了一些其它事情：它把“it's”分割为“it”和“'s”。这很直观，有不同的词，真的是这样。当你思考所有必须处理的细小任务时，标记化是惊人的精妙任务。幸运的是，spaCy为我们处理这些内容处理的相当的好。例如，上面我们看到了当“.” 结束一个句子时，它被分割了，但是在首字母缩略词或数字中，就不会被分割：

```
first(spacy(['The U.S. dollar $1 is $1.00.']))
```

Out: (#9) ['The' , 'U.S.' , 'dollar' , ' $ ' , ' 1 ' , 'is' , '  \$ ' , '1.00' , ' . ']

fastai then adds some additional functionality to the tokenization process with the `Tokenizer` class:

fastai然后增加了一些附属功能，用`Tokenizer`类来做标记化处理：

```
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))
```

Out: (#228) ['xxbos' , 'xxmaj' , 'this' , 'movie' , ',' , 'which' , 'i' , 'just' , 'discovered' , 'at' , 'the' , 'video' , 'store' , ',' , 'has' , 'apparently' , 'sit' , 'around' , 'for' , 'a' , 'couple' , 'of' , 'years' , 'without' , 'a' , 'distributor' , '.' , 'xxmaj' , 'it',"'s",'easy' ...]

Notice that there are now some tokens that start with the characters "xx", which is not a common word prefix in English. These are *special tokens*.

注意有一些以“xx”为开始的标记，在英文中这不是常见的词前缀。有一些*专用标记*。

For example, the first item in the list, `xxbos`, is a special token that indicates the start of a new text ("BOS" is a standard NLP acronym that means "beginning of stream"). By recognizing this start token, the model will be able to learn it needs to "forget" what was said previously and focus on upcoming words.

例如，在列表中的第一个数据项，是一个专用标记，表明一个新文本的开始（“BOS”是一个标准的自然语言首字母缩略词，意思是“beginning of stream”）。通过识别这个开始的标记，模型将能够理解“忘记”之前说的内容并关注将要到来的词。

These special tokens don't come from spaCy directly. They are there because fastai adds them by default, by applying a number of rules when processing text. These rules are designed to make it easier for a model to recognize the important parts of a sentence. In a sense, we are translating the original English language sequence into a simplified tokenized language—a language that is designed to be easy for a model to learn.

这些专用标记不是直接来自spaCy。它们的存在是因为fastai在处理文本时通过应用一些规则默认添加了它们。这些规则设计使得模型更容易识别句子的重要部分。某种意义上，我们翻译原始英文序列进入一个简单的标记语言（设计了一个模型容易学习的语言）。

For instance, the rules will replace a sequence of four exclamation points with a special *repeated character* token, followed by the number four, and then a single exclamation point. In this way, the model's embedding matrix can encode information about general concepts such as repeated punctuation rather than requiring a separate token for every number of repetitions of every punctuation mark. Similarly, a capitalized word will be replaced with a special capitalization token, followed by the lowercase version of the word. This way, the embedding matrix only needs the lowercase versions of the words, saving compute and memory resources, but can still learn the concept of capitalization.

例如：这个规则会有专有*重复字符* 标记替换四个感叹号序列，然后是数据四，然后一个感叹号。用这个方法模型的嵌入矩阵能够编码常用概念信息（如重复的标点符号），而不需要对每个标点标记的重复做单独的标记。同样的，一个大写的单词会用一个专用大写标记来替换，然后是该单词的小写。这样嵌入矩阵只需要该单词的小写版，节省的计算和内存资源，但还能一直学习大写的概念。

Here are some of the main special tokens you'll see:

- `xxbos`:: Indicates the beginning of a text (here, a review)
- `xxmaj`:: Indicates the next word begins with a capital (since we lowercased everything)
- `xxunk`:: Indicates the next word is unknown

下面你会看到有一些主要的专用标记：

- `xxbos`：表明一个文本的开始（这里是一条评论）
- `xxmaj`：表明下个单词是大写字母开始的（因为我小写了所有单词）
- `xxunk`：表明下个单词是未知的

To see the rules that were used, you can check the default rules:

来查看一下使用的规则，你能够查看默认规则：

```
defaults.text_proc_rules
```

Out: $\begin{array}{ll} 
[&<function fastai.text.core.fix\_html(x)>,\\
 &<function fastai.text.core.replace\_rep(t)>,\\
 &<function fastai.text.core.replace\_wrep(t)>,\\
 &<function fastai.text.core.spec\_add\_spaces(t)>,\\
 &<function fastai.text.core.rm\_useless\_spaces(t)>,\\
 &<function fastai.text.core.replace\_all\_caps(t)>,\\
 &<function fastai.text.core.replace\_maj(t)>,\\
 &<function fastai.text.core.lowercase(t, add\_bos=True, add\_eos=False)>&]
\end{array}$

As always, you can look at the source code of each of them in a notebook by typing:

像之前讲的，在notebook中通过输入下述命令，我们能够查看它们每个规则的源代码：

```
??replace_rep
```

Here is a brief summary of what each does:

- `fix_html`:: Replaces special HTML characters with a readable version (IMDb reviews have quite a few of these)
- `replace_rep`:: Replaces any character repeated three times or more with a special token for repetition (`xxrep`), the number of times it's repeated, then the character
- `replace_wrep`:: Replaces any word repeated three times or more with a special token for word repetition (`xxwrep`), the number of times it's repeated, then the word
- `spec_add_spaces`:: Adds spaces around / and #
- `rm_useless_spaces`:: Removes all repetitions of the space character
- `replace_all_caps`:: Lowercases a word written in all caps and adds a special token for all caps (`xxup`) in front of it
- `replace_maj`:: Lowercases a capitalized word and adds a special token for capitalized (`xxmaj`) in front of it
- `lowercase`:: Lowercases all text and adds a special token at the beginning (`xxbos`) and/or the end (`xxeos`)

下面是每个规则所做事情的简短总结：

- `fix_html`：用一个可阅读的版本来替换专用HTML字符（IMDb评论有相当多这样的内容）
- `replace_rep`：用一个专用重复标记（`xxrep`）替换任意重复三次或以上的字符，然后重复的次数，然后是字符
- `replace_wrep`：用一个专用单词重复标记（`xxwrep`）替换任意重复三次或以上的单词，然后重复的次数，然后是单词
- `spec_add_spaces`： / 和 # 周围增加空格
- `rm_useless_spaces`：移除所有重复的空格字符
- `replace_all_caps`：对于所有大写字符都小写化单词，然后在单词前面添加一个所有大写化专用（`xxup`）标记
- `replace_maj`：小写一个大写单词，并在单词前面添加一个大写化专用标记（`xxmaj`）
- `lowercase`：小写所有文本，并在开始添加专用标记（`xxbos`）和（/或）结尾添加专用标记（`xxeos`）

Let's take a look at a few of them in action:

让我们看一下他们中一些规则的操作：

```
coll_repr(tkn('&copy;   Fast.ai www.fast.ai/INDEX'), 31)
```

Out: "(#11) ['xxbos' , '©' , 'xxmaj' , 'fast.ai' , 'xxrep' , '3' , 'w' , '.fast.ai' , ' / ' , 'xxup' , 'index'...]"

Now let's take a look at how subword tokenization would work.

现在让我们看一下子词标记化如何处理。

### Subword Tokenization

### 子词标记化

In addition to the *word tokenization* approach seen in the last section, another popular tokenization method is *subword tokenization*. Word tokenization relies on an assumption that spaces provide a useful separation of components of meaning in a sentence. However, this assumption is not always appropriate. For instance, consider this sentence: 我的名字是郝杰瑞 ("My name is Jeremy Howard" in Chinese). That's not going to work very well with a word tokenizer, because there are no spaces in it! Languages like Chinese and Japanese don't use spaces, and in fact they don't even have a well-defined concept of a "word." There are also languages, like Turkish and Hungarian, that can add many subwords together without spaces, creating very long words that include a lot of separate pieces of information.

除了在上节中看到的*词标记化*方法，另一个流行的标记化方法是*子词标记化*。词标记化依赖一个假设，在句子中空格提供了含义要素的一个有效分割。然而，这个假设不总是适合的。例如，思考一下这个句子：我的名字是郝杰瑞（“My name is Jeremy Howard” in English）。用词标记器将不会处理的很好，因为在句子中没有空格！像中文和日文不使用空格，事实上他们甚至没有很好的定义一个“词”的概念。也有很多语言，像土耳其语和匈牙利语，能够添加很好子词在一起而没有空格，创建了包含很多独立信息的很长的词。

To handle these cases, it's generally best to use subword tokenization. This proceeds in two steps:

1. Analyze a corpus of documents to find the most commonly occurring groups of letters. These become the vocab.
2. Tokenize the corpus using this vocab of *subword units*.

处理这些案例，通常最好使用子词标记化。这个过程有两步：

1. 分析文档的语料库找出最发生的字母组。这会变为 词汇表。
2. 使用这个*子词单元* 的词汇表来标记化语料库。

Let's look at an example. For our corpus, we'll use the first 2,000 movie reviews:

我们看一个例子。我们会使用头 2,000 条电影评论为我们的语料库：

```
txts = L(o.open().read() for o in files[:2000])
```

We instantiate our tokenizer, passing in the size of the vocab we want to create, and then we need to "train" it. That is, we need to have it read our documents and find the common sequences of characters to create the vocab. This is done with `setup`. As we'll see shortly, `setup` is a special fastai method that is called automatically in our usual data processing pipelines. Since we're doing everything manually at the moment, however, we have to call it ourselves. Here's a function that does these steps for a given vocab size, and shows an example output:

我们实例化我们的标记器，传递我们希望创建的词汇表尺寸，然后我们需要“训练”它。即，我们需要让它阅读我们的文档和查找常用字符序列来创建词汇表。这是用`setup`来完成。我们很快就会看到，`setup`是一个特定fastai方法，在我们通用数据处理管道中它会自动调用。然而，由于我们在手动做任何事情的时候，我们必须自己来调用它。下面是对于给定的词汇表尺寸执行这些步骤的函数，且展示了一个输出事例：

```
def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])
```

Let's try it out:

让我们做一下实验：

```
subword(1000)
```

Out: '▁This ▁movie , ▁which ▁I ▁just ▁dis c over ed ▁at ▁the ▁video ▁st or e , ▁has ▁a p par ent ly ▁s it ▁around ▁for ▁a ▁couple ▁of ▁years ▁without ▁a ▁dis t ri but or . ▁It'

When using fastai's subword tokenizer, the special character `▁` represents a space character in the original text.

If we use a smaller vocab, then each token will represent fewer characters, and it will take more tokens to represent a sentence:

当使用fastai的子词标记器时，特殊字符`▁`代表了原始文本中的一个空格字符。

如果我们使用了一个更小尺寸的词汇表，其后每个标记交付代表更少的字符，且它将要花费更多的标记来代表一个句子：

```
subword(200)
```

Out: '▁ T h i s ▁movie , ▁w h i ch ▁I ▁ j us t ▁ d i s c o ver ed ▁a t ▁the ▁ v id e o ▁ st or e , ▁h a s'

On the other hand, if we use a larger vocab, then most common English words will end up in the vocab themselves, and we will not need as many to represent a sentence:

换句话说，如果我们使用了一个更大尺寸的词汇表，绝大数常用英文单词将最终在它自己的词汇表中，我们将不需要太多的标记来代表一个句子：

```
subword(10000)
```

Out: "▁This ▁movie , ▁which ▁I ▁just ▁discover ed ▁at ▁the ▁video ▁store , ▁has ▁apparently ▁sit ▁around ▁for ▁a ▁couple ▁of ▁years ▁without ▁a ▁distributor . ▁It ' s ▁easy ▁to ▁see ▁why . ▁The ▁story ▁of ▁two ▁friends ▁living"

Picking a subword vocab size represents a compromise: a larger vocab means fewer tokens per sentence, which means faster training, less memory, and less state for the model to remember; but on the downside, it means larger embedding matrices, which require more data to learn.

选择一个子词词汇表的尺寸代表着一个妥协：一个大尺寸词汇表意味着每个句子更少的标记，这表示更快的训练、更少的存在使用和对于模型更少的状态来记忆。但不利的一面是，这意味着更大的嵌入矩阵，需要更多的数据来训练。

Overall, subword tokenization provides a way to easily scale between character tokenization (i.e., using a small subword vocab) and word tokenization (i.e., using a large subword vocab), and handles every human language without needing language-specific algorithms to be developed. It can even handle other "languages" such as genomic sequences or MIDI music notation! For this reason, in the last year its popularity has soared, and it seems likely to become the most common tokenization approach (it may well already be, by the time you read this!).

总体来说，子词标记化提供了一个方法，很容易在字符标记化（即，使用一个小的子词词汇表）和单词标记化（即，使用一个大的子词词汇表）之间轻松的缩放，不需要开发特定语言算法来处理每一个人类语言。它甚至能够处理其它“语言”，例如基因序列或MIDI音乐符号！基于这个原因，在过去的一年里它的受欢迎程度猛增，且它好像可能成为最常用的标记化方法（在你阅读这部分内容时，它很可能已经是这样了！）。

Once our texts have been split into tokens, we need to convert them to numbers. We'll look at that next.

一旦我们的文本已经被分割为标记，我们需要把它们转化为数值。接下来我们看一下这方面的内容。

### Numericalization with fastai

### 用fastai数值化

*Numericalization* is the process of mapping tokens to integers. The steps are basically identical to those necessary to create a `Category` variable, such as the dependent variable of digits in MNIST:

1. Make a list of all possible levels of that categorical variable (the vocab).
2. Replace each level with its index in the vocab.

Let's take a look at this in action on the word-tokenized text we saw earlier:

*数值化* 是映射标记到整型数值的过程。这些步骤基本上与那些必须创建一个`分类`变量是相同的，例如MINIST中的数字因变量：

1. 生成一个所有可能级别的分类列表（词汇表）。
2. 用词汇表中的索引替换每个级别。

让我们在之前看到的词标记文本上做一下操作来看一下：

```
toks = tkn(txt)
print(coll_repr(tkn(txt), 31))
```

Out: (#228) ['xxbos' , 'xxmaj' , 'this' , 'movie' , ',' , 'which' , 'i' , 'just' , 'discovered' , 'at' , 'the' , 'video' , 'store' , ',' , 'has' , 'apparently' , 'sit' , 'around' , 'for' , 'a' , 'couple' , 'of' , 'years' , 'without' , 'a' , 'distributor' , '.' , 'xxmaj' , 'it',"'s",'easy'...]

Just like with `SubwordTokenizer`, we need to call `setup` on `Numericalize`; this is how we create the vocab. That means we'll need our tokenized corpus first. Since tokenization takes a while, it's done in parallel by fastai; but for this manual walkthrough, we'll use a small subset:

就像`SubwordTokenizer`一样，我们需要在`Numericalize`上调用`setup`，这是我们如何创建词汇表。这表示我们首先需要标记语料库。因为标记化需要一段时间 ，它通过fastai平行完成。但对于这个手工演练，我们将使用一个小子集：

```
toks200 = txts[:200].map(tkn)
toks200[0]
```

Out[ ]: (#228) ['xxbos' , 'xxmaj' , 'this' , 'movie' , ',' , 'which' , 'i' , 'just' , 'discovered' , 'at' ...]

We can pass this to `setup` to create our vocab:

我们能够传递这个信息给`setup`来创建我们词汇表：

```
num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20)
```

Out: "(#2000) ['xxunk' , 'xxpad' , 'xxbos' , 'xxeos' , 'xxfld' , 'xxrep' , 'xxwrep' , 'xxup' , 'xxmaj' , 'the' , '.' , ',' , 'a' , 'and' , 'of' , 'to' , 'is' , 'in' , 'i' , 'it' ...]"

Our special rules tokens appear first, and then every word appears once, in frequency order. The defaults to `Numericalize` are `min_freq=3,max_vocab=60000`. `max_vocab=60000` results in fastai replacing all words other than the most common 60,000 with a special *unknown word* token, `xxunk`. This is useful to avoid having an overly large embedding matrix, since that can slow down training and use up too much memory, and can also mean that there isn't enough data to train useful representations for rare words. However, this last issue is better handled by setting `min_freq`; the default `min_freq=3` means that any word appearing less than three times is replaced with `xxunk`.

首先显示的是我们特定的标记规则，随后按照频率顺序显示的每个词。对于`Numericalize`的默认设置是`min_freq=3,max_vocab=60000`。在fastai中`max_vocab=60000`的结果是替换所有除了常用的60,000个词外其它词为*未知词标记*（`xxunk`）。这是用来避免有一个过大的嵌入矩阵，因为这能够减慢训练和使用太多的内存，且也表示不会有足够的数据来用于表示罕见词的训练。因此，这是最后的问题，通过设置`min_freq`来更好的处理。`min_freq=3`是默认的，表示任何词出现小于三次会被`xxunk`所替换。

fastai can also numericalize your dataset using a vocab that you provide, by passing a list of words as the `vocab` parameter.

通过传递单词列表作为`词汇表`的参数，fastai也能够使用你提供的词汇表来数值化你的数据集。

Once we've created our `Numericalize` object, we can use it as if it were a function:

一旦我们创建`Numericalize`对象，我们就能够像是一个函数那样使用它：

```
nums = num(toks)[:20]; nums
```

Out: tensor([  2,   8,  21,  28,  11,  90,  18,  59,   0,  45,   9, 351, 499,  11,  72, 533, 584, 146,  29,  12  ])

This time, our tokens have been converted to a tensor of integers that our model can receive. We can check that they map back to the original text:

这次，我们的标记已经转换为一个我们醋能够接收的整形张量。我们能够通过它们映射回原始文本来检查这个张量：

```
' '.join(num.vocab[o] for o in nums)
```

Out: 'xxbos xxmaj this movie , which i just xxunk at the video store , has apparently sit around for a'

Now that we have numbers, we need to put them in batches for our model.

现在我们已经有了数值，我们需要分批把他们放入模型。

### Putting Our Texts into Batches for a Language Model

### 我们的文本分批给语言模型

When dealing with images, we needed to resize them all to the same height and width before grouping them together in a mini-batch so they could stack together efficiently in a single tensor. Here it's going to be a little different, because one cannot simply resize text to a desired length. Also, we want our language model to read text in order, so that it can efficiently predict what the next word is. This means that each new batch should begin precisely where the previous one left off.

当处理图片时，在一个最小批次中组合所有图片前，我们需要重新调整所有图片到相同的高度和宽度，这样他们能够在单长量中有效率的堆在一起。现在对于文本会有一点不同，因为它不能简单的调整文本大小到一个期望的长度。同样，我们也希望我们的语言模型井然有序的阅读文本，那么它就能够有效的预测下一个词是什么了。这表示每一新的批次都应该准确的从上一个批次恰好停止的地方开始。

Suppose we have the following text:

假设我们有如下文本：

> : In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while.

The tokenization process will add special tokens and deal with punctuation to return this text:

标记化过程交付添加专用标记和处理标点符号，并返回这个文本：

> : xxbos xxmaj in this chapter , we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface . xxmaj first we will look at the processing steps necessary to convert text into numbers and how to customize it . xxmaj by doing this , we 'll have another example of the preprocessor used in the data block xxup api . \n xxmaj then we will study how we build a language model and train it for a while .

We now have 90 tokens, separated by spaces. Let's say we want a batch size of 6. We need to break this text into 6 contiguous parts of length 15:

现在我们有了90个标记，通过空格分割。比方说我们希望一个批次尺寸为 6 。我需要把这个文本分为 6 个长度 15 的连续部分：

```
#hide_input
stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."
tokens = tkn(stream)
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| xxbos | xxmaj   | in           | this    | chapter | ,       | we         | will  | go        | back   | over    | the  | example | of      | classifying |
| movie | reviews | we           | studied | in      | chapter | 1          | and   | dig       | deeper | under   | the  | surface | .       | xxmaj       |
| first | we      | will         | look    | at      | the     | processing | steps | necessary | to     | convert | text | into    | numbers | and         |
| how   | to      | customize    | it      | .       | xxmaj   | by         | doing | this      | ,      | we      | 'll  | have    | another | example     |
| of    | the     | preprocessor | used    | in      | the     | data       | block | xxup      | api    | .       | \n   | xxmaj   | then    | we          |
| will  | study   | how          | we      | build   | a       | language   | model | and       | train  | it      | for  | a       | while   | .           |

In a perfect world, we could then give this one batch to our model. But that approach doesn't scale, because outside of this toy example it's unlikely that a single batch containing all the texts would fit in our GPU memory (here we have 90 tokens, but all the IMDb reviews together give several million).

在理想世界，那么我们能够我们模型一个批次。但是这个方法不能收放，因为在这个小例子之外，不可能单个批次所包含的文本能够正好适合我们的GPU内存（现在我们有90个标记，但是所有的IMDb评论合起来有好几百万）。

So, we need to divide this array more finely into subarrays of a fixed sequence length. It is important to maintain order within and across these subarrays, because we will use a model that maintains a state so that it remembers what it read previously when predicting what comes next.

所以我们需要更加精细的把这个数组分为固定序列长度的子数组。这对于维护子数组内部和数组之间的顺序是很重要的，因为我们将需要使用一个维护状态的模型，以便预测下个词的时候，它记住了之前读到的内容。

Going back to our previous example with 6 batches of length 15, if we chose a sequence length of 5, that would mean we first feed the following array:

返回先前长度为15的6个批次例子中，如果我们了长度为5的序列，表示我们首先需要喂给下述数组：

```
#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```

| 1 | 2 | 3 | 4 | 5 |
| ----- | ------- | ------------ | ------- | ------- |
| xxbos | xxmaj   | in           | this    | chapter |
| movie | reviews | we           | studied | in      |
| first | we      | will         | look    | at      |
| how   | to      | customize    | it      | .       |
| of    | the     | preprocessor | used    | in      |
| will  | study   | how          | we      | build   |

Then this one:

然后是这个：

```
#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```

| 1 | 2 | 3 | 4 | 5 |
| ------- | ---------- | ----- | --------- | ------ |
| ,       | we         | will  | go        | back   |
| chapter | 1          | and   | dig       | deeper |
| the     | processing | steps | necessary | to     |
| xxmaj   | by         | doing | this      | ,      |
| the     | data       | block | xxup      | api    |
| a       | language   | model | and       | train  |

And finally:

这是最后的部分：

```
#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```

| 1 | 2 | 3 | 4 | 5 |
| ------- | ---- | ------- | ------- | ----------- |
| over    | the  | example | of      | classifying |
| under   | the  | surface | .       | xxmaj       |
| convert | text | into    | numbers | and         |
| we      | 'll  | have    | another | example     |
| .       | \n   | xxmaj   | then    | we          |
| it      | for  | a       | while   | .           |

Going back to our movie reviews dataset, the first step is to transform the individual texts into a stream by concatenating them together. As with images, it's best to randomize the order of the inputs, so at the beginning of each epoch we will shuffle the entries to make a new stream (we shuffle the order of the documents, not the order of the words inside them, or the texts would not make sense anymore!).

返回到我们的电影评论数据集，第一步是通过把联系在一起的孤立的文本换为流。像图片一样，它最好是随机输入顺序，所以在每个周期的开始，我们会重新整理记录来生成一个新的流（我们整理文档的顺序，不是它们内部单词的顺序，或使得的文本不再有任何意义！）。

We then cut this stream into a certain number of batches (which is our *batch size*). For instance, if the stream has 50,000 tokens and we set a batch size of 10, this will give us 10 mini-streams of 5,000 tokens. What is important is that we preserve the order of the tokens (so from 1 to 5,000 for the first mini-stream, then from 5,001 to 10,000...), because we want the model to read continuous rows of text (as in the preceding example). An `xxbos` token is added at the start of each during preprocessing, so that the model knows when it reads the stream when a new entry is beginning.

然后我们剪切这个流为一个确定数目的批次（它是我们的*批次尺寸*）。例如，如果流有 50,000 个标记，且我们设置批次尺寸为 10 ，这会给我们提供包含 5,000个标记的 10 个最小流。重要的是我们保护了标记顺序（因此第一个最小流从 1 到 5,000 ，然后从 5,001 到 10,000...），因为我们想让模型连续的读取文本行（正如先前的例子）。一个`xxbos`标记在每次预处理期间的一开始添加的，所以当是一个新的条目一开始的时候，模型知道读取知道何时读取数据流。

So to recap, at every epoch we shuffle our collection of documents and concatenate them into a stream of tokens. We then cut that stream into a batch of fixed-size consecutive mini-streams. Our model will then read the mini-streams in order, and thanks to an inner state, it will produce the same activation whatever sequence length we picked.

所以总的来说，在每个周期上我们整理了我们收集的文档，并串联它们为一个标记流。然后我们把流剪切为固定尺寸的连续最小流批次。因此我们的模型会按照顺序读取最小批次，这要归功于一个内部状态，无论我们选取什么序列长度，它都会产生相同的激活。

This is all done behind the scenes by the fastai library when we create an `LMDataLoader`. We do this by first applying our `Numericalize` object to the tokenized texts:

当我们通过fastai库创建一个`LMDataLoader`时，这就是所有的幕后的操作。我们首先通过应用我们的`Numericalize`对象做这个操作来标记文本：

```
nums200 = toks200.map(num)
```

and then passing that to `LMDataLoader`:

然后传递这个对象给`LMDataLoader`：

```
dl = LMDataLoader(nums200)
```

Let's confirm that this gives the expected results, by grabbing the first batch:

通过抓取第一个批次，让我们确认一下是否它提供了所期望的结果：

```
x,y = first(dl)
x.shape,y.shape
```

Out: (torch.Size([64, 72]), torch.Size([64, 72]))

and then looking at the first row of the independent variable, which should be the start of the first text:

然后查看自变量的第一行，它应该是第一个文本的开始：

```
' '.join(num.vocab[o] for o in x[0][:20])
```

Out: 'xxbos xxmaj this movie , which i just xxunk at the video store , has apparently sit around for a'

The dependent variable is the same thing offset by one token:

因变量是通过一个标记偏移的相同的内容：

```
' '.join(num.vocab[o] for o in y[0][:20])
```

Out: 'xxmaj this movie , which i just xxunk at the video store , has apparently sit around for a couple'

This concludes all the preprocessing steps we need to apply to our data. We are now ready to train our text classifier.

这完成了我们需要应用到我们数据上的所有预操作步骤。现在我们准备训练我们的文本分类器了。

## Training a Text Classifier

## 训练一个文本分类器

As we saw at the beginning of this chapter, there are two steps to training a state-of-the-art text classifier using transfer learning: first we need to fine-tune our language model pretrained on Wikipedia to the corpus of IMDb reviews, and then we can use that model to train a classifier.

正如我们在本章一开始，有两步使用迁移学习来训练一个先进的文本分类器：首先，我们需要我们微调在Wikipedia上预训练的语言模型为IMDb评论语料库，其后我们能够用这个模型训练一个分类器。

As usual, let's start with assembling our data.

照例，让我们从集成我们的数据开始。

### Language Model Using DataBlock

### 使用DataBlock语言模型

fastai handles tokenization and numericalization automatically when `TextBlock` is passed to `DataBlock`. All of the arguments that can be passed to `Tokenize` and `Numericalize` can also be passed to `TextBlock`. In the next chapter we'll discuss the easiest ways to run each of these steps separately, to ease debugging—but you can always just debug by running them manually on a subset of your data as shown in the previous sections. And don't forget about `DataBlock`'s handy `summary` method, which is very useful for debugging data issues.

当`TextBlock`被传递给`DataBlock`时，fastai自动的处理标记化和数值化。所有参数能够被传递给`Tokenize`和`Numericalize`也能够被传递给`TextBlock`。在下一章我们会讨论更容易的方法来分别运行这些步骤的每一个，与简单的调试。但是你能够像上一部分所说明的那样，在你数据的子集上通过手动的运行它们来进行调试。且不要忘记`DataBlock`的处理`总结`方法，它对于调试数据问题很有用处。

Here's how we use `TextBlock` to create a language model, using fastai's defaults:

下面是我们使用fastai的默认设置，使用`TextBlock`来创建一个语言模型：

```
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)
```

One thing that's different to previous types we've used in `DataBlock` is that we're not just using the class directly (i.e., `TextBlock(...)`, but instead are calling a *class method*. A class method is a Python method that, as the name suggests, belongs to a *class* rather than an *object*. (Be sure to search online for more information about class methods if you're not familiar with them, since they're commonly used in many Python libraries and applications; we've used them a few times previously in the book, but haven't called attention to them.) The reason that `TextBlock` is special is that setting up the numericalizer's vocab can take a long time (we have to read and tokenize every document to get the vocab). To be as efficient as possible it performs a few optimizations:

有一个事情与我们之前在`DataBlock`中使用的类型不同，那是我们不只是直接使用类（即，`TextBlock（...）`），相反调用了一个*类方法*。一个类方法是一个Python方法，作为命名建议其术语一个*类*而不是*对象*。（如果你不熟悉这些内容，一定在线搜索一些关于类方法的信息，因为它们在很多Python库和应用中是常用的。在本书前面的部分我们已经使用了很多次了，但对他们没有引起主意。）`TextBlock`是特殊的原因是建立数值化器的词汇表需要花费很长时间（我们不得不读取和标记每个文档来获取词汇表）。为尽可能的高效它做一些优化：

- It saves the tokenized documents in a temporary folder, so it doesn't have to tokenize them more than once
- It runs multiple tokenization processes in parallel, to take advantage of your computer's CPUs
- 它在临时文件夹保存标记文档，因此它就不需要标记它们多次
- 并行运行多个标记化过程，来利用你计算机的多颗CPU

We need to tell `TextBlock` how to access the texts, so that it can do this initial preprocessing—that's what `from_folder` does.

我们需要告知`Textblock`如何获取文本，因此它能够做这个初始预处理，即`from_folder`做的内容。

`show_batch` then works in the usual way:

然后`show_batch`以通常的方法运行：

```
dls_lm.show_batch(max_n=2)
```

|      |                                                         text |                                                        text_ |
| ---: | -----------------------------------------------------------: | -----------------------------------------------------------: |
|    0 | xxbos xxmaj it 's awesome ! xxmaj in xxmaj story xxmaj mode , your going from punk to pro . xxmaj you have to complete goals that involve skating , driving , and walking . xxmaj you create your own skater and give it a name , and you can make it look stupid or realistic . xxmaj you are with your friend xxmaj eric throughout the game until he betrays you and gets you kicked off of the skateboard | xxmaj it 's awesome ! xxmaj in xxmaj story xxmaj mode , your going from punk to pro . xxmaj you have to complete goals that involve skating , driving , and walking . xxmaj you create your own skater and give it a name , and you can make it look stupid or realistic . xxmaj you are with your friend xxmaj eric throughout the game until he betrays you and gets you kicked off of the skateboard xxunk |
|    1 | what xxmaj i 've read , xxmaj death xxmaj bed is based on an actual dream , xxmaj george xxmaj barry , the director , successfully transferred dream to film , only a genius could accomplish such a task . \n\n xxmaj old mansions make for good quality horror , as do portraits , not sure what to make of the killer bed with its killer yellow liquid , quite a bizarre dream , indeed . xxmaj also , this | xxmaj i 've read , xxmaj death xxmaj bed is based on an actual dream , xxmaj george xxmaj barry , the director , successfully transferred dream to film , only a genius could accomplish such a task . \n\n xxmaj old mansions make for good quality horror , as do portraits , not sure what to make of the killer bed with its killer yellow liquid , quite a bizarre dream , indeed . xxmaj also , this is |

Now that our data is ready, we can fine-tune the pretrained language model.

现在我们的数据准备好了，我们能够微调预训练语言模型了。

### Fine-Tuning the Language Model

### 微调语言模型

To convert the integer word indices into activations that we can use for our neural network, we will use embeddings, just like we did for collaborative filtering and tabular modeling. Then we'll feed those embeddings into a *recurrent neural network* (RNN), using an architecture called *AWD-LSTM* (we will show you how to write such a model from scratch in <chapter_nlp_dive>). As we discussed earlier, the embeddings in the pretrained model are merged with random embeddings added for words that weren't in the pretraining vocabulary. This is handled automatically inside `language_model_learner`:

转换整形词索引为激活，我们就能够用于我们的神经网络，我们会使用嵌入，就像我们在协同过滤和表格建模上做的那样。然后我们会使用一个称为*AWD-LSTM* 的架构（我们会在<章节：自然语言深潜>中说明从零开始编写这样一个模型）把那些嵌入喂给*卷积神经网络*（RNN）。正如我们早先讨论的，在预训练模型中的嵌入是与不是预训练词汇表中的那些词添加的随机嵌入是合并的。这个是在`language_model_learner`内部自动处理的：

```
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
```

The loss function used by default is cross-entropy loss, since we essentially have a classification problem (the different categories being the words in our vocab). The *perplexity* metric used here is often used in NLP for language models: it is the exponential of the loss (i.e., `torch.exp(cross_entropy)`). We also include the accuracy metric, to see how many times our model is right when trying to predict the next word, since cross-entropy (as we've seen) is both hard to interpret, and tells us more about the model's confidence than its accuracy.

默认使用的损失函数是交叉熵损失，因为本质上来说我们有一个分类问题（在我们词汇表中不同的分类是词）。这里使用的*困惑*度通常被用于自然语言处理的语言模型：它是损失的指数（即`torch.exp(cross_entropy)`）。我们也包含精度招标，来查看尝试预测一个词的时候我们模型的正确次数，因为交差熵（我们已经学过了）有两个特点：难以解释和告诉我们更多关于模型置信度而不是精度。

Let's go back to the process diagram from the beginning of this chapter. The first arrow has been completed for us and made available as a pretrained model in fastai, and we've just built the `DataLoaders` and `Learner` for the second stage. Now we're ready to fine-tune our language model!

让我们返回到本章节一开始的流程图。我们已经完成了第一个箭头部分且在fastai中作为一个可用的预训练模型，第二阶段我们只是创建了`DataLoaders`和`Learner`。现在我们准备来微调我们的语言模型了！

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00027.png" alt="Diagram of the ULMFiT process" width="700" caption="The ULMFiT process" id="ulmfit_process" />
  </p>
  <p align="center">图：通用语言模型微调过程</p>
</div>

It takes quite a while to train each epoch, so we'll be saving the intermediate model results during the training process. Since `fine_tune` doesn't do that for us, we'll use `fit_one_cycle`. Just like `cnn_learner`, `language_model_learner` automatically calls `freeze` when using a pretrained model (which is the default), so this will only train the embeddings (the only part of the model that contains randomly initialized weights—i.e., embeddings for words that are in our IMDb vocab, but aren't in the pretrained model vocab):

它需要花费一些时间来训练每个周期，所以我们会保存训练过程期间的中间模型结果。因为`fine_tune`不能为我们做这个事情，所以我们会用`fit_one_cycle`。就像`cnn_learner`一样，当使用一个预训练模型时`language_model_learner`自动调用`freeze`（它是默认的），所以它只会训练嵌入层（模型唯一包含随机初始权重的部分。即，是我们IMDb词汇表中的词嵌入）：

```
learn.fit_one_cycle(1, 2e-2)
```

| epoch | train_loss | valid_loss | accuracy | perplexity |  time |
| ----: | ---------: | ---------: | -------: | ---------: | ----: |
|     0 |   4.120048 |   3.912788 | 0.299565 |  50.038246 | 11:39 |

This model takes a while to train, so it's a good opportunity to talk about saving intermediary results.

这个模型需要一段时间来训练，所以这是讨论保存中间结果的好时机。

### Saving and Loading Models

### 保存和加载模型

You can easily save the state of your model like so:

你能够像下面这样保存你的模型状态：

```
learn.save('1epoch')
```

This will create a file in `learn.path/models/` named *1epoch.pth*. If you want to load your model in another machine after creating your `Learner` the same way, or resume training later, you can load the content of this file with:

这会在`learn.path/models/`路径下创建一个名为*1epoch.pth*的文件。如果在另外的机器上以同样的方法创建了你的`学习器`你希望加载你的模型，或恢复先前的训练，你能够使用一下操作加载这个文件的内容：

```
learn = learn.load('1epoch')
```

Once the initial training has completed, we can continue fine-tuning the model after unfreezing:

一旦初始的训练完成，我们就能够持续的微调解冻后的模型：

```
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
```

| epoch | train_loss | valid_loss | accuracy | perplexity |  time |
| ----: | ---------: | ---------: | -------: | ---------: | ----: |
|     0 |   3.893486 |   3.772820 | 0.317104 |  43.502548 | 12:37 |
|     1 |   3.820479 |   3.717197 | 0.323790 |  41.148880 | 12:30 |
|     2 |   3.735622 |   3.659760 | 0.330321 |  38.851997 | 12:09 |
|     3 |   3.677086 |   3.624794 | 0.333960 |  37.516987 | 12:12 |
|     4 |   3.636646 |   3.601300 | 0.337017 |  36.645859 | 12:05 |
|     5 |   3.553636 |   3.584241 | 0.339355 |  36.026001 | 12:04 |
|     6 |   3.507634 |   3.571892 | 0.341353 |  35.583862 | 12:08 |
|     7 |   3.444101 |   3.565988 | 0.342194 |  35.374371 | 12:08 |
|     8 |   3.398597 |   3.566283 | 0.342647 |  35.384815 | 12:11 |
|     9 |   3.375563 |   3.568166 | 0.342528 |  35.451500 | 12:05 |

Once this is done, we save all of our model except the final layer that converts activations to probabilities of picking each token in our vocabulary. The model not including the final layer is called the *encoder*. We can save it with `save_encoder`:

一旦这个工作做完，我保存除了最后一层之外的所有模型，最后一层是转化激活为我们词汇表中每个选中标记的概率。不包含最后一层的模型被称为*编码器*。我们能够用`save_encoder`来保存它：

```
learn.save_encoder('finetuned')
```

> jargon: Encoder: The model not including the task-specific final layer(s). This term means much the same thing as *body* when applied to vision CNNs, but "encoder" tends to be more used for NLP and generative models.

> 术语：编码器：不包含特定任务最后层的模型。当应用到视觉卷积神经网络时，这个术语表示的意义与*身体*差不多，但“编码器”倾向更多用于自然语言处理和生成模型。

This completes the second stage of the text classification process: fine-tuning the language model. We can now use it to fine-tune a classifier using the IMDb sentiment labels.

现在完成了文本分割过程的第二阶段：微调语言模型。我们现在能够使用IMDb情绪标签和这个模型来微调一个分类器。

### Text Generation

### 文本生成

Before we move on to fine-tuning the classifier, let's quickly try something different: using our model to generate random reviews. Since it's trained to guess what the next word of the sentence is, we can use the model to write new reviews:

在我们继续微调分类器之彰，让我们快速尝试一些不一样的事情：使用我们的模型来生成随机评论。因为它被训练来猜测句子的一个词是什么，我们能够使用模型来写新的评论：

```
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
```

```
print("\n".join(preds))
```

Out: [i liked this movie because of its story and characters . The story line was very strong , very good for a sci - fi film . The main character , Alucard , was very well developed and brought the whole story
i liked this movie because i like the idea of the premise of the movie , the ( very ) convenient virus ( which , when you have to kill a few people , the " evil " machine has to be used to protect]()

As you can see, we add some randomness (we pick a random word based on the probabilities returned by the model) so we don't get exactly the same review twice. Our model doesn't have any programmed knowledge of the structure of a sentence or grammar rules, yet it has clearly learned a lot about English sentences: we can see it capitalizes properly (*I\* is just transformed to \*i* because our rules require two characters or more to consider a word as capitalized, so it's normal to see it lowercased) and is using consistent tense. The general review makes sense at first glance, and it's only if you read carefully that you can notice something is a bit off. Not bad for a model trained in a couple of hours!

正如你看到的，我们添加了一些随意性（基于模型返回的概率我们选取了随机词），所以我们不能够获得两个完全相同的评论。我们的模型没有任何设定的句子结构的知识和语法规则，然而它已经清晰的学会了一些英文：我们能够看到它合理的大写（只是*I\**转换为*i\**，因为我们的规则需要两个或以上的特征才会认为一个词需要大写，所以看到它是小写，这是正常的）和使用了一致的时态。乍一看生成的评论还算合理，但是当你认真阅读后你会注意到它有点不合理。对于几个小时的模型训练结果不算太糟糕！

But our end goal wasn't to train a model to generate reviews, but to classify them... so let's use this model to do just that.

但是我们的最终目标不是训练一个生成评论的模型，而是来分类评论...  那么，让我们用这个模型做这个工作吧。

### Creating the Classifier DataLoaders

### 创建分类器DataLoader

We're now moving from language model fine-tuning to classifier fine-tuning. To recap, a language model predicts the next word of a document, so it doesn't need any external labels. A classifier, however, predicts some external label—in the case of IMDb, it's the sentiment of a document.

现在我们从语言模型微调移到分类器微调。总的来说，语言模型预测文档的下个词，所以它不需要任何外部标签。然而，一个分类器预测一些外部标签，在IMDb案例中，它是文档的情绪。

This means that the structure of our `DataBlock` for NLP classification will look very familiar. It's actually nearly the same as we've seen for the many image classification datasets we've worked with:

这意味着我们自然语言处理分类`DataBlock`结构看起来会非常熟悉。它实际上与我们见过的很多所处理的图像分类数据集几乎相同：

```
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
```

Just like with image classification, `show_batch` shows the dependent variable (sentiment, in this case) with each independent variable (movie review text):

与图像分类一样，`show_batch`显示了自变量（电影评论文本）的因变量（在这个例子中，情绪）：

```
dls_clas.show_batch(max_n=3)
```

|      |                                                         text | category |
| ---: | -----------------------------------------------------------: | -------: |
|    0 | xxbos i rate this movie with 3 skulls , only coz the girls knew how to scream , this could 've been a better movie , if actors were better , the twins were xxup ok , i believed they were evil , but the eldest and youngest brother , they sucked really bad , it seemed like they were reading the scripts instead of acting them … . spoiler : if they 're vampire 's why do they freeze the blood ? vampires ca n't drink frozen blood , the sister in the movie says let 's drink her while she is alive … .but then when they 're moving to another house , they take on a cooler they 're frozen blood . end of spoiler \n\n it was a huge waste of time , and that made me mad coz i read all the reviews of how |      neg |
|    1 | xxbos i have read all of the xxmaj love xxmaj come xxmaj softly books . xxmaj knowing full well that movies can not use all aspects of the book , but generally they at least have the main point of the book . i was highly disappointed in this movie . xxmaj the only thing that they have in this movie that is in the book is that xxmaj missy 's father comes to xxunk in the book both parents come ) . xxmaj that is all . xxmaj the story line was so twisted and far fetch and yes , sad , from the book , that i just could n't enjoy it . xxmaj even if i did n't read the book it was too sad . i do know that xxmaj pioneer life was rough , but the whole movie was a downer . xxmaj the rating |      neg |
|    2 | xxbos xxmaj this , for lack of a better term , movie is lousy . xxmaj where do i start … … \n\n xxmaj cinemaphotography - xxmaj this was , perhaps , the worst xxmaj i 've seen this year . xxmaj it looked like the camera was being tossed from camera man to camera man . xxmaj maybe they only had one camera . xxmaj it gives you the sensation of being a volleyball . \n\n xxmaj there are a bunch of scenes , haphazardly , thrown in with no continuity at all . xxmaj when they did the ' split screen ' , it was absurd . xxmaj everything was squished flat , it looked ridiculous . \n\n xxmaj the color tones were way off . xxmaj these people need to learn how to balance a camera . xxmaj this ' movie ' is poorly made , and |      neg |

Looking at the `DataBlock` definition, every piece is familiar from previous data blocks we've built, with two important exceptions:

- `TextBlock.from_folder` no longer has the `is_lm=True` parameter.
- We pass the `vocab` we created for the language model fine-tuning.

看一下`DataBlock`的定义，每一部分与之前我们创建的数据块是相同的，除了两个重要例外：

- `TextBlock.from_folder`不再有`is_lm=True`的参数。
- 我们为语言模型微调传递我们所创建的`词汇表`。

The reason that we pass the `vocab` of the language model is to make sure we use the same correspondence of token to index. Otherwise the embeddings we learned in our fine-tuned language model won't make any sense to this model, and the fine-tuning step won't be of any use.

我们传递预测模型的`词汇表`的原因是确保我们使用相相符的标记来索引。否则在我们微调语言模型中学到的嵌入对这个模型没有任何意义，且微调步骤不会有任何用处。

By passing `is_lm=False` (or not passing `is_lm` at all, since it defaults to `False`) we tell `TextBlock` that we have regular labeled data, rather than using the next tokens as labels. There is one challenge we have to deal with, however, which is to do with collating multiple documents into a mini-batch. Let's see with an example, by trying to create a mini-batch containing the first 10 documents. First we'll numericalize them:

传递`is_lm=False`（或压根不传递`is_lm`，因为它默认为`False`）我们告诉`TextBlock`我们有规则的标记后的数据，而不是使用下一个标记作为标签。然而，有一个我们必须处理的挑战是依次把多个文本整理到一个小批次。让我们看一个例子，创建一个包含头10个文档的小批次。首先我们会数值化他们：

```
nums_samp = toks200[:10].map(num)
```

Let's now look at how many tokens each of these 10 movie reviews have:

现在我们看一下这10个电影评论每个有多个标记：

```
nums_samp.map(len)
```

Out: (#10) [228,238,121,290,196,194,533,124,581,155]

Remember, PyTorch `DataLoader`s need to collate all the items in a batch into a single tensor, and a single tensor has a fixed shape (i.e., it has some particular length on every axis, and all items must be consistent). This should sound familiar: we had the same issue with images. In that case, we used cropping, padding, and/or squishing to make all the inputs the same size. Cropping might not be a good idea for documents, because it seems likely we'd remove some key information (having said that, the same issue is true for images, and we use cropping there; data augmentation hasn't been well explored for NLP yet, so perhaps there are actually opportunities to use cropping in NLP too!). You can't really "squish" a document. So that leaves padding!

记住，PyTorch`DataLoader`需要整理一个批次中的所有数据项为一个单张量，且这个单张量有一个固定的形态（即，它在每个坐标上有一些特定的长度，且所有数据项必须是一致的）。这听起来很熟悉：对于图片我们有相同的问题。在那种情况下，我们使用裁剪、填充，和/或挤压来使得所有的输入是相同的尺寸。对于文档裁剪可能不是一个好的想法，因为可能我们会移除一些关键信息（话说回来，我们在图像上使用裁剪，也存在相同的问题；数据增强对于自然语言处理还没有很好的探索，所以也许真的的机会在自然语言处理中也使用裁剪呢！）。我们不能真的“挤压”一个文档。所以就剩下填充了！

We will expand the shortest texts to make them all the same size. To do this, we use a special padding token that will be ignored by our model. Additionally, to avoid memory issues and improve performance, we will batch together texts that are roughly the same lengths (with some shuffling for the training set). We do this by (approximately, for the training set) sorting the documents by length prior to each epoch. The result of this is that the documents collated into a single batch will tend of be of similar lengths. We won't pad every batch to the same size, but will instead use the size of the largest document in each batch as the target size. (It is possible to do something similar with images, which is especially useful for irregularly sized rectangular images, but at the time of writing no library provides good support for this yet, and there aren't any papers covering it. It's something we're planning to add to fastai soon, however, so keep an eye on the book's website; we'll add information about this as soon as we have it working well.)

我们会扩大最短文本使得它们都一样的尺寸。我们使用一个模型会忽略的特殊的填充标记来做这个事情。另外，为避免内存问题和改善性能，我们会批处理文本到大致相同的长度（对训练集做一些整理）。在每个周期前我们通过（大约对训练集进行）通过长度对文档进行排序做到这一点。这样的结果是文档整理为一个倾向相似长度的单一批次。我们不会填充每个批次为相同的尺寸，但相替代的会使用每个批次中最大尺寸文档作为目标尺寸。（它与图像做的事情是相似的，其对于规则长方形图像尺寸尤为有用，但是在编写本书的时候还没有库为此提供良好的支持，且还没有任何论文覆盖这个问题。因此我们计划尽快添加这个功能到fastai，所以随时留意本书的网站。一旦这个功能运行良好我们就会添加相关信息。）

The sorting and padding are automatically done by the data block API for us when using a `TextBlock`, with `is_lm=False`. (We don't have this same issue for language model data, since we concatenate all the documents together first, and then split them into equally sized sections.)

We can now create a model to classify our texts:

当使用`TextBlock`时利用`is_lm=False`参数，排序和填充通过数据块API为我们自己处理了。（对于语言模型数据我们没有相同的问题，因此我们首先串联所有文档在一起，然后把它们分割为相同尺寸的切片。）

现在我们能够创建一个模型来分类我们的文本了：

```
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()
```

The final step prior to training the classifier is to load the encoder from our fine-tuned language model. We use `load_encoder` instead of `load` because we only have pretrained weights available for the encoder; `load` by default raises an exception if an incomplete model is loaded:

完成训练分类器的最后一步是从我们的微调语言模型中加载编码器。我们使用`load_encoder`来替代`load`，因为对于编码器我们只有可获取的预训练权重。如果加载一个未完成的模型，`load`会产生异常：

```
learn = learn.load_encoder('finetuned')
```