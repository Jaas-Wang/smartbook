# A Language Model from Scratch

# 从零开始一个语言模型

We're now ready to go deep... deep into deep learning! You already learned how to train a basic neural network, but how do you go from there to creating state-of-the-art models? In this part of the book we're going to uncover all of the mysteries, starting with language models.

现在我们准备深入研究深度学习！你已经学习了如何训练一个基础的神经网络，但是你如何从这里创建一个先进的模型呢？在本部分我们会揭示所有语言模型的秘密。

You saw in <chapter_nlp> how to fine-tune a pretrained language model to build a text classifier. In this chapter, we will explain to you what exactly is inside that model, and what an RNN is. First, let's gather some data that will allow us to quickly prototype our various models.

在<章节：自然语言处理>中我们学习了如何微调一个预训练语言模型以创建一个文本分类器。在这个章节，我们会给你解释这个模型内部究竟什么内容，及递归神经网络是什么。首先让我们收集一些数据，以使我们快速构建各种模型原型。

## The Data

## 数据

Whenever we start working on a new problem, we always first try to think of the simplest dataset we can that will allow us to try out methods quickly and easily, and interpret the results. When we started working on language modeling a few years ago we didn't find any datasets that would allow for quick prototyping, so we made one. We call it *Human Numbers*, and it simply contains the first 10,000 numbers written out in English.

每当我们开始处理一个新的问题，我们总是首先思考能让我们快速且容易的尝试我们方法的最简单的数据集和可解释的结果。当我们几年前开始从事语言建模时，我们没有找到任何允许原型制作的任何数据集，所以我们做了一个。我们称它为*人类数字*，它简单的包含了用英文写的头10,000个数字。

> j: One of the most common practical mistakes I see even amongst highly experienced practitioners is failing to use appropriate datasets at appropriate times during the analysis process. In particular, most people tend to start with datasets that are too big and too complicated.

> 杰：即使在具有丰富经验从业人员中我见过的最常见的实践问题之一是分析处理期间无法在使用的时间使用合适的数据集。在实践中，多数人倾向用太大和太复杂的数据集开始。

We can download, extract, and take a look at our dataset in the usual way:

我们能够用常用的方法下载、抽取，并查看我们的数据集：

```
from fastai.text.all import *
path = untar_data(URLs.HUMAN_NUMBERS)
```

```
#hide
Path.BASE_PATH = path
```

```
path.ls()
```

Out: (#2) [Path('train.txt') , Path('valid.txt')]

Let's open those two files and see what's inside. At first we'll join all of the texts together and ignore the train/valid split given by the dataset (we'll come back to that later):

让我们打开那两个文件和查看内部是什么。首先我们会连接起来所有文本并忽略数据集给出的训练/验证分割（我们稍后会返回到这个话题）：

```
lines = L()
with open(path/'train.txt') as f: lines += L(*f.readlines())
with open(path/'valid.txt') as f: lines += L(*f.readlines())
lines
```

Out: (#9998) ['one \n' , 'two \n' , 'three \n' , 'four \n' , 'five \n' , 'six \n' , 'seven \n' , 'eight \n' , 'nine \n' , 'ten \n'...]

We take all those lines and concatenate them in one big stream. To mark when we go from one number to the next, we use a `.` as a separator:

我们取了所有那些行并串连他们为一个大流。当我们标记从一个数字到下个数字时，我们使用`.`作为分割器：

```
text = ' . '.join([l.strip() for l in lines])
text[:100]
```

Out: 'one . two . three . four . five . six . seven . eight . nine . ten . eleven . twelve . thirteen . fo'

We can tokenize this dataset by splitting on spaces:

我们能够通过在空格上分割来标记这个数据集：

```
tokens = text.split(' ')
tokens[:10]
```

Out: ['one' , '.' , 'two' , '.' , 'three' , '.' , 'four' , '.' , 'five' , '.']

To numericalize, we have to create a list of all the unique tokens (our *vocab*):

我们必须创建一个所有唯一标记（我们的*词汇*）的列表来数值化：

```
vocab = L(*tokens).unique()
vocab
```

Out[ ]: (#30) ['one' , '.' , 'two' , 'three' , 'four' , 'five' , 'six' , 'seven' , 'eight' , 'nine'...]

Then we can convert our tokens into numbers by looking up the index of each in the vocab:

然后我们能够通过查找词汇表中每个索引来转换标记为数字：

```
word2idx = {w:i for i,w in enumerate(vocab)}
nums = L(word2idx[i] for i in tokens)
nums
```

Out: (#63095) [0, 1, 2, 1, 3, 1, 4, 1, 5, 1...]

Now that we have a small dataset on which language modeling should be an easy task, we can build our first model.

现在我们有了一个小数据集，在它上面语言建模应该是一个很容易的任务，我们可以创建我们的第一个模型了。

## Our First Language Model from Scratch

## 从零开始我们的第一个语言模型

One simple way to turn this into a neural network would be to specify that we are going to predict each word based on the previous three words. We could create a list of every sequence of three words as our independent variables, and the next word after each sequence as the dependent variable.

将其转换为神经网络的一个简单方法是指定我们基于前三个单词来预测每个单词。我们能够创建每个都包含三个词的序列列表作为自变量，每个序列后面的其后单词作为因变量。

We can do that with plain Python. Let's do it first with tokens just to confirm what it looks like:

我们能够用纯Python来做这个事情。让我们只是用标记完成这个工作，只是确认一下它看起来像什么：

```
L((tokens[i:i+3], tokens[i+3]) for i in range(0,len(tokens)-4,3))
```

Out: (#21031) [(['one' , '.' , 'two'], '.') , (['.' , 'three' , '.'], 'four') , (['four' , '.' , 'five'], '.') , (['.' , 'six' , '.'], 'seven') , (['seven' , '.' , 'eight'], '.') , (['.' , 'nine' , '.'], 'ten') , (['ten' , '.' , 'eleven'], '.') , (['.' , 'twelve' , '.'], 'thirteen') , (['thirteen' , '.' , 'fourteen'], '.') , (['.' , 'fifteen' , '.'], 'sixteen')...]

Now we will do it with tensors of the numericalized values, which is what the model will actually use:

现在我们用数值化值的张量做处理，它是模型实际使用的内容：

```
seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0,len(nums)-4,3))
seqs
```

Out: (#21031) [(tensor([0, 1, 2]), 1) , (tensor([1, 3, 1]), 4) , (tensor([4, 1, 5]), 1) , (tensor([1, 6, 1]), 7) , (tensor([7, 1, 8]), 1) , (tensor([1, 9, 1]), 10) , (tensor([10,  1, 11]), 1) , (tensor([ 1, 12,  1]), 13) , (tensor([13,  1, 14]), 1) , (tensor([ 1, 15,  1]), 16)...]

We can batch those easily using the `DataLoader` class. For now we will split the sequences randomly:

我们能够使用`DataLoader`类轻松的批量处理那些内容。现在，我们会随机分割序列：

```
bs = 64
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=64, shuffle=False)
```

We can now create a neural network architecture that takes three words as input, and returns a prediction of the probability of each possible next word in the vocab. We will use three standard linear layers, but with two tweaks.

现在我们能够创建一个取三个词作为输入的神经网络架构，并返回一个词汇表中每个可能下个词的概率预测。我们会使用三个标准的线性层，但是要有两个调整。

The first tweak is that the first linear layer will use only the first word's embedding as activations, the second layer will use the second word's embedding plus the first layer's output activations, and the third layer will use the third word's embedding plus the second layer's output activations. The key effect of this is that every word is interpreted in the information context of any words preceding it.

第一个调整是，第一个线性层会只用第一个词的嵌入作为激活，第二层会使用第二个词的嵌入加上第一层的输出激活，及第三层会用第三个词嵌入加上第二层的输出激活。这样做的关键效果是每个词都在其前面的信息语境中进行解释。

The second tweak is that each of these three layers will use the same weight matrix. The way that one word impacts the activations from previous words should not change depending on the position of a word. In other words, activation values will change as data moves through the layers, but the layer weights themselves will not change from layer to layer. So, a layer does not learn one sequence position; it must learn to handle all positions.

第二个调整是这三层的任何一层会使用相同的权重矩阵。一个词影响之前词激活的方式应该不会根据词的位置而改变。换句话说，激活值在数据通过层时会改变，但是层的权重自身不会逐层改变。因此，一个层不会学习一个序列的位置，它必须学习处理所有的位置。

Since layer weights do not change, you might think of the sequential layers as "the same layer" repeated. In fact, PyTorch makes this concrete; we can just create one layer, and use it multiple times.

因为层权重不会发生改变，你可能会认为这些序列层是作为“相同的层”的重复。实际上，PyTorch做了这个考虑。我们能够只创建一个层，且使用这个层多次。

### Our Language Model in PyTorch

### 用PyTorch创建我们的语言模型

We can now create the language model module that we described earlier:

现在我们能够创建我们先前描述的语言模型了：

```
class LMModel1(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = F.relu(self.h_h(self.i_h(x[:,0])))
        h = h + self.i_h(x[:,1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:,2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)
```

As you see, we have created three layers:

- The embedding layer (`i_h`, for *input* to *hidden*)
- The linear layer to create the activations for the next word (`h_h`, for *hidden* to *hidden*)
- A final linear layer to predict the fourth word (`h_o`, for *hidden* to *output*)

正如你看到的，我们已经创建了三个层：

- 嵌入层（`i_h`，*输入到隐藏*）
- 线性层为下个词创建激活（`h_h`，*隐藏到隐藏*）
- 最后的线性层来预测第四个单词（`h_o`，*隐藏到输出*）

This might be easier to represent in pictorial form, so let's define a simple pictorial representation of basic neural networks. <img_simple_nn> shows how we're going to represent a neural net with one hidden layer.

这可能以示意图方式更容易表达，所以让我们定义一个简单的基础神经网络示意图表示。<简单的神经网络示意图表示>显示了我们打算表达的一个带有单隐含层的神经网络。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00020.png" alt="Pictorial representation of simple neural network" width="400" caption="Pictorial representation of a simple neural network" id="img_simple_nn">
  </p>
  <p align="center">图：简单的神经网络示意图表示</p>
</div>

Each shape represents activations: rectangle for input, circle for hidden (inner) layer activations, and triangle for output activations. We will use those shapes (summarized in <img_shapes>) in all the diagrams in this chapter.

每个形状所代表的激活：矩形为输入，原型为隐含（内部的）层激活，三角形为输出激活。在本章节的所有示意图中我们会使用这些形状（<示意图中所使用的图形>做了总结）。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00021.png" alt="Shapes used in our pictorial representations" width="200" id="img_shapes" caption="Shapes used in our pictorial representations">
  </p>
  <p align="center">图：示意图中所使用的图形</p>
</div>

An arrow represents the actual layer computation—i.e., the linear layer followed by the activation function. Using this notation, <lm_rep> shows what our simple language model looks like.

箭头线表示实际的层计算。即，线性层后面跟随着激活函数。使用这一标记，<基础语言模型表示>图中展示了我们的简单的语言模型看起来是什么样子。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00022.png" alt="Representation of our basic language model" width="500" caption="Representation of our basic language model" id="lm_rep" >
  </p>
  <p align="center">图：基础语言模型表示</p>
</div>

To simplify things, we've removed the details of the layer computation from each arrow. We've also color-coded the arrows, such that all arrows with the same color have the same weight matrix. For instance, all the input layers use the same embedding matrix, so they all have the same color (green).

Let's try training this model and see how it goes:

简化了的事情是我们已经移除了每个箭头的层计算细节。我们也彩色编码了箭头线，这样有相同颜色的箭头线有着相同的权重矩阵。例如，所有的输入层使用了相同的嵌入矩阵，所以它们都有着相同的颜色（绿色）。

让我们尝试训练这个模型并查看效果如何：

```
learn = Learner(dls, LMModel1(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   1.824297 |   1.970941 | 0.467554 | 00:02 |
|     1 |   1.386973 |   1.823242 | 0.467554 | 00:02 |
|     2 |   1.417556 |   1.654497 | 0.494414 | 00:02 |
|     3 |   1.376440 |   1.650849 | 0.494414 | 00:02 |

To see if this is any good, let's check what a very simple model would give us. In this case we could always predict the most common token, so let's find out which token is most often the target in our validation set:

看看这是否有用，让我们检查一下一个非常简单的模型会提供给我们什么。在本例子中我们总会预测最常见的标记，所以让我们找出在我们的验证集中哪个标记是最常见的目标：

```
n,counts = 0,torch.zeros(len(vocab))
for x,y in dls.valid:
    n += y.shape[0]
    for i in range_of(vocab): counts[i] += (y==i).long().sum()
idx = torch.argmax(counts)
idx, vocab[idx.item()], counts[idx].item()/n
```

Out: (tensor(29), 'thousand', 0.15165200855716662)

The most common token has the index 29, which corresponds to the token `thousand`. Always predicting this token would give us an accuracy of roughly 15%, so we are faring way better!

最常见的标记索引是29，其与标记`thousand`相关联。总是预测这个标高会给我们大约15%的准确度，所以我们将会好的好！

> A: My first guess was that the separator would be the most common token, since there is one for every number. But looking at `tokens` reminded me that large numbers are written with many words, so on the way to 10,000 you write "thousand" a lot: five thousand, five thousand and one, five thousand and two, etc. Oops! Looking at your data is great for noticing subtle features and also embarrassingly obvious ones.

> 亚：我的第一个猜想是分割器也许是最常见的标记，因为每个数字都有一个。但是查看`标记`让我想起大的数字用很多单词写出来的，所以以这样的方式对于10,000这个数值，你要写很多个`thousand`： five thousand, five thousand and one, five thousand and two，等等。哎呦！查看你的数据对于注意细微特征很有益处的，同样尴尬也是显而易见的。

This is a nice first baseline. Let's see how we can refactor it with a loop.

这是非常好的第一个基线。让我们看一下如何用循环重构它。

### Our First Recurrent Neural Network

### 我们首个递归神经网络

Looking at the code for our module, we could simplify it by replacing the duplicated code that calls the layers with a `for` loop. As well as making our code simpler, this will also have the benefit that we will be able to apply our module equally well to token sequences of different lengths—we won't be restricted to token lists of length three:

看我们模型的代码，我们能够使用一个`for`循环替换调用层的重复代码来简化它。同样这使得我们的代码更加简单，这也会有一个收益，就是我们能够应该我们的模型同样很好的标记不同长度的序列。我们不会被限制在标记长度为三的列表：

```
class LMModel2(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)
```

Let's check that we get the same results using this refactoring:

让我们检查一下，使用这个重构的代码我们获得相同的结果：

```
learn = Learner(dls, LMModel2(len(vocab), 64), loss_func=F.cross_entropy, 
                metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
```

| epoch | train_loss | valid_loss | accuracy |  time |
| ----: | ---------: | ---------: | -------: | ----: |
|     0 |   1.816274 |   1.964143 | 0.460185 | 00:02 |
|     1 |   1.423805 |   1.739964 | 0.473259 | 00:02 |
|     2 |   1.430327 |   1.685172 | 0.485382 | 00:02 |
|     3 |   1.388390 |   1.657033 | 0.470406 | 00:02 |

We can also refactor our pictorial representation in exactly the same way, as shown in <basic_rnn> (we're also removing the details of activation sizes here, and using the same arrow colors as in <lm_rep>).

我们也能够以完全相同的方式重构在我们的示意图表示，如图<基础递归神经网络>所示（在这里我们同样移除了激活尺寸的细节，并用不用 了与图<基础语言模型表示>相同的箭头线颜色）。

<div style="text-align:center">
  <p align="center">
    <img src="./_v_images/att_00070.png" alt="Basic recurrent neural network" width="400" caption="Basic recurrent neural network" id="basic_rnn"  >
  </p>
  <p align="center">图：基础递归神经网络</p>
</div>

You will see that there is a set of activations that are being updated each time through the loop, stored in the variable `h`—this is called the *hidden state*.

> Jargon: hidden state: The activations that are updated at each step of a recurrent neural network.

我们会看到，有一系统的激活每次通过循环被更新，存储在变量`h`中，这被称为*隐含状态*。

> 术语：隐含状态：激活在每一步的递归神经网络中被更新。

A neural network that is defined using a loop like this is called a *recurrent neural network* (RNN). It is important to realize that an RNN is not a complicated new architecture, but simply a refactoring of a multilayer neural network using a `for` loop.

> A: My true opinion: if they were called "looping neural networks," or LNNs, they would seem 50% less daunting!

神经网络使用一个循环来定义，就橡这个被称为的*递归神经网络*（RNN）。意识到RNN不是一个复杂的新架构，而是使用 一个`for`循环的多层神经网络的简单重构，这是很重要的。

> 亚：我的真实观点：如果它们被称为“循环神经网络”，或LNN，它们似乎减少了50%的畏惧！

Now that we know what an RNN is, let's try to make it a little bit better.

现在我们知道了RNN是什么，让我们尝试让他更好一些。

## Improving the RNN

## 改善递归神经网络

Looking at the code for our RNN, one thing that seems problematic is that we are initializing our hidden state to zero for every new input sequence. Why is that a problem? We made our sample sequences short so they would fit easily into batches. But if we order the samples correctly, those sample sequences will be read in order by the model, exposing the model to long stretches of the original sequence.

查看我们的RNN代码，有一个事情好像有问题那就是对每个新输入序列我们初始化隐含状态为零。为什么那是一个问题呢？我们缩短了我们的样本序列，它们会很容易的融合到批次中。但是如果我们正确的排序了样本，那些样本序列会被模型按照顺序读取，从而暴露模型在原始序列长度片段下。

Another thing we can look at is having more signal: why only predict the fourth word when we could use the intermediate predictions to also predict the second and third words?

我们能够看到的另外事情是有更多的信号：当我们使用中间预测结果来同样预测第二和第三个词的时候，为什么只预测第四个词？

Let's see how we can implement those changes, starting with adding some state.

让我们学习，我们能够如何实现这些变化，从添加一些状态开始。