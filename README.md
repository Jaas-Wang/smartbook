# 人人可学的深度学习

## 项目简介

本项目是基于 Jeremy Howard 与 Sylvain Gugger 所著的「Deep Learning for Coders with fastai & PyTorch：Al Applications Without a PhD」的开源版 [Fastbook](https://github.com/fastai/fastbook)，**感谢 [Jeremy Howard](https://jeremy.fast.ai) 对项目的授权与支持。**

<img alt="" width="500" caption="" id="img_spect" src="https://user-images.githubusercontent.com/10573092/234517915-a592b706-a6bb-4778-87e5-48944715cf33.jpg">

本项目由**单博士**与**王博士**共同创建并维护
- **单博士** 承担了同 Jeremy Howard 对接，及翻译工作。
- **王博士** 负责翻译、校对、技术、评审工作。

本项目是讲授深度学习快速实现的教程，并包含工程代码。用于帮助那些有志挑战深度学习，但苦于没有深厚的工程或数学背景，及英文不是非常精通的朋友们。

很多人认为想要在深度学习上取得很好的成果，需要各种难以获取的资源，但是通过学习本教程您会发现这种想法是错误的。下表列举了几项你绝对不需要做世界级深度学习的内容。

|     AI神话（不需要）     |                     真相                     |
| :--------------------: | :------------------------------------------: |
|   大量专业数学知识   |         高中数学就足够了         |
|       海量数据       | 我们曾见过用不到50条的数据取得破纪录结果 |
| 大量昂贵的算力 |        你可以免费获得最先进的工作成果        |

记住，你不需要任何特定的学术背景就能在深度学习中取得成功。许多重要的突破都是由没有博士学位的人在研究和实践中取得的，比如这篇论文[“基于深度卷积生成对抗网络的无监督表示学习”](https://arxiv.org/abs/1511.06434)，这是过去十年中最具影响力的论文之一，引用次数超过5000次，是亚历克·雷德福（Alec Radford）在读本科的时候撰写的。即使在特斯拉，他们正在努力解决制造自动驾驶汽车这一极其艰巨的挑战，首席执行官埃隆·马斯克说:
> “绝对不要求必须具有博士学位。重要的是对人工智能的深刻理解和具备采用切实有效的方法来实现神经网络的能力（后者才是真正的难点）。高中毕业也无所谓的。”
>
然而，要想成功，你需要做的是将你在本书中学到的东西应用到个人项目中，并始终坚持下去。

## 配置运行环境

本项目对硬件的要求不是特别挑剔，如动手能力强可以选择自己喜欢的硬件。**但本项目是讲授深度学习的，为了聚焦学习内容本身，避免出现因环境问题而导致实践代码训练模型出错（真的会出现）而分散学习精力，强烈建议使用配有英伟达GPU的计算机。**

1. 利用Anaconda配置一个虚拟运行环境，安装教程参见[机器学习环境配置系列三之Anaconda](https://www.cnblogs.com/jaww/p/9846092.html)。如果感觉Anaconda太大，且对命令交互情有独钟，可以安装miniforge来配置环境，可达到同样的效果。
2. 创建您的运行环境

   ```sh
   conda create -n 环境名 python=版本号
   ```

   如，定义环境名为**smartbook**，使用python的版本为3.10.6:

   ```sh
   conda create -n smartbook python=3.10.6
   ```

3. 激活您的运行环境

   ```sh
   conda activate 环境名
   ```

   如，激活命名为**smartbook**的环境：

   ```sh
   conda activate smartbook
   ```

4. 安装所需的依赖包。

    ```sh
    pip install -r requirements.txt
    ```
    如果感觉安装速度慢，可以使用国内源，如下所示

    ```sh
    pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

    在运行本项目实践代码时，如提示依赖包缺失的问题，可以单独安装。

    ```sh
    pip install 依赖包名 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
    以安装fastbook依赖包为例，如下所示：
   ```shell
    pip install fastbook -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. 安装Jupyter Notebook命令。

   ```sh
   pip install jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

6. 进入项目文件夹。
   ```sh
   cd 项目文件夹
   ```
   **下面是进入项目文件夹的命令，具体命令根据您的系统不同而不同。**

   **Windows系统：**
   如项目文件目录在：D:\Projects\Smartbook，命令如下：
   ```sh
     cd D:\Projects\Smartbook
     ```
   **MacOS、Linux、Ubuntu系统：**
   如项目文件目录在：/Users/用户名/Projects/Smartbook，命令如下：
   ```sh
     cd /Users/用户名/Projects/Smartbook
   ```
7. 现在可以启动您的Jupyter来运行本项目了，命令如下：

   ```sh
   jupyter lab
   ```

   > **⚠️注意：务必进入本项目文件夹后运行Jupyter，否则您会找不到运行文件。**
8. 截至第7步您可以在本地环境运行本项目了。因本项目需要运行模型训练实践代码，对机器性能有一定的要求，如果您有自己的GPU服务器，并在上面配置运行环境，需要远程登陆Jupyter服务来运行本项目，可以参见[机器学习环境配置系列六之jupyter notebook远程访问](https://www.cnblogs.com/jaww/p/9846491.html)配置教程。
  
## 目录

[第一章 开启深度学习之旅](./Smartbook_01_intro.ipynb)（完成）

[第二章 产品](./Smartbook_02_production.ipynb)（校对中）

[第三章 数据伦理]()（未开始）

[第四章 mnist基础]()（未开始）

[第五章 基于宠物品种的图像分类]()（未开始）

[第六章 多标签分类]()（未开始）

[第七章 数据尺寸和测试数据增强]()（未开始）

[第八章 协同过滤]()（未开始）

[第九章 表格模型]()（未开始）

[第十章 自然语言处理]()（未开始）

[第十一章 数据处理中级API]()（未开始）

[第十二章 自然语言处理深潜]()（未开始）

[第十三章 卷积]()（未开始）

[第十四章 残差网络]()（未开始）

[第十五章 应用架构深入研究]()（未开始）

[第十六章 加速随机梯度下降]()（未开始）

[第十七章 基础神经网络]()（未开始）

[第十八章 类激活映射]()（未开始）

[第十九章 学习器]()（未开始）

[第二十章 总结思考]()（未开始） 

## 共同成长
如您希望与更多的伙伴交流切磋，可扫描下方微信二维码申请加群。

**PS：务必扫码添加备注：【人工智能学习交流】**

<img alt="" width="300" caption="" id="img_spect" src="https://user-images.githubusercontent.com/10573092/234518086-63731a4e-b31a-40f9-901b-8e3c3727e518.jpg">

**也可关注公众号，获取最新AI资讯**

<img alt="" width="600" caption="" id="green" src="https://github.com/user-attachments/assets/d07157fd-57ba-4ce4-86d7-06c155e92a8d">
