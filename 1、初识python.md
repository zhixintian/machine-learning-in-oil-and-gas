#### Python简介

Python是由荷兰人吉多·范罗苏姆（Guido von Rossum，后面都称呼他为Guido）发明的一种编程语言。

#### Python的历史

1. 1989年圣诞节：Guido开始写Python语言的编译器。
2. 1991年2月：第一个Python解释器诞生，它是用C语言实现的，可以调用C语言的库函数。
3. 1994年1月：Python 1.0正式发布。
4. 2000年10月：Python 2.0发布，Python的整个开发过程更加透明，生态圈开始慢慢形成。
5. 2008年12月：Python 3.0发布，引入了诸多现代编程语言的新特性，但并不完全兼容之前的Python代码。

截至目前Python最新的版本为3.9.1

> **说明**：大多数软件的版本号一般分为三段，形如A.B.C，其中A表示大版本号，当软件整体重写升级或出现不向后兼容的改变时，才会增加A；B表示功能更新，出现新功能时增加B；C表示小的改动（例如：修复了某个Bug），只要有修改就增加C。

#### Python的优点

Python的优点很多，简单为大家列出几点。

1. 简单明确，跟其他很多语言相比，Python更容易上手。
2. 开放源代码，拥有强大的社区和生态圈。
3. 能够在Windows、macOS、Linux等各种系统上运行。

#### Python的应用领域

目前Python在**Web服务器应用开发**、云基础设施开发、**网络数据采集**（爬虫）、**数据分析**、量化交易、**机器学习**、**深度学习**、**自动化测试**、**自动化运维**等领域都有用武之地。


#### Python的安装

两步走：

1、anaconda（简单且自带多种包）的安装,进入[anaconda的官网](<https://www.anaconda.com/>),按顺序点击Get Start->Download->选择相应版本。

注意：检查安装是否成功的方法->首先，组合键Win键 + r，打开运行框，输入cmd，点击确定，打开命令提示行，输入 python，检查python是否安装成功，以及python版本；然后重新进入cmd运行窗口，输入conda info，检查环境变量是否配置成功。成功如下图所示；

![image1](https://github.com/zhixintian/machine-learning-in-oil-and-gas/blob/main/anaconda.jpg)

![image2](https://github.com/zhixintian/machine-learning-in-oil-and-gas/blob/main/huanjingpeizhi.jpg)

>**提示**：如果不显示图片，参考[链接](https://zhuanlan.zhihu.com/p/107196957)

2、pycharm的安装，在[JetBrains的官方网站](<https://www.jetbrains.com/>)上提供了PyCharm的[下载链接](<https://www.jetbrains.com/pycharm/download>)，其中社区版（Community）是免费的但功能相对弱小，专业版（Professional）功能非常强大，但需要按年或月付费使用，新用户可以试用30天时间。安装PyCharm只需要直接运行下载的安装程序，然后持续的点击“Next”（下一步）按钮就可以，安装完成后点击“Finish”（结束）按钮关闭安装向导，然后可以通过双击桌面的快捷方式来运行PyCharm。启动PyCharm之后会来到一个欢迎页，在欢迎页上我们可以选择“Create New Project”（创建新项目）、“Open”（打开已有项目）和“Get from Version Control”（从版本控制系统中检出项目）。我们可以在项目上点击鼠标右键，选择“New”，在选择“Python File”来创建Python代码文件。

注意：安装步骤可以参考[链接](<https://zhuanlan.zhihu.com/p/159394831>)。除了Pycharm这个编辑器，当然也可以使用Jupyter与IPython这两个交互式环境

#### 代码的注释

注释是编程语言的一个重要组成部分，用于在源代码中解释代码的作用从而增强程序的可读性。当然，我们也可以将源代码中暂时不需要运行的代码段通过注释来去掉，这样当你需要重新使用这些代码的时候，去掉注释符号就可以了。简单的说，**注释会让代码更容易看懂但不会影响程序的执行结果**。

Python中有两种形式的注释：

1. 单行注释：以#和空格开头，可以注释掉从`#`开始后面一整行的内容。
2. 多行注释：三个引号开头，三个引号结尾，通常用于添加多行说明性内容。

```Python
"""
第一个Python程序 - hello, world

Version: **
Author: **
"""
# print('hello, world')
print("你好，世界！")
```
