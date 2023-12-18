---
layout:     post
title:      打开数据通向模型的大门——Embedding
subtitle:   Embedding是数据进入模型的入场券，数据是怎样处理的呢？
date:       2023-12-12
author:     SF
header-img: img/Embedding-post-bg.jpeg
catalog: true
tags:
    - Embedding
    - LLM
    - Data Processing
---


# 前言--我的懵逼时刻
作为一个学习大模型的新手，在看过一定数量的论文、博客、视频之后，我对一些词的意思很迷惑！Embedding，encoding，tokenization,encoder,one-hot encoding和embedding是否有关，tokenization中的encode和embedding是否相关，和transformer中的encoder是否有关，tokenization和embedding之间的关系是什么？说实话，我看的越多，我越疑惑。这还只是英文方面的字词。看了许多的中文博客，看到了词嵌入（embedding）,词向量(word2cvec)，文本token化（tokenization），文本向量化编码（encoding），编码（encoder）,解码（decoder），我更疑惑了。

有的博客会说，embedding(这里先特指NLP领域)是将文字表征为低维向量，所以从文字到向量的整个过程都是embedding在做吗？有的博客也会说，one-hot编码作为典型的文本向量化方法，将文字转化成编码；而论文也会讲，word2vec可以将学习文字间的关系，将文字表征成低维向量...等等等。我从这些博客里面学到了很多，但也积累了很多的问题，这么多仿佛都是在做一件事，将文字转成向量？one-hot编码是？embedding是？word2vec是？甚至于，transformer中的encoder是？

于是，我决定自己针对这个问题，好好做做功课，以笔记的形式梳理我的思路，记录我的理解。

# 问题聚焦
不仅是语言模型，还是其他模型。以现在主要的人工智能两大领域来说，NLP和CV，各种模型的结构、大小各不相同。但是，有一点不变的是，**模型是对数字矩阵或者向量等进行运算处理的**。那么模型在进行训练或者推理之前，第一步永远都是数据的处理————如何获得进入模型的入场券 **向量化数据**。

因此，这篇博客的主要内容就是在将，数据进入模型前的那些事。这里，我将主要内容放在embedding上，这里的embedding指的是广义的embedding，即如何将数据映射成一个能够表达数据内、数据间信息的低维稠密向量（或矩阵）。
整个博客主要解释以下几个问题：

- 文本向量化的概念是什么？有哪些不同的含义导致上述我的理解错乱？
- 不同文本向量化是不是都是从文字开始，到得到向量为止？
- 常说的NLP LLM相关的文本向量化流程是什么？
- CV领域的向量化有哪些？
- 其他的向量化代表有哪些？

# 阅读提要
根据问题，不难看出，我想从文字的向量化入手，然后转到图像方面，最后对其他的向量化过程进行总结。这里我为了分类方便，将文字的向量化部分统称为word embedding,图像向量化部分统称为image embedding,其他方面，主要介绍两种向量化应用：item embedding（对象向量化） 和 graph embedding（图网络向量化）。

# 博客开始
## Word Embedding
**首先我们来解决第一个问题**，*什么是文本向量化，文本向量化到底有哪些不同的含义？*

文本向量化，顾名思义就是，将文本用向量表示。那么表示有很多种意思，比如类似于键值对的表示方法，就像查字典一样，规定一个向量就表示一个文字或者字符。这种表示，是人为指定的表示，相互交换不同的键值，对文本的表示影响不大，只需要在文本对应的两个字符位置相互调换就行了，对其他的字符没有任何改变。这里这个向量的作用只是一个指针而已，上面所说的ont-hot encoding就是这样。我讲这种表示称为表层表征。

但是如果说，我需要这个向量能够反映不同文本间的关系，反映文本内容，那么这种表达方式做不到。

但在模型训练的时候，我们需要整个的向量不只是作为文本的一个指针，更需要向量能够传递出更多关于文本的信息，比如这句话的意思，两个词之间是否有关系等等。我们需要用向量来表达文本中的内容以及字符间的内在联系，那么这就需要另外一种向量化的方式。这里我认为这个才是word embedding。**这也解释了第二个问题的答案**。

而需要注意的是，我们进行word embedding并不是一步完成的，其实可以分为两步来完成：
- 首先找到文本的表层表征向量$\vec{X_1}$
- 利用不同的word embedding方法将这个向量映射成我们需要的地位稠密向量$\vec{X_{2}}$

可以看出，表层表征是word embedding的第一步。下面，我将依次介绍word embedding的整个过程中的关键过程及技术，**即解决第三个问题**，因为整个的技术就是按embedding的流程来进行的。
### Tokenization
#### tokenization作用
在想着如何将文字（或者语言或者其他）向量化之前，我们首先应该想到的是，组成一段话的字符是有很多个的，字符和字符的组合也是无数的。我们不可能以一段话为一个整体去分别生成一个对应的表层表征，那么我们应该如何去实现呢？想想英文中的26个字母，虽然英文中的字词多，组成的语句无数，但是基本的组成部分仍然是不变的，那么我们在进行一段话的表征时，不就是去表征各个字母的排列组合吗？那么只要我们将这些字母用向量或者数字进行表征后，整个的语句不就被表征完了吗？

因此，在NLP领域，对长字符串类型的数据，在embedding前进行的处理中，在进行表层表征的时候，首先要做的就是将一个长字符串分解成一个个的**容易被直接表征的组成单元**，而这些**容易被直接表征的组成单元**就被成为token。将长字符串分解成一个个单独的token的过程就称为tokenization（分词）。**但是需要注意的是**，tokenization做的不止这些，分解完之后的tokens还会在这一步进行表层表征，即输出的不再只是token的列表，而是一个向量。

> （这里解释一下，其实tokenization在我的认识里面也是**分为广义和狭义**两种的，狭义的tokenization就只是分词得到token列表，例如在**BertTokenizer**类中对应的函数就是*BertTokenizer.tokenize()*；而广义的tokenization还包括将token列表转化成向量，对应的函数为*BertTokenizer.encode()*,其功能与*BertTokenizer.tokenize() + BertTokenizer._convert_token_to_id()*类似，但也有点不同，那就是一些特殊符号，比如一句话的开始标志或者终止标志等，默认情况下，仅tokenize()函数是不会加入用来标注token在词语或者语句中位置的标志）。
> 
> 下面是一个示例：

```python
import torch
from transformers import BertTokenizer

sentence = 'i am songf,nice to meet you'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 分词和向量化分开
output_chars = tokenizer.tokenize(sentence)
print(output_chars)
output_tensor1 = tokenizer.convert_tokens_to_ids(output_chars)
print(output_tensor)

# 直接encode
output_tensor2 = tokenizer.encode(sentence)
print(output_tensor)
```

> 运行结果你会发现，两个函数生成的向量长度不同，原因是由特殊标志导致的。特殊标志是用来区分前后两个语句或者token在一个词内部的位置的。

#### 常见的分词方法
从token的组成部分来分，tokenization分词主要由三个level：
word-level, subword-level 和 char-level。

word-level很好理解，就是将单个的字词作为一个token，这种方式的好处是：分词简单，便于操作；但也存在劣势，容易出现OOV（out-of-vocabulary）问题，即如果出现一个词汇表中没有出现的新词时，很难找到对应的替代词，只能用‘UNK’来替代；假设词汇表中有'cat'这个词，但是没有'cats'这个词时，即使后者只是前者的复数，但利用word-level的tokenization无法发现两个词间的联系，这对于NLP来说是一个巨大的损失。

char-level就是将单个的字符作为一个token，这样的好处是任何的单词或者句子都可以表示，表示完整；例如将'i am sf'分解成['i',' ','a','m',' ','s','f']。

subword-level介于两者之间，学习语言中常用到的一些词或者词的一部分，例如'darkest'可以分成两个频率高的子词'dark'和'est'。这也是目前最常用的一种分词方法。

而在subword-level分词方法中，针对于如何来确定词汇表中的子词又有不同的方法，接下来我将主要介绍几种大模型中使用较为广泛的分词方法。

- **BPE(Byte-Pair Encoding)**

    使用这种分词方法的代表模型为GPT家族模型，其分词思想为：
    - 首先将输入语句中不同的单词隔开，在每个单词结束后加上一个单词结束符'< /w >';
    - 利用char-level的分词方法将所有的单词全部分成单字；
    - 将出现频率最高的单子合并；
    - 迭代上一步，直到结果满足条件（例如满足词汇库大小要求等）；
    
    对于可能出现的未在词汇库中的子词，用'< unk >'表示；
- **Word-Piece**
  
    word-Piece主要由Bert家族模型采用，其分词思想为：
    - 在一句话的句首添加'[CLS]',在两句话之间添加'[sep]'，'[CLS]'主要用在需要进行句子语气分类等场景下，放在句子开头，便于识别句子的类别；
    - 寻找出现最频繁的子词组合：认为两个单字出现概率和小于两个单字组成的子词出现的概率，则认为该子词出现频繁；
- **sentence-piece**
  
    这种方式的优势在于对多语言的兼容性。不同于直接对语句进行分词，其将不同语言的子词先行转化成unicode编码在进行分词，另外对一些格式字符进行转换。这一部分具体请参考[HeptaAI博客](https://zhuanlan.zhihu.com/p/631463712)

对tokenization源码感兴趣的朋友，可以学习[huggingface transformer](https://github.com/huggingface/transformers) *src/model/*中相应模型的tokenization方法。
#### 表层表征encoding
表层表征encoding就是将token转换成一一对应的向量或者矩阵的过程。其实利用数字来表示文字很常见，比如通信领域，利用01编码来表示文字。这些encoding中，数字和文字只是存在一一对应关系，但这个数字本身是和文字没有任何关系的，是人为给定的，因此称之为表层表征。

需要注意的是，不论什么编码形式，都是需要指定一一对应关系，那么就需要存储由这种一一对应关系的基本资源库存在。就像拿着货单取货一样，一个货号对应一件商品，你拿着货单总要到仓库里面才能取货。没有仓库这个货单也没有任何作用。

- **hashing encoding**
  
    hashing encoding就是利用hash方法进行的encoding方式，hash方法的特点是具有编码的单向性，可以将文字hash成一串数字，但数字反过来无法解码出文字，这也是密码学中经常用到的方法，但在语言模型中很少用到。

- **one-hot encoding**
  
    one-hot编码在模型训练中经常用到，其主要优势在于，给定词汇表后，每个token对应一个向量，这个向量中仅一位为1，其余为0，故为one-hot编码。但这种编码形式存在一个很大的问题，每个向量的长度都是词汇表的长度，而一句话会有很多个token存在，这些向量组成了一个很大的矩阵，存储信息十分稀疏，并不适合模型对其中信息的学习。于是有人将各个token向量相加而被并列，最终获得一个向量，将信息集中。这种方法将数据进行一定程度的压缩，但是其损失了token之间的次序关系，这一点在语句中是十分重要的。目前用one-hot编码的技术主要由word2vec。

- **index-based encoding**

    index-based encoding是目前使用最为广泛的一种表层表征的encoding方式。这种encoding方法要求tokenization时候形成的token词汇库满足以下格式：

    $$vacabs = \{ token1:0,token2:1,...\}$$
    
    将token当作key，每个token对应有自己的index。当将一句话进行tokenization后，依次寻找token对应的index，由此得到一个向量。

    同样这种方法也存在一个问题，那就是不同的语句的token量不同。因此每个模型其实在输入的时候都会严格的控制token数，以避免token数过多；而当token数过少时，则需要利用填充方法(padding)进行向量填充。

除上述encoding方法之外，还有很多其他的encoding方法，在此不一一赘述了。需要注意的是，不论什么encoding方式，最重要的是对应的词汇表。不同的词汇表意味着不同的映射关系，会导致同一句话的表层表征不同，对于模型来说，就意味着数据不同了。因此大家需要注意在模型微调等的时候，选择的tokenization方法和embedding方法最好来自同一模型的，这样一句话才能得到正确的表层表征（正确是相对于当前模型的embedding而言的），通过embedding才能得到对应的embedding向量表示。

### word embedding method

#### NNLM

#### word2vec
Nagetive Sampling

hierarchical softmax

CBOW

Skip-Gram

## Image Embedding

## Item Embedding

## Graph Embedding



# Reference
