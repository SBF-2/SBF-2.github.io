---
layout:     post
title:      打开数据通向模型的大门--Embedding
subtitle:   *了解不等于理解，理解不等于掌握。当我觉得自己好像懂了transformer,pre-training,fine-tuning时，结果证明，我甚至不懂，如何开始获得模型的真实输入的！*
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

但在模型训练的时候，我们需要整个的向量不只是作为文本的一个指针，更需要向量能够传递出更多关于文本的信息，比如这句话的意思，两个词之间是否有关系等等。我们需要用向量来表达文本中的内容以及字符间的内在联系，那么这就需要另外一种向量化的方式。这里我认为这个才是word embedding。

而需要注意的是，我们进行word embedding并不是一步完成的，其实可以分为两步来完成：
- 首先找到文本的表层表征向量$\vec{X_1}$
- 利用不同的word embedding方法将这个向量映射成我们需要的地位稠密向量$\vec{X_{2}}$

可以看出，表层表征是word embedding的第一步。下面，我将依次介绍word embedding的整个过程中的关键过程及技术。
### Tokenization
### Tokenization

### 表层encode
hashing encoding

one-hot encoding

index-based encoding
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
