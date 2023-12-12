---
layout:     post
title:      打开数据通向模型的大门--Embedding
subtitle:   了解不等于理解，理解不等于掌握。当我觉得自己好像懂了transformer,pre-training,fine-tuning时，结果证明，我甚至不懂，如何开始获得模型的真实输入的！
date:       2023-12-12
author:     SF
header-img: img/Embedding-post-bg.jpeg
catalog: true
tags:
    - Embedding
    - LLM
    - Data Processing
---


# 前言--基础但致命的问题
一直以来，我都在给自己说，我学习的是LLM、多模态大模型的应用。我也看过很多相关的论文，也做过一些项目。NLP是我之前没有接触过的，直到chatGPT的出现，大语言模型吸引了我的注意力。大语言模型的强大能力让人惊呼，相信很多再看这篇文章（或许没人看）的人，都是因为LLM之类的在看这篇博客。

在我决定学习LLM相关知识的时候，我想着先学习LLM相关的各种组件，架构等。因此学习了transformer，attention；我本人学习东西比较拖泥带水，因为学了transformer，attention,我就需要往前去搞清楚这些知识的前提，我想搞清楚语言模型的一般流程以及进化历程。因此我又看了RNN，看了seq2seq，看了word2vec，不可避免的也接触到了vector database。不得不说，确实我的知识储备很欠缺。

前段时间，我觉得我对LLM有了不少的了解，知道了从NN到RNN到transformer的进化历程以及背后的原因。然后我继续学习了LLM的prompt engineering，确实是LLM中最简单的一部分。而当我觉得自己不错的时候，我准备开始好好看看GPT2的架构，想好好看看LLM是怎么样进行预训练的时候，我突然发现了一个很重要的问题，语言是如何进行训练的？

在我最开始学习LLM的时候，我会回答，embedding。but,再进一步问问，embedding是什么，embedding的输入是什么，如果是文字，文字是怎样变成能够输入模型的向量或者矩阵的？毕竟都知道，模型实质上进行的是数学运算。

这个时候，我开始混乱，因为我不知道embedding是什么，同时我又想起了word2vec，想起了one-hot encoding，也想起了encoder，我觉得他们好像很像，**作用好像都是将一个主体用一个向量来表征**，而我仿佛又经常看到一些博客将上面其中的两个并列出现。one-hot编码和word2vec是不一样的？embedding和word2vec是不一样的？

我有很多这样的疑问，直到此刻，我知道我之前一直都在浮于表面，甚至没有开始，模型训练的第一步我都不清楚。因此我想自己好好去看看论文，而非别人总结过的博客，毕竟我不知道谁是权威的。同时为了梳理我的思路，我写下了这篇博客，梳理我疑惑过的问题，总结我看过的论文，写下我自己的理解。

我将我的疑问依次记录了下来：
- NLP领域如何将语言输入模型？
- 文字变成向量的方式是什么,one-hot编码是不是所有途径的第一步？
- 什么是embedding，什么是word2vec，什么是encoder?他们之间的关系是什么？
带着以上的疑问，于是，有了下面的博客。

# 结论
为了让博客更加简单易懂，先说结论：
- 无论任何领域，目前所用到的各种基于transformer架构的模型或者传统的深度模型，**作为输入的都是以数字为载体的向量或者矩阵**；
- **文字变成向量的第一步就是one-hot**，但这只是第一步；如果想让这个向量真正有资格代表这个文字或者短语，需要的就是对这个one-hot编码的进一步处理了；
- Embedding表示的是一种范式，表示通过一定的处理，将一个主体或者对象映射成一个低维的向量，用向量来尽可能地表征这个对象。对象可以用很多种:文字，图像，图网络等等。word2vec是word embedding的一种实现方式，这个名词被提起的主要是因为，被google提出的word2vec应用最广泛，且名字很贴切。word2vec广义上可以理解为"word to vector"，因此总觉得和word embedding这个名词冲突，实际上再被提出的时候，仅代表着一种实现的途径，这种途径下，google提出了两种模型：连续词袋模型(CBOW)和skip-gram模型。encoder和上述的不同在于(这里主要讨论transformer中的encoder)，encoder是embedding之后的模块，如果说embedding是为了用向量表征主体，那么encoder是为了榨取这个表征向量的内在信息。

# 阅读提要
通过上述的问题和结论，我将整个的问题分为了两大部分，embedding和encoder。这一篇博客我想主要从进入模型讲起，encoder部分已经在模型主体里，在此先不赘述，下一篇博客我会具体分享。这一篇我主要针对embedding进行总结。这又回到了我最开始说的那句话，我认为自己在做的是多模态模型的应用。那么embedding就不局限于word embedding了。

我将整个的博客叙述流程设置如下，方便读者以及我自己之后进行回顾或者增删改查。
> 首先我将从word embedding入手，首先介绍words的前处理过程，然后依次介绍one-hot编码、word embedding具体技术及其发展路线，最后我将总结当前论文或者博客中出现的word embedding和我介绍的word embedding之间的关系。
>
> 介绍完word embedding后，我将继续介绍关于item embedding，graph embedding和CV领域关于embedding的内容。

# 博客开始
## Word Embedding
### pre-processing

### one-hot encoding

### NNLM
### word2vec
#### log 
#### CBOW
#### Skip-Gram





## Item Embedding

## Graph Embedding

## Embedding in CV

# Reference
