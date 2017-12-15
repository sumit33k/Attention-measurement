原始论文：UMD-TTIC-UW at SemEval-2016 Task 1: Attention-Based
Multi-Perspective Convolutional Neural Networks for Textual Similarity
Measurement (http://aclweb.org/anthology/S/S16/S16-1170.pdf)
（1）Word embadding 使用的是 "paragram-phrase-XXL" 字典（大小50,000多）， 字典中没有查到的暂时先不考虑（去大写等，也可以随机或者用平均代替）；
（2）回归问题分类化，把0-5之间的浮点型score变成整形（四舍五入），用向量表示，例如[0,0,0,1,0,0]表示score=3；
（3）使用12-15年的数据做为训练数据，16年的数据作为测试数据；
（4）本code基本实了论文cnn的框架，可以跑通，但是存在一些问题，例如：全局卷积，余弦距离如果两个都是0向量的情况等。


