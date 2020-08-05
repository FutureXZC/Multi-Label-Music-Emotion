# 基于多标记分类技术的音乐情感分析研究与实现

## Background

&emsp;&emsp;在日益丰富的音乐应用场景下，实现基于音乐情感的音乐检索、音乐分类和精准推送愈发重要，而一曲音乐在不同片段的情感多变性却使得音乐情感分析任务变得十分复杂。传统的以人工方式对音乐进行情感分析费时费力，简单地使用单标记分类技术也无法准确地分析出音乐所蕴含的复杂而多样的情感，然而，其复杂性与多变性却十分契合多标记分类技术。  
&emsp;&emsp;针对此问题，**基于多标记分类技术，在依据音乐的物理信息提取的带有情感标记的数据集上，先利用信息熵的思想筛选原始音乐特征空间中关键的数值型向量特征，再利用聚类方法进行特征转换，并完成数据的标准化，从而获得一组原始数据空间对应的映射空间，实现音乐元数据的降维操作，最后再以此导入基本分类器进行音乐情感分析。**该方法有着完备的理论支持，且实验结果表明，基本分类器在映射空间上的多标记分类表现整体上优于原始数据空间。

## Usage

GUI.py是控制台界面实现程序。  
main.py是主函数。  
musicLoad.py是导入音乐数据集以及包含所有运算的实现的程序。  
使用时只需要利用IDE运行main.py即可。

## Algorithm

### Feature Selection

&emsp;&emsp;先借助聚类算法对每一个特征进行特征内部的分类，将数值型的特征转换为类别型的特征，然后据此计算每一个特征与标记集的信息增益，并在归一化和正态分布化后设定阈值，若最终某一特征对于标记集的信息增益小于阈值则将其判定为不相关并将其剔除，否则将该特征保留。

### Feature Transform

&emsp;&emsp;实行特征转换的目的是，将元数据集在完成上面介绍的特征选择的基础之上进一步优化特征和标记之间相关性的表现，获得更好的多标记分析结果。算法的核心思想是：首先根据每个实例所属的标记情况，分别为每个标记划分正例集合P<sub>k</sub>和负例集合N<sub>k</sub>，然后对每个标记的正例集合P<sub>k</sub>和负例集合N<sub>k</sub>利用K-Means聚类分别求出m个聚类中心，接着将实例与这2m个聚类中心的欧式距离作为新的特征，完成特征转换，为了适应线性二分类器的工作还需要将标记的取值进行标准化，最后将特征转换后的新特征与标准化后的标记拼接起来，得到最终的输出数据集D<sub>k</sub><sup>*</sup>。

## Practice

### Dataset
- [`emotions`](http://mulan.sourceforge.net/datasets-mlc.html)

### Basic Classifiers

&emsp;&emsp;为了完成了音乐情感分析的多标记分类任务，需要对emotions数据集完成上述特征选择和特征转换，然后使用一些基本分类器实现最终的分类。在这里采用了多标记学习中常见的几种分类器：
- KNN：初始k近邻设置为10，距离度量采用欧式距离，临近点的权重根据距离变化，距离当前对象较近的点权重比距离较远的点大，k的值可由用户在GUI中自定义。
- 朴素贝叶斯分类器：由于样本特征大部分是连续值，故在此采用高斯核。
- 分类器链（CC）：内部分类器选用的也是高斯核的朴素贝叶斯分类器。
- 决策树：相关参数直接使用sklearn中设置的默认值。
- 支持向量机（SVM）：采用较为通用的高斯径向基函数作为核函数，启用概率估计。

### Metrics

- 准确率（Accuracy）
- 汉明损失（Hamming Loss）
- 精度（Precision）
- F1值

## Analysis

&emsp;&emsp;算法处理后的数据在大部分情景下能够比直接使用元数据进行多标记分类有更好的算法表现，但在使用了高斯核的CC和朴素贝叶斯分类器下表现不甚良好，在未来可以考虑在基本分类器的选择上下文章，或改进算法以提升适应性。

## References

- [`Zhang Min-Ling, Zhou Zhi-Hua. A Review on Multi-Label Learning Algorithms[J]. IEEE Transactions on Knowledge & Data Engineering, 2014, 26(8): 1819-1837.`](http://palm.seu.edu.cn/zhangml/)
- [`张振海, 李士宁, 李志刚, 陈昊. 一类基于信息熵的多标签特征选择算法[J]. 计算机研究与发展, 2013, 50(06): 1177-1184.`](https://kns.cnki.net/kns/download.aspx?filename=KFTStRGVo1mVGpWZHFzVElmY1FmdVV0L3QWeEZVUq5GeidTNJJ3SW9CRlJkT0onZa9WRsVFTwRGelVlQx0mYzMkRIZWMUF3NIBDSSJTe6h2MRNTSrcUQ1EFMvVTZUV0a6hUR2IUV2MzaFN1bnpmaxsGOqNWTsJ0N&tablename=CJFD2013&dflag=pdfdown)
- [`Zhang Min-Ling, and Lei Wu. "Lift: Multi-Label Learning with Label-Specific Features." 37 (2015): 107-120.`](http://palm.seu.edu.cn/zhangml/)

## Author

- [`@FutureXZC`](https://github.com/FutureXZC)