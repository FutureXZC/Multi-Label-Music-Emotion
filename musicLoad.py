# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import sklearn.cluster as skc
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from scipy.sparse import csc_matrix

class Music:
    '''
    
    '''
    def __init__(self):
        '''
        导入数据模型，初始化基本分类器
        '''
        emotionsTrain = loadmat("../dataset/original/train_data.mat")
        self.emotionsTrainData = emotionsTrain['train_data']  # 训练集的实例特征
        emotionsTrain = loadmat("../dataset/original/train_target.mat")
        self.emotionsTrainTarget = emotionsTrain['train_target']  # 训练集的实例标记
        emotionsTest = loadmat("../dataset/original/test_data.mat")
        self.emotionsTestData = emotionsTest['test_data']  # 测试集的实例特征
        emotionsTest = loadmat("../dataset/original/test_target.mat")
        self.emotionsTestTarget = emotionsTest['test_target']  # 测试集的实例标记
        self.figureIndex = []
        self.metricDataOrigin, self.metricDataSelected = [], []
        self.BRMetricOrigin, self.BRMetricLift = [], []
        self.figureSelectedTrain, self.figureSelectedTest = [], []
        self.yTrain, self.yTest = [], []
        self.knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance', p = 2)
        self.cc = ClassifierChain(GaussianNB())
        self.dt = DecisionTreeClassifier()
        self.svm = SVC(kernel='rbf', gamma = 'auto', probability = True)
        self.nb = GaussianNB()

    def getFigureMatrix(self, data):
        '''
        将所有特征在一维空间上聚类, 便于后面计算信息熵
        聚类方式为DBSCAN密度聚类
        Args:
            data: 原数据集的特征参数矩阵, row = 实例数, column = 特征数
        Returns:
            figure: 聚类后的特征标记矩阵, 记录每个特征在实例中的发生情况, row = 特征数, column = 实例数
        '''
        figure = []
        xTrain = np.zeros((len(data), 1), dtype = float)  # 单列的array, 用于训练
        for j in range(len(data[0])):  # 特征数量
            for i in range(len(data)):  # 将实例的特征集中取出
                xTrain[i][0] = data[i][j]
            db = skc.DBSCAN().fit(xTrain)  # 密度聚类
            figure.append(db.labels_.tolist())
        return figure

    def getProbability(self, mat):
        '''
        求给定特征/标记矩阵对应的概率分布
        Args:
            mat: 特征/标记矩阵, 记录每个特征/标记在实例中的发生情况, row = 特征/标记数, column = 实例数
        Returns:
            pr: 每个特征/标记在给定实例中的概率分布, row = 特征/标记数, column = 该行表示的特征/标记对应的类别数
        '''
        pr = []
        for i in range(len(mat)):
            temp = []
            classDic = {}
            for item in mat[i]:
                if item in classDic:
                    classDic[item] += 1
                else: 
                    classDic[item] = 1
            for item in classDic:
                temp.append(classDic[item] / len(mat[i]))
            pr.append(temp)
        return pr

    def calcEntropy(self, data):
        '''
        计算信息熵
        Args:
            data: 含有某一特征的概率分布情况的列表, len = 类别数
        Returns:
            ent: 该特征对应的信息熵, elementType = float
        '''
        ent = 0.0
        for p in data:
            ent = ent - p * np.log(p)
        return ent

    def getEntropy(self, prob):
        '''
        获取含有每个特征/标记的信息熵的列表
        Args:
            figureProb: 含有每个特征/标记对应的概率分布的矩阵, row = 特征数, column = 该行表示的特征/标记对应的类别数
        Returns:
            entList: 保存每一个特征/标记的信息熵的列表, len = 特征/标记数
        '''
        entList = []
        for p in prob:
            entList.append(self.calcEntropy(p))
        return entList

    def getMixFigureAndLabelMatrix(self, figureMat, labelMat, k):
        '''
        获取用于第k个标记之于所有特征的联合分布矩阵
        特征类别F和标记编号L都为int型数，混合时只需要记录 F * 10 + L 即可
        矩阵内每个元素包含特征类别和所属标记的情况, 如类别1对应标记1, 该元素记为'11'
        特别的: 特征类别0对应的任意标记的记法与原标记一致，如类别0对应标记1记为'1', 该记法简化计算的同时不会丢失信息
        Args: 
            figureMat: 特征标记矩阵, 记录每个特征在实例中的发生情况, row = 特征数, column = 实例数
            labelMat: 标记矩阵, 记录每个特征对应的标记, row = 特征数, column = 实例数
            k: 第k个标记
        Returns:
            mixMat: 同时含有特征信息和第k个标记信息的联合分布矩阵, row = 特征数, column = 实例数
        '''
        mixMat = []
        for fig in figureMat:
            temp = []
            for j in range(len(fig)):
                temp.append(fig[j] * 10 + labelMat[k][j])
            mixMat.append(temp)
        return mixMat

    def getConditionalEntropy(self, figureProb, figureMat, labelMat):
        '''
        计算条件熵
        Args:
            figureProb: 含有每个特征对应的概率分布的矩阵, row = 特征数, column = 该行表示的特征对应的类别数
            figureMat: 特征标记矩阵, 记录每个特征在实例中的发生情况, row = 特征数, column = 实例数
            labelMat: 标记矩阵, 记录每个特征对应的标记, row = 特征数, column = 实例数
        Returns:
            condEnt: 条件熵矩阵, 记录已知某特征的条件下求某标记的不确定度, row = 标记数, column = 特征数
        '''
        condEnt = []
        for k in range(len(labelMat)):  # 标记数
            temp = []
            for i in range(len(figureMat)):  # 特征数
                classDic = {}
                for j in range(len(figureMat[i])):  
                    if figureMat[i][j] not in classDic:
                        classDic[figureMat[i][j]] = 0
                    # 仅记录标记为1的实例数即可，标记为0可由len减之
                    if labelMat[k][j]:
                        classDic[figureMat[i][j]] += 1
                jointPr = 0.0
                j = 0
                for item in classDic:
                    p1 = classDic[item] / figureMat[i].count(item)  # label = 1的条件概率
                    p0 = 1 - p1  # label = 0的条件概率
                    jp1 = p1 * figureProb[i][j]  # label = 1的联合概率
                    jp0 = p0 * figureProb[i][j]  # label = 0的联合概率
                    # 需判断某一条件概率为0的情况，否则log(0)为-inf，会使得jointPr为nan
                    if p1 == 0:
                        jointPr = jointPr  - jp0 * np.log(p0)
                    elif p0 == 0:
                        jointPr = jointPr  - jp1 * np.log(p1)
                    else:
                        jointPr = jointPr - jp1 * np.log(p1) - jp0 * np.log(p0)
                    j += 1
                temp.append(jointPr)  # 条件熵
            condEnt.append(temp)
        return condEnt

    def modelPredict(self, dataTrain, targetTrain, dataTest, targetTest, clf):
        '''
        模型预测，并输出评价指标
        Args: 
            dataTrain: 训练数据
            targetTrain: 训练标记
            dataTest: 测试数据
            targetTest: 测试标记
            clf: 基本分类器
        Return:
            ans: 含有所有评价指标数据的列表
        '''
        ans = []
        # 基本分类器由clf传入
        for i in range(len(targetTest)):
            clf.fit(dataTrain, targetTrain[:, i])
            yPred = clf.predict(dataTest)
            yTest = targetTest[i].transpose()
            if type(yPred) == csc_matrix:
                # 若判别结果为稀疏矩阵，将其转为稠密矩阵
                yPred = yPred.todense()
            ans.append(metrics.accuracy_score(yTest, yPred))
        # 对整体进行预测和评估
        clf.fit(dataTrain, targetTrain)
        yPred = clf.predict(dataTest)
        yTest = targetTest.transpose()
        if type(yPred) == csc_matrix:
            yPred = yPred.todense()
        ans.append(metrics.hamming_loss(yTest, yPred))
        ans.append(np.mean(metrics.precision_score(yTest, yPred, average = None)))
        ans.append(metrics.f1_score(yTest, yPred, average = 'micro'))
        ans.append(metrics.f1_score(yTest, yPred, average = 'macro'))
        return ans

    def getEuclideanDistance(self, data, npData, centers, m):
        '''
        聚类, 计算样本与聚类中心的欧氏距离
        Args:
            data: 原数据
            npData: 正例/负例数据
            centers: 聚类中心数组
            m: 聚类中心个数
        Return:
            ed: 包含欧氏距离数据的矩阵
        '''
        ed = []
        for i in range(len(data)):
            temp = []
            for j in range(m):
                temp.append(np.linalg.norm(data[i] - centers[j]))
            ed.append(temp)
        return np.array(ed)

    def getMapping(self, dataTrain, targetTrain, dataTest, targetTest, r):
        '''
        截取正负实例矩阵, 根据原特征聚类, 返回根据算法映射的新特征矩阵
        Args:
            dataTrain: 训练集实例的特征集
            targetTrain: 训练集实例对应的标记集
            dataTest: 测试集实例的特征集
            targetTest: 测试集实例对应的标记集
            r: 聚类比例系数
        Return:
            mappingTrain: 新的训练集特征矩阵, 数据为实例到各个聚类中心的欧式距离
            ykTrain: 新的训练集标记记法, 正例标记为1，负类标记为-1
            mappingTest: 新的测试集特征矩阵, 数据为实例到各个聚类中心的欧式距离
            ykTest: 新的测试集标记记法, 正例标记为1，负类标记为-1
        '''
        mappingTrain, ykTrain, mappingTest, ykTest = [], [], [], []
        for i in range(len(targetTrain[0])):
            pIndexTrain, pIndexTest = targetTrain[:, i] == 1, targetTest[:, i] == 1
            nIndexTrain, nIndexTest = targetTrain[:, i] == 0, targetTest[:, i] == 0
            pDataTrain, pDataTest = dataTrain[pIndexTrain, :], dataTest[pIndexTest, :]  # 正例
            nDataTrain, nDataTest = dataTrain[nIndexTrain, :], dataTest[nIndexTest, :]  # 负例
            # 由训练集获取m值
            m = (int)(r * min(len(pDataTrain), len(nDataTrain)))
            # 求聚类中心
            pCenters = KMeans(n_clusters = m).fit(pDataTrain).cluster_centers_
            nCenters = KMeans(n_clusters = m).fit(nDataTrain).cluster_centers_
            # 处理训练集映射
            temp = self.getEuclideanDistance(dataTrain, pDataTrain, pCenters, m)
            temp = np.hstack((temp, self.getEuclideanDistance(dataTrain, nDataTrain, nCenters, m)))
            mappingTrain.append(temp.tolist())
            ykTrain.append([1 if pIndexTrain[i] else -1 for i in range(len(targetTrain))])
            # 处理测试集映射
            temp = self.getEuclideanDistance(dataTest, pDataTest, pCenters, m)
            temp = np.hstack((temp, self.getEuclideanDistance(dataTest, nDataTest, nCenters, m)))
            mappingTest.append(temp.tolist())
            ykTest.append([1 if pIndexTest[i] else -1 for i in range(len(targetTest))])  # 适应svm分类器
        ykTrain = [[row[i] for row in ykTrain] for i in range(len(ykTrain[0]))]
        ykTest = [[row[i] for row in ykTest] for i in range(len(ykTest[0]))]
        return mappingTrain, ykTrain, mappingTest, ykTest

    def BRModelPredict(self, dataTrain, targetTrain, dataTest, targetTest, clf, m):
        '''
        用二分类器对模型进行预测，并输出评价指标
        Args:
            dataTrain: 训练集实例的特征集
            targetTrain: 训练集实例对应的标记集
            dataTest: 测试集实例的特征集
            targetTest: 测试集实例对应的标记集
            clf: 基二分类器
            m: 模式参数，0代表特征转换后的模式，1代表原数据模式
        Return:
            BMetric: 含有每个标记单独的评价指标的列表
        '''
        BRMetric = []
        count = len(targetTrain[0]) if m == 0 else len(dataTrain)
        for i in range(count):
            if m == 0:
                clf.fit(np.array(dataTrain), np.array(targetTrain)[:, i])
                yPred = clf.predict(dataTest)
            else:
                clf.fit(np.array(dataTrain[i]), np.array(targetTrain)[:, i])
                yPred = clf.predict(dataTest[i])
            if type(yPred) == csc_matrix:
                # 若判别结果为稀疏矩阵，将其转为稠密矩阵
                yPred = yPred.todense()
            temp = []
            temp.append(metrics.hamming_loss(yPred, np.array(targetTest)[:, i]))
            temp.append(metrics.precision_score(yPred, np.array(targetTest)[:, i]))
            temp.append(metrics.f1_score(yPred, np.array(targetTest)[:, i], average = 'micro'))
            temp.append(metrics.f1_score(yPred, np.array(targetTest)[:, i], average = 'macro'))
            BRMetric.append(temp)
        return BRMetric

    def figureSelect(self):
        '''
        完成特征选择的全部流程，并保存特征选择前后相关基本分类器的评价指标
        '''
        figureMatrix = self.getFigureMatrix(self.emotionsTrainData)  # 特征类别矩阵
        figurePr = self.getProbability(figureMatrix)  # 特征的概率分布
        figureEnt = self.getEntropy(figurePr)  # 特征的信息熵
        labelMatrix = self.emotionsTrainTarget.tolist()  # 标记矩阵
        labelPr = self.getProbability(labelMatrix)  # 标记的概率分布
        labelEnt = self.getEntropy(labelPr)  # 标记的信息熵
        condEnt = self.getConditionalEntropy(figurePr, figureMatrix, labelMatrix)  # 条件熵
        ig, su = [], []
        for i in range(len(condEnt)):
            tempInfo, tempSu = [], []
            for j in range(len(condEnt[i])):
                info = labelEnt[i] - condEnt[i][j]
                s = 2 * (info / (figureEnt[j] + labelEnt[i]))
                tempInfo.append(info)
                tempSu.append(s)
            ig.append(tempInfo)  # 每一特征与每一标记的信息增益
            su.append(tempSu)  # 归一化ig
        igs = []
        for j in range(len(ig[0])):
            temp = 0.0
            for i in range(len(ig)):
                temp += su[i][j]
            igs.append(temp)  # 每一特征与所有标记的信息增益
        igsMean = np.mean(igs)  # igs的均值
        igsVar = np.var(igs)  # igs的方差
        igz = [(i - igsMean) / igsVar for i in igs]  # 信息增益正态分布化
        igzMean = np.mean([abs(i) for i in igz])  # 阈值
        for i in range(len(igz)):
            if abs(igz[i]) < igzMean:
                # 获取通过选择的特征在原特征矩阵的下标
                self.figureIndex.append(i)
        # 截取特征矩阵，获得仅含有被选中特征的矩阵
        self.figureSelectedTrain = self.emotionsTrainData[:, self.figureIndex]
        self.figureSelectedTest = self.emotionsTestData[:, self.figureIndex]
        self.yTrain = self.emotionsTrainTarget.transpose()
        self.yTest = self.emotionsTestTarget.transpose()
        # 对数据集进行预测，并输出评价指标，对标记选择前后的多标记表现进行对比
        # KNN
        self.metricDataOrigin.append(self.modelPredict(self.emotionsTrainData, self.yTrain, self.emotionsTestData, self.emotionsTestTarget, self.knn))
        self.metricDataSelected.append(self.modelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.emotionsTestTarget, self.knn))
        # Classifier Chain
        self.metricDataOrigin.append(self.modelPredict(self.emotionsTrainData, self.yTrain, self.emotionsTestData, self.emotionsTestTarget, self.cc))
        self.metricDataSelected.append(self.modelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.emotionsTestTarget, self.cc))
        # Decision Tree
        self.metricDataOrigin.append(self.modelPredict(self.emotionsTrainData, self.yTrain, self.emotionsTestData, self.emotionsTestTarget, self.dt))
        self.metricDataSelected.append(self.modelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.emotionsTestTarget, self.dt))
    
    def figureMapping(self, r = 0.1):
        '''
        根据r值聚类，完成特征转换，并保存相关评价指标
        Args: 
            r: 特征转换时的聚类算法中的参数(聚类中心所占总特征数的比例)
        '''
        # 筛选正负例, 通过聚类算法映射新特征
        mappingTrain, ykTrain, mappingTest, ykTest = self.getMapping(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.yTest, r)
        # 对每个标记进行预测，并输出评价指标，对特征处理前后的多标记表现进行对比
        self.BRMetricOrigin, self.BRMetricLift = [], []
        # SVM
        self.BRMetricOrigin.append(self.BRModelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.yTest, self.svm, 0))
        self.BRMetricLift.append(self.BRModelPredict(mappingTrain, ykTrain, mappingTest, ykTest, self.svm, 1))
        # Naive Bayes
        self.BRMetricOrigin.append(self.BRModelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.yTest, self.nb, 0))
        self.BRMetricLift.append(self.BRModelPredict(mappingTrain, ykTrain, mappingTest, ykTest, self.nb, 1))
        # Classifier Chain
        self.BRMetricOrigin.append(self.BRModelPredict(self.figureSelectedTrain, self.yTrain, self.figureSelectedTest, self.yTest, self.cc, 0))
        self.BRMetricLift.append(self.BRModelPredict(mappingTrain, ykTrain, mappingTest, ykTest, self.cc, 1))
