# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import sklearn.cluster as skc
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def getFigureMatrix(data):
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
        # km = KMeans().fit(xTrain)  # KMeans聚类
        figure.append(db.labels_.tolist())
    return figure

def getProbability(mat):
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

def calcEntropy(data):
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

def getEntropy(prob):
    '''
    获取含有每个特征/标记的信息熵的列表
    Args:
        figureProb: 含有每个特征/标记对应的概率分布的矩阵, row = 特征数, column = 该行表示的特征/标记对应的类别数
    Returns:
        entList: 保存每一个特征/标记的信息熵的列表, len = 特征/标记数
    '''
    entList = []
    for p in prob:
        entList.append(calcEntropy(p))
    return entList

def getMixFigureAndLabelMatrix(figureMat, labelMat, k):
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

def getConditionalEntropy(figureProb, figureMat, labelMat):
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

def modelPredict(dataTrain, targetTrain, dataTest, targetTest, clf):
    '''
    模型预测，并输出评价指标
    Args: 
        dataTrain: 训练数据
        targetTrain: 训练标记
        dataTest: 测试数据
        targetTest: 测试标记
        clf: 基分类器
    Return:
        ans: 含有所有评价指标数据的列表
    '''
    ans = []
    # 基分类器由clf传入
    for i in range(len(targetTest)):
        clf.fit(dataTrain, targetTrain[:, i])
        yPred = clf.predict(dataTest)
        yTest = targetTest[i].transpose()
        ans.append(metrics.accuracy_score(yTest, yPred))
        # print('Accuracy-L{}:\t\t{:.4f}{}'.format(i+1, metrics.accuracy_score(yTest, yPred), '↗'))
    # 对整体进行预测和评估
    clf.fit(dataTrain, targetTrain)
    yPred = clf.predict(dataTest)
    yTest = targetTest.transpose()
    ans.append(metrics.hamming_loss(yTest, yPred))
    ans.append(metrics.label_ranking_loss(yTest, yPred))
    ans.append(np.mean(metrics.precision_score(yTest, yPred, average = None)))
    ans.append(metrics.f1_score(yTest, yPred, average = 'micro'))
    ans.append(metrics.roc_auc_score(yTest, yPred, average = 'micro'))
    ans.append(metrics.f1_score(yTest, yPred, average = 'macro'))
    ans.append(metrics.roc_auc_score(yTest, yPred, average = 'macro'))
    # print('Hamming Loss: \t\t{:.4f}{}'.format(metrics.hamming_loss(yTest, yPred), '↘'))
    # print('Ranking Loss: \t\t{:.4f}{}'.format(metrics.label_ranking_loss(yTest, yPred), '↘'))
    # print('Avg. Precision: \t{:.4f}{}'.format(np.mean(metrics.precision_score(yTest, yPred, average = None)), '↗'))
    # print('Micro-F1: \t\t{:.4f}{}'.format(metrics.f1_score(yTest, yPred, average = 'micro'), '↗'))
    # print('Micro-AUC: \t\t{:.4f}{}'.format(metrics.roc_auc_score(yTest, yPred, average = 'micro'), '↗'))
    # print('Macro-F1: \t\t{:.4f}{}'.format(metrics.f1_score(yTest, yPred, average = 'macro'), '↗'))
    # print('Macro-AUC: \t\t{:.4f}{}'.format(metrics.roc_auc_score(yTest, yPred, average = 'macro'), '↗'))
    # print()
    return ans

def getEuclideanDistance(data, npData, centers, m):
    '''
    聚类, 计算特征与聚类中心的欧氏距离
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
            # temp.append(sum((x - y)**2 for x, y in zip(data[i], centers[j]))**0.5)
        ed.append(temp)
    return np.array(ed)

def getMapping(dataTrain, targetTrain, dataTest, targetTest):
    '''
    截取正负实例矩阵, 根据原特征聚类, 返回根据算法映射的新特征矩阵
    Args:
        dataTrain: 训练集实例的特征集
        targetTrain: 训练集实例对应的标记集
        dataTest: 测试集实例的特征集
        targetTest: 测试集实例对应的标记集
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
        m = (int)(0.1 * min(len(pDataTrain), len(nDataTrain)))
        # 求聚类中心
        pCenters = KMeans(n_clusters = m).fit(pDataTrain).cluster_centers_
        nCenters = KMeans(n_clusters = m).fit(nDataTrain).cluster_centers_
        # 处理训练集映射
        temp = getEuclideanDistance(dataTrain, pDataTrain, pCenters, m)
        temp = np.hstack((temp, getEuclideanDistance(dataTrain, nDataTrain, nCenters, m)))
        mappingTrain.append(temp.tolist())
        ykTrain.append([1 if pIndexTrain[i] else -1 for i in range(len(targetTrain))])
        # 处理测试集映射
        temp = getEuclideanDistance(dataTest, pDataTest, pCenters, m)
        temp = np.hstack((temp, getEuclideanDistance(dataTest, nDataTest, nCenters, m)))
        mappingTest.append(temp.tolist())
        ykTest.append([1 if pIndexTest[i] else -1 for i in range(len(targetTest))])
    ykTrain = [[row[i] for row in ykTrain] for i in range(len(ykTrain[0]))]
    ykTest = [[row[i] for row in ykTest] for i in range(len(ykTest[0]))]
    return mappingTrain, ykTrain, mappingTest, ykTest

if __name__ == '__main__':

    emotionsTrain = loadmat("../dataset/original/train_data.mat")
    emotionsTrainData = emotionsTrain['train_data']  # 训练集的实例特征
    emotionsTrain = loadmat("../dataset/original/train_target.mat")
    emotionsTrainTarget = emotionsTrain['train_target']  # 训练集的实例特征

    emotionsTest = loadmat("../dataset/original/test_data.mat")
    emotionsTestData = emotionsTest['test_data']  # 测试集的实例特征
    emotionsTest = loadmat("../dataset/original/test_target.mat")
    emotionsTestTarget = emotionsTest['test_target']  # 测试集的实例标记
    
    yTrain = emotionsTrainTarget.transpose()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    svm = SVC(gamma = 'auto', probability = True)
    # nb = GaussianNB()

    figureMatrix = getFigureMatrix(emotionsTrainData)  # 特征类别矩阵
    figurePr = getProbability(figureMatrix)  # 特征的概率分布
    figureEnt = getEntropy(figurePr)  # 特征的信息熵
    # print(figurePr)

    labelMatrix = emotionsTrainTarget.tolist()  # 标记矩阵
    labelPr = getProbability(labelMatrix)  # 标记的概率分布
    labelEnt = getEntropy(labelPr)  # 标记的信息熵
    # print(labelMatrix)

    condEnt = getConditionalEntropy(figurePr, figureMatrix, labelMatrix)  # 条件熵
    # print(condEnt)

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
    # print(ig)
    # print(su)

    igs = []
    for j in range(len(ig[0])):
        temp = 0.0
        for i in range(len(ig)):
            temp += su[i][j]
        igs.append(temp)  # 每一特征与所有标记的信息增益
    # print(igs)

    igsMean = np.mean(igs)  # igs的均值
    igsVar = np.var(igs)  # igs的方差
    igz = [(i - igsMean) / igsVar for i in igs]  # 信息增益正态分布化
    # print(igz)

    figureIndex = []
    igzMean = np.mean([abs(i) for i in igz])  # 阈值
    for i in range(len(igz)):
        if abs(igz[i]) < igzMean:
            figureIndex.append(i)  # 获取通过选择的特征在原特征矩阵的下标
    # 截取特征矩阵，获得仅含有被选中特征的矩阵
    figureSelectedTrain = emotionsTrainData[:, figureIndex]
    figureSelectedTest = emotionsTestData[:, figureIndex]
    yTrain = emotionsTrainTarget.transpose()
    yTest = emotionsTestTarget.transpose()
    # print(len(figureSelectedTrain[0]))  # 67

    # 筛选正负例, 通过聚类算法映射新特征
    mappingTrain, ykTrain, mappingTest, ykTest = getMapping(figureSelectedTrain, yTrain, figureSelectedTest, yTest)

    metricDataOrigin, metricDataSelected = [], []
    # 对每个标记进行预测，并输出评价指标
    metricDataOrigin.append(modelPredict(emotionsTrainData, yTrain, emotionsTestData, emotionsTestTarget, knn))
    metricDataSelected.append(modelPredict(figureSelectedTrain, yTrain, figureSelectedTest, emotionsTestTarget, knn))
    metricDataOrigin.append(modelPredict(emotionsTrainData, yTrain, emotionsTestData, emotionsTestTarget, dt))
    metricDataSelected.append(modelPredict(figureSelectedTrain, yTrain, figureSelectedTest, emotionsTestTarget, dt))
    print("==========================================")
    print('svm:')
    hl,pr, mif1, maf1  = [], [], [], []
    print('Origin:')
    for i in range(len(yTrain[0])):
        svm.fit(np.array(figureSelectedTrain), np.array(yTrain)[:, i])
        y = svm.predict(figureSelectedTest)
        hl.append(metrics.hamming_loss(y, np.array(yTest)[:, i]))
        pr.append(metrics.precision_score(y, np.array(yTest)[:, i]))
        mif1.append(metrics.f1_score(y, np.array(yTest)[:, i], average = 'micro'))
        maf1.append(metrics.f1_score(y, np.array(yTest)[:, i], average = 'macro'))
    print('Hamming Loss: ',hl)
    print('Precision Score: ', pr)
    print('Micro F1: ', mif1)
    print('Macro F1: ', maf1)
    hl,pr, mif1, maf1  = [], [], [], []
    print('LIFT:')
    for i in range(len(mappingTrain)):
        svm.fit(np.array(mappingTrain[i]), np.array(ykTrain)[:, i])
        y = svm.predict(mappingTest[i])
        hl.append(metrics.hamming_loss(y, np.array(ykTest)[:, i]))
        pr.append(metrics.precision_score(y, np.array(ykTest)[:, i]))
        mif1.append(metrics.f1_score(y, np.array(ykTest)[:, i], average = 'micro'))
        maf1.append(metrics.f1_score(y, np.array(ykTest)[:, i], average = 'macro'))
    print('Hamming Loss: ',hl)
    print('Precision Score: ', pr)
    print('Micro F1: ', mif1)
    print('Macro F1: ', maf1)
    print("==========================================")
    # # print('Naive Bayes:')
    # # print('Origin:')
    # # modelPredict(emotionsTrainData, yTrain, emotionsTestData, emotionsTestTarget, nb)
    # # print('Selected:')
    # # modelPredict(figureSelectedTrain, yTrain, figureSelectedTest, emotionsTestTarget, nb)
    # # print("==========================================")
    # metricDataOrigin = np.array(metricDataOrigin)
    # metricDataSelected = np.array(metricDataSelected)

    # 绘图
    figureSize = len(metricDataOrigin)
    plt.figure(figureSize)
    xIndex = np.arange(1, 26, 2)
    xData = ('L1 Accuracy', 'L2 Accuracy', 'L3 Accuracy', 'L4 Accuracy',
             'L5 Accuracy', 'L6 Accuracy', 'Hamming Loss',
             'Label Ranking', 'Precision Score',
             'Micro F1', 'Micro AUC', 'Macro F1', 'Macro AUC')
    title = ('KNN', 'Decision Tree')
    barWidth = 0.60
    for i in range(len(metricDataOrigin)):
        plt.subplot(figureSize, 1, i + 1)
        plt.bar(xIndex, metricDataOrigin[i], width = barWidth,
                alpha = 0.6, color = 'b', label = 'Origin')
        plt.bar(xIndex + barWidth, metricDataSelected[i], width = barWidth,
                alpha = 0.6, color = 'r', label = 'Selected')
        plt.xticks(xIndex + barWidth / 2, xData, rotation = 40)
        plt.title(title[i])
        plt.legend()
    plt.tight_layout()
    plt.show()
