# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
import pandas as pd

def getFigureMatrix(data):
    '''
    将所有特征在一维空间上聚类, 便于后面计算信息熵
    聚类方式为K-Means
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
        km = KMeans()
        km.fit(xTrain)
        # print(km.cluster_centers_)  # 聚类中心
        res = km.labels_
        # print(km.labels_)  # 聚类结果
        # for k in range(len(km.labels_)):
        #     figure[j][k] = km.labels_[k]
        figure.append(km.labels_.tolist())
    # print(figure)
    return figure

def getProbability(mat):
    '''
    求给定特征/标记矩阵对应的概率分布
    Args:
        mat: 特征/标记矩阵, 记录每个特征/标记在实例中的发生情况, row = 特征/标记数, column = 实例数
    Returns:
        prob: 每个特征/标记在给定实例中的概率分布, row = 特征/标记数, column = 该行表示的特征/标记对应的类别数
    '''
    prob = []
    for i in range(len(mat)):
        temp = []
        classDic = {}
        for item in mat[i]:
            if item in classDic:
                classDic[item] += 1
            else: 
                classDic[item] = 1
        for item in classDic:
            temp.append(mat[i].count(item) / len(mat[i]))
        prob.append(temp)
    # print(prob)
    return prob

def calcEntropy(data):
    '''
    计算信息熵
    Args:
        data: 含有某一特征的概率分布情况的列表, len = 类别数
    Return:
        ent: 该特征对应的信息熵, elementType = float
    '''
    ent = 0.0
    for p in data:
        ent = -p * np.log(p)
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
    # print(entList)
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
    # print(mixMat)
    return mixMat


if __name__ == '__main__':
    emotions_test = loadmat("../dataset/original/test_data.mat")
    emotions_test_data = emotions_test['test_data']  # 训练集的实例特征
    emotions_test = loadmat("../dataset/original/test_target.mat")
    emotions_test_target = emotions_test['test_target']  # 训练集的实例标记
    # print(emotions_test_target)
    # print(emotions_test_data)

    figureMatrix = getFigureMatrix(emotions_test_data)  # 特征类别矩阵
    figureProb = getProbability(figureMatrix)  # 特征的概率分布
    figureEnt = getEntropy(figureProb)  # 特征的信息熵
    # print(len(figureEnt))

    labelMatrix = emotions_test_target.tolist()  # 标记矩阵
    labelProb = getProbability(labelMatrix)  # 标记的概率分布
    labelEnt = getEntropy(labelProb)  # 标记的信息熵
    # print(labelEnt)

    combinEnt = []
    for i in range(len(labelMatrix)):
        mixMatrix = getMixFigureAndLabelMatrix(figureMatrix, labelMatrix, i)  # 联合矩阵
        mixProb = getProbability(mixMatrix)  # 联合概率分布
        combinEnt.append(getEntropy(mixProb))  # 联合熵
    # print(combinEnt)

    ig = []
    su = []
    for i in range(len(combinEnt)):
        tempInfo = []
        tempSu = []
        for j in range(len(figureEnt)):
            info = figureEnt[j] + labelEnt[i] - combinEnt[i][j]
            s = 2 * (info / (figureEnt[j] + labelEnt[i]))
            tempInfo.append(info)
            tempSu.append(s)
        ig.append(tempInfo)  # 信息增益
        su.append(tempSu)  # 归一化

    igs = []
    for j in range(len(ig[0])):
        temp = 0.0
        for i in range(len(ig)):
            temp += ig[i][j]
        igs.append(temp)
    print(igs)
    print(len(igs))





    # 绘制一维聚类结果
    # plt.scatter([px for px in x], [py for py in x], marker='.')
    # plt.scatter([px for px in km.cluster_centers_], [py for py in km.cluster_centers_], marker='.')
    # plt.scatter([px for px in figure], [py for py in figure], marker='.')
    # plt.show()
