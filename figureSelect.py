# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# from math import log
from scipy.io import loadmat
from sklearn.cluster import KMeans
import pandas as pd


def getFigureMatrix(data):
    '''
    将所有特征在一维空间上聚类，便于后面计算信息熵
    聚类方式为K-Means
    Args:
        data: 原数据集的特征参数矩阵，row = 实例数，column = 特征数
    Returns:
        figure: 聚类后的特征标签矩阵, row = 特征数，column = 实例数(np.array)
    '''
    figure = np.zeros((len(data[0]), len(data)), dtype = int)
    xTrain = np.zeros((len(data), 1), dtype = float)  # 单列的array，用于训练
    for j in range(len(data[0])):  # 特征数量
        for i in range(len(data)):  # 将实例的特征集中取出
            xTrain[i][0] = data[i][j]
        km = KMeans()
        km.fit(xTrain)
        # print(km.cluster_centers_)  # 聚类中心
        res = km.labels_
        # print(km.labels_)  # 聚类结果
        for k in range(len(km.labels_)):
            figure[j][k] = km.labels_[k]
    print(figure)
    # print(len(figure), len(figure[0]))
    return figure



# def calcEntropy(testData):
#     '''
#     计算信息熵
#     '''
#     for x in testData:
#         ent = 0.0
#         for p in x:
#             ent = -p * np.log(p)
#         entList.append(ent)
#     return entList

emotions_test = loadmat("../dataset/divided/test_data.mat")
emotions_test_data = emotions_test['test_data']  # 训练集的实例特征
emotions_test = loadmat("../dataset/divided/test_target.mat")
emotions_test_target = emotions_test['test_target']  # 训练集的实例标签
# print(emotions_test_data)
# print(emotions_test_target)
# print(type(emotions_test_data))

getFigureMatrix(emotions_test_data)
# entrop_test = calcEntropy(emotions_test_data)

# 绘制一维聚类结果
# plt.scatter([px for px in x], [py for py in x], marker='.')
# plt.scatter([px for px in km.cluster_centers_], [py for py in km.cluster_centers_], marker='.')
# plt.scatter([px for px in figure], [py for py in figure], marker='.')
# plt.show()
