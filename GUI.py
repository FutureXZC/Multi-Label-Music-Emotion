# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.messagebox

class Console:
    '''
    控制台窗体
    '''
    def __init__(self, music):
        '''
        初始化窗体
        '''
        self.lastR = 0.1
        self.recentR = 0.1
        self.m = music
        self.top = tk.Tk()
        self.top.title('多标记分类控制台')
        self.title = tk.Label(self.top, text='基于多标记分类的音乐情感分析')
        self.selectedLabel = tk.Label(self.top, text='特征选择前后的算法表现对比')
        self.transformenLabel = tk.Label(self.top, text='特征转换前后的算法表现对比')
        self.rFrame = tk.Frame(self.top)
        self.lFrame = tk.Frame(self.top)
        # 输入框
        self.rStr = tk.StringVar()
        self.rStr.set('0.1')
        self.rLabel = tk.Label(self.rFrame, text='r = ')
        self.rEntry = tk.Entry(self.rFrame, textvariable=self.rStr, width=5)
        # 按钮事件
        self.cc1Btn = tk.Button(self.lFrame, text='CC', command=self.cc1BtnClick)
        self.knnBtn = tk.Button(self.lFrame, text='KNN', command=self.knnBtnClick)
        self.dtBtn = tk.Button(self.lFrame, text='决策树', command=self.dtBtnClick)
        self.cc2Btn = tk.Button(self.rFrame, text='CC', command=self.cc2BtnClick)
        self.svmBtn = tk.Button(self.rFrame, text='SVM', command=self.svmBtnClick)
        self.nbBtn = tk.Button(self.rFrame, text='朴素贝叶斯', command=self.nbBtnClick)
        # 左侧面板布局
        self.title.grid(row=0, column=2)
        self.selectedLabel.grid(row=1, column=0)
        self.transformenLabel.grid(row=1, column=3)
        self.lFrame.grid(row=2, column=0)
        self.rFrame.grid(row=2, column=3)
        self.cc1Btn.grid(row=0, column=0)
        self.knnBtn.grid(row=1, column=0)
        self.dtBtn.grid(row=2, column=0)
        # 右侧面板布局
        self.cc2Btn.grid(row=0, column=4)
        self.svmBtn.grid(row=1, column=4)
        self.nbBtn.grid(row=2, column=4)
        # 输入框布局
        self.rLabel.grid(row=0, column=0)
        self.rEntry.grid(row=0, column=1)
        # 运行窗口
        self.top.mainloop()

    def drawFigure1(self, i):
        '''
        绘制特征选择后的对应基本分类器的算法表现的统计图
        Args:
            i: 选择绘制何种基本分类器下的算法表现(0: KNN, 1: Classifier Chain, 2: Decision Tree)
        Output:
            对应基本分类器算法表现的统计图
        '''
        xIndex = np.arange(1, 20, 2)
        xData = ('L1 Accuracy', 'L2 Accuracy', 'L3 Accuracy', 'L4 Accuracy',
                'L5 Accuracy', 'L6 Accuracy', 'Hamming Loss',
                'Precision', 'Micro F1',  'Macro F1')
        title = ('KNN', 'Classifier Chain', 'Decision Tree')
        barWidth = 0.85
        plt.bar(xIndex, self.m.metricDataOrigin[i], width = barWidth,
                alpha = 0.6, color = 'b', label = 'Origin')
        plt.bar(xIndex + barWidth, self.m.metricDataSelected[i], width = barWidth,
                alpha = 0.6, color = 'r', label = 'Selected')
        plt.xticks(xIndex + barWidth / 2, xData, rotation = 40)
        plt.title(title[i])
        plt.legend()
        plt.ylim(0, 1.0)
        for x, y1, y2 in zip(xIndex, self.m.metricDataOrigin[i], self.m.metricDataSelected[i]):
                plt.text(x, y1, '%.4f' % y1, ha = 'center', va = 'bottom')
                plt.text(x + barWidth, y2, '%.4f' % y2, ha = 'center', va = 'bottom')
        plt.tight_layout()
        plt.show()

    def drawFigure2(self, i):
        '''
        绘制特征转换后的对应基本分类器的算法表现的统计图
        Args:
            i: 选择绘制何种基本分类器下的算法表现(0: SVM, 1: Naive Bayes, 2: Classifier Chain)
        Output:
            对应基本分类器算法表现的统计图
        '''
        self.recentR = float(self.rEntry.get())
        if self.recentR > 0.6 or self.recentR < 0.1:
            tk.messagebox.askquestion(title='参数范围错误', message='请输入0.1-0.6之间的正确数字！')
            return
        if self.lastR != self.recentR:
            # 若近期修改过r值，则需重新计算评价指标
            self.lastR = self.recentR
            print(self.lastR)
            self.m.figureMapping(self.lastR)
        # print(self.recentR)
        figureSize = len(self.m.BRMetricOrigin[0])
        xIndex = np.arange(1, 8, 2)
        xData = ('Hamming Loss', 'Precision Score', 'Micro F1', 'Macro F1')
        title = ('SVM', 'Naive Bayes', 'Classifier Chain')
        label = ('L1', 'L2', 'L3', 'L4', 'L5','L6')
        barWidth = 0.60
        for j in range(figureSize):
            plt.subplot(figureSize / 2, 2, j + 1)
            plt.bar(xIndex, self.m.BRMetricOrigin[i][j], width = barWidth,
                    alpha = 0.6, color = 'b', label = 'Origin')
            plt.bar(xIndex + barWidth, self.m.BRMetricLift[i][j], width = barWidth,
                    alpha = 0.6, color = 'r', label = 'Processed')
            plt.xticks(xIndex + barWidth / 2, xData)
            plt.title(title[i] + '-' + label[j])
            plt.legend()
            plt.ylim(0, 1.0)
            for x, y1, y2 in zip(xIndex, self.m.BRMetricOrigin[i][j], self.m.BRMetricLift[i][j]):
                plt.text(x, y1, '%.4f' % y1, ha = 'center', va = 'bottom')
                plt.text(x + barWidth, y2, '%.4f' % y2, ha = 'center', va = 'bottom')
        plt.tight_layout()
        plt.show()

    def cc1BtnClick(self):
        '''
        左侧CC按钮事件
        '''
        self.drawFigure1(1)
    
    def knnBtnClick(self):
        '''
        左侧KNN按钮事件
        '''
        self.drawFigure1(0)
    
    def dtBtnClick(self):
        '''
        左侧决策树按钮事件
        '''
        self.drawFigure1(2)

    def svmBtnClick(self):
        '''
        右侧SVM按钮事件
        '''
        self.drawFigure2(0)

    def cc2BtnClick(self):
        '''
        右侧CC按钮事件
        '''
        self.drawFigure2(2)

    def nbBtnClick(self):
        '''
        右侧朴素贝叶斯按钮事件
        '''
        self.drawFigure2(1)
