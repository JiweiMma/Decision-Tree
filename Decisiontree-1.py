from math import log

#计算给定数据集的熵
def calcShannonEnt(dataSet):
    #返回数据集的行数
    numEntries=len(dataSet)
    #保存每个标签（Label）出现次数的字典
    labelCounts={}
    #对每一组的特征向量进行统计
    for featVec in dataSet:
        #提取标签的信息
        currentLabel=featVec[-1]
        #查看是否放入字典中，没有就添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    #香农熵
    shannonEnt=0.0
    #计算香农熵
    for key in labelCounts:
        #选择标签的概率
        prob=float(labelCounts[key])/numEntries
        #利用公式计算熵
        shannonEnt -=prob*log(prob,2)
    return shannonEnt

def createDataSet():#数据集
    dataSet=[[0,0,0,0,'no'],[0,0,0,1,'no'],[0,1,0,1,'yes'],[0,1,1,0,'yes'],[0,0,0,0,'no'],
             [1,0,0,0,'no'],[1,0,0,1,'no'],[1,1,1,1,'yes'],[1,0,1,2,'yes'],[1,0,1,2,'yes'],
             [2,0,1,2,'yes'],[2,0,1,1,'yes'],[2,1,0,1,'yes'],[2,1,0,2,'yes'],[2,0,0,0,'no']]
    labels=['年龄','有工作','有自己的房子','信贷情况']#数据的分类属性
    return dataSet,labels
    #返回数据集和分类属性


#按照给定的特征划分数据集
def splitDataSet(dataSet, axis, value):
    # dataSet - 待划分的数据集
    # axis - 划分数据集的特征
    # value - 需要返回的特征的值
    retDataSet = []                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]             #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet                           #返回划分后的数据集

if __name__ == '__main__':
    myDat, labels = createDataSet()
    DataSet=splitDataSet(myDat,0,1)
    print(DataSet)
    DataSet1=splitDataSet(myDat,0,2)
    print(DataSet1)