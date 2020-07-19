import operator
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

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                   #特征数量
    baseEntropy = calcShannonEnt(dataSet)               #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                        #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                              #创建set集合{},元素不可重复
        newEntropy = 0.0                                        #经验条件熵
        for value in uniqueVals:                                #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)        #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))        #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                     #信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))         #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                           #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature                                          #返回信息增益最大的特征的索引值

#多数表决的方法决定叶子节点的分类
#统计classList中出现此处最多的元素(类标签)
def majorityCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    #根据字典的值降序排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet, labels):
    # 取分类标签(本题为是否放贷:yes or no)
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分，函数的第一个停止条件
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类标签，函数第二个停止条件
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}
    # 删除已经使用特征标签
    del(labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVals = set(featValues)
    # 遍历特征，创建决策树。
    for value in uniqueVals:
        #复制了类标签，当函数的参数是列表类型时，参数引用方式传递
        #为了不改变原始的列表内容，使用subLabels代替原始列表。
        subLabels=labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)

