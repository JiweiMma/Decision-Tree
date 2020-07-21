from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
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
    dataSet=[[2,1,0,0,'no'],[1,0,0,1,'no'],[0,1,0,1,'yes'],[0,1,0,0,'yes'],[0,0,0,0,'no'],
             [0,0,0,0,'no'],[2,0,0,1,'no'],[2,1,1,1,'yes'],[1,0,1,2,'yes'],[1,0,1,2,'yes'],
             [1,0,1,2,'yes'],[0,0,1,1,'yes'],[2,1,0,1,'yes'],[2,1,0,0,'yes'],[2,0,0,2,'no']]
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
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                    #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree

#获取决策树叶子结点的数目
def getNumLeafs(myTree):
    # 初始化叶子
    numLeafs = 0
    #firstStr=myTree.keys()[0]
    # python3中myTree.keys()返回的是dict_keys,不是list
    #所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #如果是字典，继续递归调用函数
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs


#函数说明:获取决策树的层数
def getTreeDepth(myTree):
    # 初始化决策树深度
    maxDepth = 0
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        # 更新层数
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 使用文本注解绘制树节点
#nodeTxt - 结点名   centerPt - 文本位置   parentPt - 标注的箭头位置   nodeType - 结点格式
# centerPt 箭头指向坐标， parentPt 箭头终点坐标
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 设置中文字体
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    # 绘制结点
    # axes fraction 左下角部分    xytext文本位置   xy:标注位置   nodeTxt节点内容
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


#标注有向边属性值
#cntrPt--子位置坐标
#txtString - 标注的内容
def plotMidText(cntrPt, parentPt, txtString):
    # 计算标注位置，在两个点的中间位置，用数学公式计算
    #分别计算xy轴的位置
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

#绘制决策树
#myTree - 决策树(字典类型) parentPt - 标注的内容  nodeTxt - 结点名
def plotTree(myTree, parentPt, nodeTxt):
    # 定义文本框和箭头格式
    # 定义文本框的类型，为锯齿型，边框线的粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    #获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置，计算子节点坐标
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    #标注子节点有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制子结点
    secondDict = myTree[firstStr]
    #y偏移，改变深度，纵坐标进行减一层
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__=='dict':
            #不是叶结点，递归调用继续绘制
            plotTree(secondDict[key],cntrPt,str(key))
        else:                    #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
            #循环结束，递归回溯
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


#创建绘制面板
def createPlot(inTree):
    # 创建新的绘画窗口，名为figure1,背景色为白色
    fig = plt.figure(1, facecolor='white')
    #清空绘图区
    fig.clf()
    # 定义横纵坐标轴,注意不要设置xticks和yticks的值
    axprops = dict(xticks=[], yticks=[])
    # createPlot.ax1为全局变量，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    # frameon表示是否绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    #获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xOff = -0.5/plotTree.totalW;
    #赋值给绘制节点的初始值为1.0
    plotTree.yOff = 1.0;
    # 绘制决策树， 开始父节点的位置
    plotTree(inTree, (0.5,1.0), '')
    # 显示绘图结果
    plt.show()

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
