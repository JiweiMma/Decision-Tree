import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#定义文本框和箭头格式
#定义文本框的类型，为锯齿型，边框线的粗细
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
#定义箭头方向
arrow_args=dict(arrowstyle="<-")

#使用文本注解绘制树节点
#nodeTxt - 结点名   centerPt - 文本位置   parentPt - 标注的箭头位置   nodeType - 结点格式
# centerPt 箭头指向坐标， parentPt 箭头终点坐标
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 设置中文字体
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 绘制结点
    #axes fraction 左下角部分    xytext文本位置   xy:标注位置   nodeTxt节点内容
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args,FontProperties=font)

def createPlot():
    #创建新的绘画窗口，名为figure1,背景色为白色
    fig=plt.figure(1,facecolor='white')
    #清空绘图区
    fig.clf()
    # createPlot.ax1为全局变量，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    # frameon表示是否绘制坐标轴矩形
    createPlot.ax1=plt.subplot(111,frameon=True)
    #绘制节点
    plotNode('决策结点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    #显示绘图结果
    plt.show()

if __name__ == '__main__':
    myTree = createPlot()
    print(myTree)