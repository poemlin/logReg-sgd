from numpy import *

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 用于最后测试时对测试函数进行分类的函数
def classifyVec(inx,weights):
    pro = sigmoid(sum(inx*weights))
    if pro>=0.5:
        return 1.0
    else:
        return 0.0

# 随机梯度下降算法
def SgdAscent(dataSet,dataLabels,numIlter = 150):
    m,n = shape(dataSet)
    # weights是全局的
    weights = ones(n)   # 参数先全置1
    for j in range(numIlter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.0001  # 每一次的步长是与轮次相关的
            # 每次随机选取一个样本更新参数
            randIndex = int(random.uniform(0,len(dataIndex)))
            # 计算该参数下的sigmoid概率，后面两个都是行向量，故sum
            h = sigmoid(sum(dataSet[randIndex]*weights))
            # groundtruth的类别即其概率
            error = dataLabels[randIndex] - h
            weights = weights + alpha * error * dataSet[randIndex]
            # 删除已经用于更新参数的样本
            del(dataIndex[randIndex])
    return weights

# 训练数据导入
def trainDataInput(trainData):
    frtrain = open(trainData)
    trainSet = []
    trainLabels = []
    for line in frtrain.readlines():
        line = line.strip().split('\t')
        thisline = []
        for i in range(21):
            thisline.append(float(line[i]))
        trainSet.append(thisline)
        trainLabels.append(float(line[-1]))
    return trainSet,trainLabels

# 训练过程
def training(trainSet,trainLabels):
    trainWeights = SgdAscent(array(trainSet),trainLabels,1000)
    return trainWeights

# 测试过程
def testing(trainWeights,testData):
    frtest = open(testData)
    # 分错的数目
    errorCount = 0.0
    # 测试样本数目
    numTest = 0.0
    for line in frtest.readlines():
        numTest +=1.0
        line = line.strip().split('\t')
        thisline = []
        for i in range(21):
            thisline.append(float(line[i]))
        if int(classifyVec(thisline,trainWeights))!=int(line[21]):
            # print(int(classifyVec(thisline,trainWeights)),line[21])
            errorCount +=1.0
    errorRate = errorCount/numTest
    print("此次测试错误率为 %f" % errorRate)
    return errorRate

# 多次测试计算平均错误率
def multest(testTimes,trainData,testData):
    errorsum = 0.0
    for i in range(testTimes):
        trainSet,trainLabels = trainDataInput(trainData)
        weights = training(trainSet,trainLabels)
        errorate = testing(weights,testData)
        errorsum +=errorate
    print("经过 %d 次测试，最终测试平均错误率为 %f" % (testTimes,errorsum/testTimes))


multest(10,"horseColicTraining.txt","horseColicTest.txt")





