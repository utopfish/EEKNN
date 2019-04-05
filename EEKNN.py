from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
import numpy as np
import  math
import sys

class EEKNN():
    data=0
    target=0
    Wck=0
    neighbors=0
    def getData(self,trainData,trainTarget):
        '''
        存入训练集
        :param trainData:
        :param trainTarget:
        :return:
        '''
        self.data=trainData
        self.target=trainTarget

    def EEKNN(self):
        '''
        通过信息熵得到所有权重值
        :return:
        '''
        print("正在训练样本...")
        E=0
        for i in set(self.target):
            E-=self.p(i,self.target)*math.log2(self.p(i,self.target))
        E_aj_vjk=[{} for i in range(len(self.data[0]))]
        for i in range(len(self.data[0])):
                E_aj_vjk[i]=self.Eav(i)

        E_aj=self.Eaj(E_aj_vjk)
        IG_aj=E-E_aj
        IG_AJ_VJK=[{} for i in range(len(E_aj_vjk))]
        for index in range(len(E_aj_vjk)):
            for point in E_aj_vjk[index].items():
                IG_AJ_VJK[index][point[0]]=E-point[1]


        totalIG_aj=0
        for i in range(len(self.data[0])):
            totalIG_aj+=IG_aj[i]
        W_aj=np.zeros(len(self.data[0]))
        for i in range(len(self.data[0])):
            W_aj[i]=IG_aj[i]/totalIG_aj
        totalIG_aj_vjk=np.zeros(len(self.data[0]))

        for k in range(len(self.data[0])):
            for value in IG_AJ_VJK[k].values():
                totalIG_aj_vjk[k]+=value
        W_aj_vjk=[{} for i in range(len(self.data[0]))]
        for index in range(len(self.data[0])):
            for point in E_aj_vjk[index].items():
                W_aj_vjk[index][point[0]]=IG_AJ_VJK[index][point[0]]/totalIG_aj_vjk[index]
        self.W_aj=W_aj
        self.W_aj_vja=W_aj_vjk

    def test(self,testData,testTarget,n_neighbors,method="Entropy Euclidean"):
        print("正在使用测试数据集，计算准确率....")
        if method==None:
            method="Entropy Euclidean"
        distanceMatric=np.zeros((len(testData),len(self.data)))
        for i in range(len(testData)):
            for j in range(len(self.data)):
                distanceMatric[i][j]=self.D(testData[i],self.data[j],method=method)

        neighbors=np.zeros((len(distanceMatric),n_neighbors),dtype=int)
        for i in range(len(distanceMatric)):
            b=np.argsort(distanceMatric[i])
            neighbors[i]=b[1:n_neighbors+1]
        WckDistance=np.zeros((len(distanceMatric),n_neighbors),dtype=float)
        for i in range(len(distanceMatric)):
            temp=[]
            for j in neighbors[i]:
                temp.append(distanceMatric[i][j])

            for j in range(len(neighbors[i])):

                WckDistance[i][j]=self.getWck(j,temp)
        result = [{} for i in range(len(testTarget))]
        for i in range(len(testTarget)):
            for init in set(self.target):
                result[i][init] = 0
            for index, j in enumerate(neighbors[i]):

                result[i][self.target[j]] += WckDistance[i][index]
        predict = np.zeros(len(testTarget))
        for index, i in enumerate(result):
            predict[index] = max(i, key=i.get)


        return predict
    def score(self,fina,testTarget):
        count=0
        for i in range(len(testTarget)):
            if fina[i]==testTarget[i]:
                count+=1
        print("准确率："+str(count/len(testTarget)))
        return count/len(testTarget)


    #公式函数
    def getWck(self,j,neighborDistance):
        Dmin=min(neighborDistance)
        Dmax=max(neighborDistance)
        if Dmax==Dmin:
            return neighborDistance[j]
        else:
            return math.exp(-(neighborDistance[j]-Dmin)/(Dmax-Dmin))
    def getwighte(self,value,wight):
        '''
        通过特征值获取对应的权重值，如果特征值不在当前特征的权重表中，分
        如下三种情况：
        1.特征值大于最大值时，返回最大特征值的权重
        2.特征值小于最小值时，返回最小特征值的权重
        3.在最大值与最小值之间，但是不在表中，返回该特征值左右两个特征值权重的中值
        :param value: 特征值
        :param wight: 特征对应的所有特征值的权重值
        :return:
        '''
        maxKey=max(wight.keys())
        minKey=min(wight.keys())
        if value in set(wight.keys()):
            return wight[value]
        elif value > maxKey:
            return wight[maxKey]
        elif value < minKey:
            return wight[minKey]
        else:
            dict = sorted(wight.items(), key=lambda d: d[0])
            mark = 0
            for i in range(len(dict) - 1):
                if dict[i][0] > value and dict[i + 1][0]:
                    mark = i
                    break
            return (dict[mark][1]+dict[mark+1][1])/2
    def p(self,i,target):
        '''
        计算概率
        :param i:
        :param target:
        :return:
        '''
        count=0
        for item in target:
            if i==item:
                count+=1
        return count/len(target)
    def Eav(self,j):
        '''
        公式2
        :param j: 第j特征
        :return: 第j个特征每个属性值对应的信息熵
        '''
        ajvjk={}
        for i in set(self.data[:][j]):
            newTarget=[]
            for index,k in enumerate(self.data[:][j]):
                if i==k:
                    newTarget.append(self.target[index])
            result=0
            for m in set(self.target):
                try:
                    result-=self.p(m,newTarget)*math.log2(self.p(m,newTarget))
                except:
                    result=sys.maxsize
            ajvjk[i]=result
        return ajvjk
    def Eaj(self,E_aj_vjk):
        '''
        公式3 计算Eaj
        :param E_aj_vjk:
        :return:
        '''
        E_aj=np.zeros(len(self.data[0]))
        for k in range(len(self.data[0])):

            for j in set(self.data[:][k]):
                count = 0
                for i in range(len(self.data)):
                    if j == self.data[i][k]:
                        count+=1
                E_aj[k]+=count/len(self.data[0])*E_aj_vjk[k][j]
        return E_aj
    def D(self,pointX,pointY,method):
        '''
        两样样本之间的距离计算公式
        :param pointX:
        :param pointY:
        :param method:
        :return:
        '''
        W_aj=self.W_aj
        W_aj_Vjk=self.W_aj_vja
        distance=0
        if method!=None:
            if method=="Entropy Euclidean":
                for j in range(len(self.data[0])):
                    self.getwighte(pointX[j],wight=W_aj_Vjk[j])
                    daj=math.pow(self.getwighte(pointX[j],wight=W_aj_Vjk[j])*pointX[j]-self.getwighte(pointY[j],wight=W_aj_Vjk[j])*pointY[j],2)
                    distance+=daj*W_aj[j]
            elif method=="Entropy Manhattan":
                for j in range(len(self.data[0])):
                    daj=abs(self.getwighte(pointX[j],wight=W_aj_Vjk[j])*pointX[j]-self.getwighte(pointY[j],wight=W_aj_Vjk[j])*pointY[j])
                    distance+=daj*W_aj[j]
            elif method=="Entropy Canberra":
                for j in range(len(self.data[0])):
                    daj=abs((self.getwighte(pointX[j],wight=W_aj_Vjk[j])*pointX[j]-self.getwighte(pointY[j],wight=W_aj_Vjk[j])*pointY[j])/(self.getwighte(pointX[j],wight=W_aj_Vjk[j])*pointX[j]+self.getwighte(pointY[j],wight=W_aj_Vjk[j])*pointY[j]))
                    distance += daj * W_aj[j]
        else:
            raise BaseException("method can't be None")
        return distance



if __name__=="__main__":


    # data=datasets.load_boston()
    # trainData=data['data']
    # trainTarget=data['target']
    #
    # test = EEKNN()
    # test.getData(trainData,trainTarget)
    # test.EEKNN()
    # #运算测试集时可以选择，与训练集之间的距离度量方法，有如下三种：
    # #Entropy Euclidean 信息熵欧几里得距离
    # #Entropy Manhattan 信息熵曼哈顿距离
    # #Entropy Canberra  信息熵堪培拉距离
    # #下面的5是knn中k的个数
    # predict=test.test(data['data'],data['target'],5)
    # test.score(predict, data['target'])

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # 读取数据
    path = r"I:\人才才能预测论文\Human_performance.xlsx"
    data_0 = pd.read_excel(path, sheet_name='dataset')
    data = np.array(data_0)  # 转换成numpy类型，float类型

    print('Loading data...')

    x = data[:, :-1]  # x表示数据特征
    y = data[:, -1]  # y表示标签

    # 训练集测试集划分 | random_state：随机数种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # train_test_split应该是交叉验证函数，有训练集、验证集和测试集。
    trainData = x_train
    trainTarget = y_train

    """
    trainData=data['data']
    trainTarget=data['target']
    """
    test = EEKNN()
    test.getData(trainData, trainTarget)
    test.EEKNN()
    # 运算测试集时可以选择，与训练集之间的距离度量方法，有如下三种：
    # Entropy Euclidean 信息熵欧几里得距离
    # Entropy Manhattan 信息熵曼哈顿距离
    # Entropy Canberra  信息熵堪培拉距离
    # 下面的5是knn中k的个数
    """
    predict=test.test(data['data'],data['target'],5)
    test.score(predict, data['target'])
    """
    predict = test.test(x_test, y_test, 5)
    test.score(predict, y_test)

    # 用sklearn中的knn验证
    # knn = KNeighborsClassifier()
    # knn.fit(data['data'], data['target'])
    # predict = knn.predict(data['data'])
    # print(predict)
    # count=0
    # for i in range(len(predict)):
    #     if predict[i]==data['target'][i]:
    #         count+=1
    # print(count/len(predict))

    #用sklearn中的knn验证
    # knn = KNeighborsClassifier()
    # knn.fit(data['data'], data['target'])
    # predict = knn.predict(data['data'])
    # print(predict)
    # count=0
    # for i in range(len(predict)):
    #     if predict[i]==data['target'][i]:
    #         count+=1
    # print(count/len(predict))


