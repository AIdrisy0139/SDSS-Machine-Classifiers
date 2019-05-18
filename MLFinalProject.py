'''
AUTHOR: ARIKUZZAMAN IDRISY 
ML: FINAL PROJECT
COURSE: CSC 59929 - Intro into Machine Learning
INSTRUCTOR: Erik Grimmelmann
COLLEGE: THE CITY COLLEGE OF NEW YORK
'''
#Tool Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
#Model Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

'''
----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

Data Loading and Exploring Functions

-------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

'''
dataTable = {}
def lockAndLoad():
    global dataTable
    file = open('SDSS_DR14.csv','r')

    df = pd.read_csv(file,
                    header =0 ,
                    usecols=["u","g","r","i","z","redshift","class"]
                    )
    #print(df)
    useCols = ["u","g","r","i","z","redshift","class"]
    dfret = df.reindex(columns= useCols)
    #print(dfret)
    data = dfret.values #Numpy Array of the data

    # return data
    features = np.array([data.T[0],data.T[1],data.T[2],data.T[3],data.T[4],data.T[5]]).T
    targetNames = np.array(data.T[6]).T
    
    targets = np.zeros(10000)
    for i in range(0,10000):
        if targetNames[i]=="STAR":
            targets[i] = 0
        if targetNames[i]=="GALAXY":
            targets[i] = 1
        if targetNames[i]=="QSO":
            targets[i] = 2

    print(targets.shape)

    #Split the Data into training and testing
    xTrain, xTest,  yTrain, yTest = train_test_split(features, targets, test_size=0.3, random_state=1, stratify=targets)
    #Normalize the Data
    sc = StandardScaler()
    sc.fit(xTrain)
    xStdTrain = sc.transform(xTrain)
    xStdTest = sc.transform(xTest)
    xStd = sc.transform(features)
    
    dataTable = {
            "features":features,
            "targets":targets,
            "xStdTrain":xStdTrain,
            "xStdTest" :xStdTest,
            "xStd": xStd,
            "yTrain" :yTrain,
            "yTest" :yTest,
            "targetNames" : targetNames,
            "raw":data}
    return dataTable
def makeColorMap():
    color = ("red","green","blue")
    
def plotData(data):
    stars = galaxies = quasars = np.array([["x","x","x","x","x","x","x"]])
    colorMap = []
    for stObj in data:
        if stObj[6] == "STAR":
            stars = np.vstack((stars,stObj))
            colorMap.append("red")
        if stObj[6] == "QSO":
            quasars = np.vstack((quasars,stObj))
            colorMap.append("green")
        if stObj[6] == "GALAXY":
            galaxies = np.vstack((galaxies,stObj))
            colorMap.append("blue")

    stars = np.delete(stars,0,axis=0)
    quasars = np.delete(quasars,0,axis=0)
    galaxies = np.delete(galaxies,0,axis=0)

    print("stars {}, qso: {},galaxies{}".format(len(stars),len(quasars),len(galaxies)))
    
    npColorMap = np.append(
        np.append
            ( np.full(4152,"red"),np.full(4998,"blue")),
        np.full(850,"green") )
    stackedData = np.vstack((stars,galaxies,quasars))
    plt.figure("Stellar Objects u vs g")
    plt.suptitle("Stellar Objects u vs g")
    #plt.scatter(stackedData.T[0],stackedData.T[1],c=npColorMap)
    #plt.legend()
    pltStar =  plt.scatter(stars.T[0],stars.T[1],color="red")
    pltGalaxy = plt.scatter(galaxies.T[0],galaxies.T[1],color="blue")   
    pltQso = plt.scatter(quasars.T[0],quasars.T[1],color="green")
    plt.legend(
        (pltStar,pltGalaxy,pltQso),("Stars","Galaxies","Quasars"),
        scatterpoints=1, loc="lower left", fontsize=10
    )

    fig = plt.figure("3D")
    fig.suptitle("THE DATA")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stars.T[0],stars.T[1],stars.T[5] ,color="red",marker="x")
    ax.scatter(quasars.T[0],quasars.T[1],quasars.T[5] ,color="green",marker="o")
    ax.scatter(galaxies.T[0],galaxies.T[1],galaxies.T[5] ,color="blue",marker="^")
    ax.set_xlabel("U Values")
    ax.set_ylabel("G Value")
    ax.set_zlabel("Redshift")
    ax.legend(["Star","Quasar","Galaxy"])
    #plt.figure("Stars, Galaxies, and Quasars")
def getAccuracy(A,B):
    correct = 0
    for i in range(len(A)):
      if(A[i]==B[i]):
        correct += 1
    return correct/len(A)

def getAverage(A):
    sum = 0
    for each in A:
        sum += each
    return sum/len(A)

def getAvg2D(A):
    sum = 0;
    for i in A:
        for j in i:
            sum +=j
    return sum/(len(A) * len(A[0]) )

'''
----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

KNN Functions

-------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

'''
def kNearest(xTrain,xTest, yTrain,yTest,neighbors):
    # Train a KNN Classifier on K=50,100,500 
    modelA = KNeighborsClassifier(n_neighbors = neighbors)
    modelA.fit(xTrain,yTrain)
    modelAPredict = modelA.predict(xTest)
    #print(getAccuracy(yTest,modelAPredict))

    return modelA,modelAPredict


def trainKNNTwoFeatures(xTrain,xTest,yTrain,yTest,k,f):
    global dataTable
    fNum = f
    kNearestArray = []
    kNNPredictArray = []
    for i in range(0,fNum):
        for j in range(i+1,fNum):
            xTrainFeatures = xTrain[:,[i,j]]
            xTestFeatures = xTest[:,[i,i]]
            kNNModel, kNNPredict = kNearest(xTrainFeatures,xTestFeatures,yTrain,yTest,k)
            kNearestArray.append(kNNModel)
            kNNPredictArray.append(kNNPredict)
            plt.figure("KNN: " + str(i) + "," + str(j))
            plot_decision_regions(dataTable["xStd"],dataTable["targets"],classifier=kNNModel,test_idx=range(7000,7000))
            print("Axis 1: " + str(i) +"  Axis 2:"+ str(j) + "  Accuracy: "+ str(getAccuracy(kNNPredict,yTest)))

    return kNNPredictArray
    
def slimKNNTwoFeatures(xTrain,xTest,yTrain,yTest,k,f):
    headers = ["u","g","r","i","z","redshift"]
    fNum = f
    testScores=[]
    trainScores=[]
    for i in range(0,fNum):
        for j in range(i+1,fNum):
            xTrainFeatures = xTrain[:,[i,j]]
            xTestFeatures = xTest[:,[i,i]]
            model, kNNPredict = kNearest(xTrainFeatures,xTestFeatures,yTrain,yTest,k)
            modelScore = model.score(xTestFeatures,yTest)
            trainScore = model.score(xTrainFeatures,yTrain)
            print("Axis 1: " + headers[i] +"  Axis 2:"+ headers[j] + "  Accuracy: "+ str(modelScore))
            
            trainScores.append(trainScore)
            testScores.append(modelScore)

    return getAverage(trainScores),getAverage(testScores)


def accVersusK():
    global dataTable
    '''
    testSubX = dataTable["xStdTest"][:,[sliceA,sliceB]]
    trainSubX = dataTable["xStdTrain"][:,[sliceA,sliceB]]
    '''
    print("accVersusK")
    allAccTrain = []
    allAccTest = []
    low = 50
    up = 1000
    step = 100
    xAxis = range(low,up,step)

    for i in range(low,up,step):
        #model,preds = kNearest(trainSubX,testSubX,yTrain,yTest,i)
        avgAccTrain,avgAccTest = slimKNNTwoFeatures(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],i,6)
        allAccTrain.append(avgAccTrain)
        allAccTest.append(avgAccTest)

    plt.figure("Accuracy vs Neighbor Count (K)")
    plt.suptitle("Accuracy vs Neighbor Count (K)")
    plt.plot(xAxis,allAccTest,color="orange")
    plt.plot(xAxis,allAccTrain,color="blue")
    plt.legend(["Testing Accuracy","Training Accuracy"])
    plt.xlabel("K: Neighbor Count")
    plt.ylabel("Accuracy")
    #print(allAcc)
    #print(preds)
'''
def trainKNNAllFeatures(xTrain,xTest,yTrain,yTest,k): 
    global dataTable
    model,pred = kNearest(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],k)
    return getAccuracy(dataTable["yTest"],pred)
'''
def trainKNNAllFeatures(xTrain,xTest,yTrain,yTest,k): 
    global dataTable
    model,pred = kNearest(xTrain,xTest,yTrain,yTest,k)
    return getAccuracy(yTest,pred)
'''
----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

SVC Functions

-------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

'''

def supportVector(xTrain,xTest,yTrain,yTest,cval):
    # Train the data with all the kernals
    # Return a dictionary of dictionarys with the key model/predict and then being kernal
    # cval = C (regularization apram)

    #Linear
    linearSVM = svm.SVC(C=cval,kernel="linear",random_state=1)
    linearSVM.fit(xTrain,yTrain)
    linearPred = linearSVM.predict(xTest)

    #PolyNomial 
    polySVM = svm.SVC(C=cval, kernel="poly",degree=3,random_state=1)
    polySVM.fit(xTrain,yTrain)
    polyPred = polySVM.predict(xTest)

    #Sigmoid SVM
    sigSVM = svm.SVC(C=cval, kernel="rbf",random_state=1)
    sigSVM.fit(xTrain,yTrain)
    sigPred = sigSVM.predict(xTest)

    predictDict = {
        "linear":linearPred,
        "poly": polyPred,
        "sigmoid":sigPred
    }
    return predictDict

def trainSVMAllFeatures(c):
    print("SVM's on all Features")
    global dataTable
    predictions = supportVector(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],c)

    linearAcc = getAccuracy(predictions["linear"],dataTable["yTest"])
    polyAcc = getAccuracy(predictions["poly"],dataTable["yTest"])
    sigAcc = getAccuracy(predictions["poly"],dataTable["yTest"])
    
    print("The Accuracy of a Linear: {}, Polynomial: {}, Sigmoid {}".format(linearAcc,polyAcc,sigAcc))

    return [linearAcc,polyAcc,sigAcc]
    #for key in predictions():
    #    print("The {} Model has accuracy of {}".format(key,getAccuracy(predictions[key])))

def accVersusC(lb,ub,s):
    print("accVersusC")
    linearPreds = []
    polyPreds = []
    sigPreds=[]
    lowBound = lb#10
    upBound = ub#50
    step = s#10
    for i in range(lowBound,upBound,step):
        preds = trainSVMAllFeatures(i)
        linearPreds.append(preds[0])
        polyPreds.append((preds[1]))
        sigPreds.append((preds[2]))

    xAxis = range(lowBound,upBound,step)
    print(len(xAxis))
    print(xAxis)
    plt.figure("Accuracy vs Regularization Factor")
    plt.suptitle("Accuracy vs Regularization Factor")

    lpPLT = plt.plot( xAxis,linearPreds,color="red")
    ppPLT = plt.plot( xAxis,polyPreds ,color="green")
    spPLT = plt.plot( xAxis,sigPreds ,color="blue")
    '''plt.legend(
    (lpPLT,ppPLT,spPLT),("Linear Kernel","Polynomial = 3 Kernel","Sigmoid Kernel"),
     loc="lower left", fontsize=10
    )'''
    plt.legend(["Linear","Polynomial","Sigmoid"])


'''
----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

DECISION TREE FUNCTIONS

-------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

'''

def dTree(xTrain,xTest,yTrain,yTest,crit,depth):
    model = DecisionTreeClassifier(criterion=crit,max_depth=depth,random_state=1)
    model.fit(xTrain,yTrain)

    preds  = model.predict(xTest)

    trainingScore = model.score(xTrain,yTrain)
    testingScore = model.score(xTest,yTest)
    acc = getAccuracy(yTest,preds)
    #print("Accuracy of D-Tree with {} Empurity = {} at MaxDepth = {}".format(crit,acc))
    return trainingScore,testingScore

def treeRunner(xTrain,xTest,yTrain,yTest,maxDepth):
    print("GINI Impurity Testing")
    print("Depth || Training Score || Testing Score")
    giniTrain = []
    giniTest=[]
    for i in range(1,maxDepth):
        trainScore, testScore = dTree(xTrain,xTest,yTrain,yTest,"gini",i)
        giniTrain.append(trainScore)
        giniTest.append(testScore)
        print(i,trainScore,testScore)

    print()
    print("ENTROPY Impurity Testing")
    print("Depth || Training Score || Testing Score")
    entropyTrain=[]
    entropyTest=[]
    for i in range(1,maxDepth):
        trainScore, testScore =dTree(xTrain,xTest,yTrain,yTest,"entropy",i)
        entropyTest.append(testScore)
        entropyTrain.append(trainScore)
        print(i,trainScore,testScore)
    
    xAxis = range(1,maxDepth)
    plt.figure("Decision Tree Scores")
    plt.suptitle("Decision Tree Scores")
    plt.plot(xAxis,entropyTest, color="red")
    plt.plot(xAxis,entropyTrain,color = "orange")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")

    plt.plot(xAxis,giniTest, color="purple")
    plt.plot(xAxis,giniTrain, color = "blue")
    plt.legend(["Entropy Test"," Entropy Train","Gini Test","Gini Train"])
'''
----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

ADA BOOST CLASSIFIER FUNCTIONS

-------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

'''
def trainAdaBoost(xTrain,xTest,yTrain,yTest,est,nu):
    ada = AdaBoostClassifier(n_estimators = est,learning_rate=nu ,random_state=0 )
    ada.fit(xTrain,yTrain)

    return ada.score(xTest,yTest)

def iterAdaOnEstimators(maxEst,step):
    xTrain = dataTable["xStdTrain"]
    xTest = dataTable["xStdTest"]
    yTrain = dataTable["yTrain"]
    yTest = dataTable["yTest"]
    score = []
    xAxis = range(50,maxEst,step) 
    for i in xAxis:
        score.append(trainAdaBoost(xTrain,xTest,yTrain,yTest,i,1.0))
    
    plt.figure("Accuracy versus Estimator Count")
    plt.suptitle("Accuracy versus Estimator Count")
    plt.plot(xAxis, score)

def iterAdaOnNu(maxNu,step):
    xTrain = dataTable["xStdTrain"]
    xTest = dataTable["xStdTest"]
    yTrain = dataTable["yTrain"]
    yTest = dataTable["yTest"]
    score = []
    xAxis = range(1,maxNu,step)
    for i in xAxis:
        score.append(trainAdaBoost(xTrain,xTest,yTrain,yTest,100,i))
    
    plt.figure("Accuracy versus Learning Rate Max:{}, Step:{}".format(maxNu,step))
    plt.suptitle("Accuracy versus Learning Rate Max:{}, Step:{}".format(maxNu,step))
    plt.plot(xAxis,score)
        

def main():
    allData = lockAndLoad()
    #plotAverages(allData)
    xTrain = dataTable["xStdTrain"]
    xTest = dataTable["xStdTest"]
    yTrain = dataTable["yTrain"]
    yTest = dataTable["yTest"]
    #x = kNearest(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],50)
    
    #Start Training Models
    print("KNN250")
    #trainKNNTwoFeatures(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],250,6)
    print("knn500")
    #trainKNNTwoFeatures(500)
    #allData['xStdTrain'],allData["xStdTest"],allData['yTrain'],allData["yTest"]
    #print(trainKNNAllFeatures(50))
    plotData(allData["raw"])
    #accVersusK()
    #accVersusC(10,100,10)

    #trainSVMAllFeatures()

    #plotData(allData["raw"])
    #giniPreds = dTree(xTrain,xTest,yTrain,yTest,"gini",10)
    #entPreds = dTree(xTrain,xTest,yTrain,yTest,"entropy",10)
    #treeRunner(xTrain,xTest,yTrain,yTest,15)

    #trainAdaBoost(xTrain,xTest,yTrain,yTest,100,1.0)
    #iterAdaOnNu(20,1)
    #iterAdaOnEstimators(200,20)
    plt.show()


    #TODO: Plot Averages
main()
