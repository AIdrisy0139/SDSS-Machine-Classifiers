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
#Model Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    '''
    modelB = KNeighborsClassifier(n_neighbors = 100)
    modelB.fit(xTrain,yTrain)
    modelBPredict = modelB.predict(xTest)
    print(getAccuracy(yTest,modelBPredict))

    modelC = KNeighborsClassifier(n_neighbors = 500)
    modelC.fit(xTrain,yTrain)
    modelCPredict = modelC.predict(xTest)
    print(getAccuracy(yTest,modelCPredict))
    '''

def trainKNNTwoFeatures(k):
    global dataTable
    xTrain = dataTable["xStdTrain"]
    xTest = dataTable["xStdTest"]
    yTrain = dataTable["yTrain"]
    yTest = dataTable["yTest"]
    fNum = 6
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
    

def trainKNNAllFeatures(k): 
    global dataTable
    model,pred = kNearest(dataTable["xStdTrain"],dataTable["xStdTest"],dataTable["yTrain"],dataTable["yTest"],k)
    return getAccuracy(dataTable["yTest"],pred)

def accVersusK():
    global dataTable
    print("accVersusK")
    allAcc = []
    low = 50
    up = 1000
    step = 100
    xAxis = range(low,up,step)

    for i in range(low,up,step):
        preds = trainKNNTwoFeatures(i)
        currAcc = 0
        for p in preds:
            currAcc += getAccuracy(p,dataTable["yTest"])
        avgAcc = currAcc/15
        allAcc.append(avgAcc)

    plt.figure("Accuracy vs Neighbor Count (K)")
    plt.suptitle("Accuracy vs Neighbor Count (K)")
    plt.plot(xAxis,allAcc)
    #print(allAcc)
    #print(preds)

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

def accVersusC():
    print("accVersusC")
    linearPreds = []
    polyPreds = []
    sigPreds=[]
    lowBound = 10
    upBound = 50
    step = 10
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




def main():
    allData = lockAndLoad()
    #plotAverages(allData)
    
    # Start Training Models
    #trainKNNTwoFeatures(50)
    #allData['xStdTrain'],allData["xStdTest"],allData['yTrain'],allData["yTest"]
    #print(trainKNNAllFeatures(50))

    accVersusK()
    #accVersusC()

    #trainSVMAllFeatures()

    #plotData(allData["raw"])
    plt.show()

    '''
    print("---STAR---")
    print(stars.shape)
    print("---GALAXY---")
    print(galaxies.shape)
    print("---QSO---")
    print(quasars.shape)
    '''

    #TODO: Plot Averages
main()
