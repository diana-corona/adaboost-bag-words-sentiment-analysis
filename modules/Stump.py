import numpy as np
from sklearn import tree

from math import log
from math import exp

class Stump:
    def __init__(self,x,y,criterion,stump_depth):
        self.x = x
        self.y = y
        self.newX = None
        self.newY = None
        self.stump = None
        self.max_depth = stump_depth
        self.totalError = 0
        self.amountOfSay = 0
        self.criterion = criterion
        self.sampleSumedWeight = np.full((len(self.x),1),1/len(self.x))
        self.sampleWeight = np.full((len(self.x),1),1/len(self.x))
        
    def createStump(self):
        self.stump = tree.DecisionTreeClassifier(criterion = self.criterion  , splitter = "best" ,max_depth = self.max_depth)
        self.stump = self.stump.fit(self.x,self.y)
        #self.printStump()
        #print("----------")
        yPredicted = self.stump.predict(self.x)
        self.initializeSampleWeight()
        self.updateTotalError(yPredicted)
        self.updateAmountOfSay()
        self.updateSampleWeight(yPredicted)
        self.normalizeSampleWeight()
        self.newX,self.newY = self.createRandomDataset()
       
    def printStump(self):
        tree.plot_tree(self.stump)
        
    def initializeSampleWeight(self):
        self.sampleWeight = np.full((len(self.x),1),1/len(self.x))
        for i in range(1,len(self.sampleWeight)):
            self.sampleSumedWeight[i]= self.sampleWeight[i]+self.sampleWeight[i-1]
        
    def updateTotalError(self,yPredicted): 
        self.totalError = (len(self.y)-sum(self.y==yPredicted))/len(self.y)
        
    def updateAmountOfSay(self):
        if not(self.totalError == 0):
            self.amountOfSay = 1/2*(log((1-self.totalError)/self.totalError))
        else:
            self.amountOfSay = 1
            
    def updateSampleWeight(self,yPredicted):
        y = self.y
        for i in range(len(y)):
            if y[i]==yPredicted[i]:
                self.sampleWeight[i]= self.sampleWeight[i] * exp(-self.amountOfSay)
            else:
                self.sampleWeight[i]= self.sampleWeight[i] * exp(self.amountOfSay)
        
    def normalizeSampleWeight(self):
        sumWeight = sum(self.sampleWeight)
        self.sampleWeight = self.sampleWeight/sumWeight
        #self.sampleWeight = np.cumsum(self.sampleWeight)
        for i in range(1,len(self.sampleWeight)):
            self.sampleSumedWeight[i]= self.sampleWeight[i]+self.sampleSumedWeight[i-1]
        
    def createRandomDataset(self):
        if self.totalError == 0 :
            #return pd.DataFrame(self.x, columns= self.x.columns),pd.DataFrame(self.y, columns= self.y.columns)
            return self.x,self.y
        newRandomDatasetX = []
        newRandomDatasetY = []
        for i in range(len(self.x)): 
            randNum = np.random.uniform(size=1)
            for j in range(len(self.x)):
                if float(self.sampleSumedWeight[j]) > float(randNum) :
                    rowx = self.x[j]
                    rowy = self.y[j]
                    newRandomDatasetX.append(rowx)
                    newRandomDatasetY.append(rowy)
                    break
                
        return newRandomDatasetX, newRandomDatasetY
        