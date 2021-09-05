from modules.Stump import Stump
import numpy as np

class BoostingClassifier:
    def __init__(self,config):
        config = config['AdaboostClassifier']
        self.stumpsNum = config['num_stumps']
        self.stumpList = []
        self.criterion = config['criterion']
        self.stump_depth = config['stump_depth']
        
    def fit(self, x,y,iteration=1):
        st= Stump(x,y,self.criterion,self.stump_depth)
        st.createStump()
        self.stumpList.append(st)
        if iteration == self.stumpsNum or st.amountOfSay==1:
            return
        else:
            self.fit(st.newX,st.newY,iteration+1)
            
    def predict(self, x):
        prediction_list = np.zeros((len(x),1))
        for xrow, i in zip(x,range(len(x))):
            prediction = {}
            for stump in self.stumpList:
                result = stump.stump.predict(xrow.reshape(1,len(xrow)))
                result = str(float(result[0]))
                if result in prediction.keys():
                    prediction[result] = prediction[result] + stump.amountOfSay
                else: 
                    prediction[result] = stump.amountOfSay
            prediction_list[i]=max(prediction, key=prediction.get)
        return prediction_list