from modules.BoostingClassifier import BoostingClassifier
from modules.BagOfWords import extract_bag_of_words
from sklearn.metrics import accuracy_score
import numpy as np

def save(config,X_test,y_pred):
    sentiment_dict = {'1':'positive','0':'negative'}
    y_results = [sentiment_dict[str(int(prediction))] for prediction in y_pred]
    np.savetxt(config['results_file_path'], y_results, header='sentiment_prediction', fmt='%s')
    


def predict(config,boostingClassifier):
    (X_test, Y_test) = extract_bag_of_words(config['test_file_path'],config)
    bc = boostingClassifier
    y_pred = bc.predict(X_test)    
    print('accuracy', accuracy_score(Y_test, y_pred))
    if 'results_file_path' in config:
        save(config,X_test,y_pred)
    return y_pred

def train_and_predict(config):
    (X_train, Y_train) = extract_bag_of_words(config['train_file_path'],config)
    (X_test, Y_test) = extract_bag_of_words(config['test_file_path'],config)
    bc = BoostingClassifier(config)
    bc.fit(X_train, Y_train)
    y_pred = bc.predict(X_test)    
    print('accuracy', accuracy_score(Y_test, y_pred))
    sentiment_dict = {1:'positive',0:'negative'}
    if 'results_file_path' in config:
        save(config,X_test,y_pred)
    return y_pred