from modules.BoostingClassifier import BoostingClassifier
from modules.BagOfWords import extract_bag_of_words
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
'''-----------------------------------------'''
'''-     TUNE ADABOOST HYPERPARAMS         -'''
'''-----------------------------------------'''
def tune(config):
    Tune_train_folder_name=config['train_file_path']
    (Tune_X_train, Tune_Y_train) = extract_bag_of_words(Tune_train_folder_name,config)
    Tune_Xtrain_O, Tune_Xtest_O, Tune_ytrain_O, Tune_ytest_O = train_test_split(Tune_X_train, Tune_Y_train, train_size=0.8, random_state=123)

    tune_params = config['AdaboostClassifier']['TuneHyperparams']
    depth_list = tune_params['stump_depth']
    number_stumps_list = tune_params['num_stumps']
    Tune_score_Xtest = np.zeros((len(depth_list),len(number_stumps_list)))
    fk_n_splits = tune_params['KFold_n_splits']
    kf = KFold(n_splits = fk_n_splits)
    kf.get_n_splits(Tune_X_train)

    for train_index, test_index in kf.split(Tune_Xtrain_O):
        #print("TRAIN:", train_index, "TEST:", test_index)
        Tune_Xtrain, Tune_Xtest = Tune_Xtrain_O[train_index], Tune_Xtrain_O[test_index]
        Tune_ytrain, Tune_ytest = Tune_ytrain_O[train_index], Tune_ytrain_O[test_index]
        for nd,depth in zip(range(len(depth_list)),depth_list):
            for n_estump,ns in zip(number_stumps_list,range(len(number_stumps_list))):
                Tune_accuracy_score = 0 
                Tune_bc = BoostingClassifier(stumpsNum=n_estump,stump_depth=depth)
                Tune_bc.fit(Tune_Xtrain, Tune_ytrain)
                #predict test 
                Tune_y_pred = Tune_bc.predict(Tune_Xtest)
                Tune_accuracy_score = accuracy_score(Tune_ytest,Tune_y_pred)
                Tune_score_Xtest[nd][ns] += Tune_accuracy_score

    Tune_score_Xtest = Tune_score_Xtest/(fk_n_splits)
    Tune_score_Xtest = np.round(Tune_score_Xtest,2)
    
    best_values = np.where(Tune_score_Xtest == np.max(Tune_score_Xtest))
    print("The best depth is ", depth_list[best_values[0][0]]," The best depth is ", number_stumps_list[best_values[0][1]], "With score ",np.max(Tune_score_Xtest))




