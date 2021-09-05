
from modules.BoostingClassifier import BoostingClassifier
from modules.BagOfWords import extract_bag_of_words
from sklearn.metrics import accuracy_score

def train(config):
	(X_train, Y_train) = extract_bag_of_words(config['train_file_path'],config)
	bc = BoostingClassifier()
	bc.fit(Xtrain, ytrain)
	y_pred = bc.predict(Xtest)
	print('train accuracy', accuracy_score(ytest, y_pred))
	return bc