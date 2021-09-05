from tune import tune
from train import train
from predict import predict
from predict import train_and_predict

from argparse import ArgumentParser

import sys
import yaml

from argparse import ArgumentParser
if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()

    #parser.add_argument("--file", required=True, help="path to text file to process")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["tuneHyperparams","train", "predict","trainAndPredict"])

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)
        config = config['config']

    boostingClassifier = None
    if opt.mode == 'train':
        print("Training...")
        boostingClassifier = train(config)
    elif opt.mode == 'tuneHyperparams':
        print("Tune hyperparams...")
        tune(config)
    elif opt.mode == 'predict':
        print("Predict...")
        if boostingClassifier == None:
            print("First train the model")
        else:
            predict(config,boostingClassifier)
    elif opt.mode == 'trainAndPredict':
        print("Train And Predict...")
        train_and_predict(config)
