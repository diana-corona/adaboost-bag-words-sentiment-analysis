# Adaboost bag of words for sentiment analysis implemented from scratch

### Train and predict 
To train a model and validate the results using the test set run:
```
!python run.py --config '/content/adaboost-bag-words-sentiment-analysis/config/movie_review_train.yaml' --mode trainAndPredict 
```

### Tune hyperparameters
To check which hyperparameters works best for your training set run:
```
!python run.py --config '/content/adaboost-bag-words-sentiment-analysis/config/movie_review_train.yaml' --mode tuneHyperparams 
```
### Jupyter Notebook Demo
A Jupyter Notebook Demo could be found in: 
```
adaboost-bag-words-sentiment-analysis-demo.ipynb

```