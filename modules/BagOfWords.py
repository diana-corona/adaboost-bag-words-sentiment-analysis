import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize  
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

def extract_bag_of_words(review_file,config):
    xy = pd.read_csv(review_file)
    x_original = xy[config['review_col_name']]
    x_filtered = []
    x = []
    y_original = xy[config['sentiment_col_name']].to_numpy()
    y = np.zeros((len(y_original)))
    for review,yindex in zip(x_original,range(len(y_original))) :
        all_words = word_tokenize(review)
        #lowercase text
        lower_case_words = [w.lower() for w in all_words]
        words = [word for word in lower_case_words if word.isalpha()]
        #remove unwanted words
        remove_words_list = set(stopwords.words(config['languaje']))
        filtered_words = [w for w in words if not w in remove_words_list]
        filtered_words = [w for w in filtered_words if not w in "br"]
        #lemmatenize words
        lemmatized_words = [WordNetLemmatizer().lemmatize(w) for w in filtered_words]
        #steam words
        porter = SnowballStemmer(config['languaje'])
        stemmed_words = [porter.stem(word) for word in lemmatized_words]
        final_text = ' '.join([str(word) for word in stemmed_words ])
        x_filtered.append(final_text)
        #y values from text to number
        if y_original[yindex]=='positive':
            y[yindex]=1
        else:
            y[yindex]=0
            
    #create bags of words 
    vectorizer = TfidfVectorizer()
    bag_of_words = vectorizer.fit_transform(x_filtered)
    x = bag_of_words.toarray()
    #PCA
    pca = PCA(config['pca'])
    pca.fit(x)
    x = pca.transform(x)

    return x,y
