import warnings
warnings.simplefilter("ignore",category=FutureWarning)

#################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
#--------------------------
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#---------------------------
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.datasets import make_classification
#################################################################
class Spam_Classifier(object):
    def __init__(self,data):
        self.data = data
        self.feature_extraction = None

        self.X_train_features = None
        self.X_test_features = None
        
        self.Y_train = None
        self.Y_test = None
        
    def process_data(self):
        mail_data = self.data.copy()
        mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
        mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

        X = mail_data['Message']
        Y = mail_data['Category']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

        # transform the text data to feature vectors 
        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
        self.X_train_features = feature_extraction.fit_transform(X_train)
        self.X_test_features = feature_extraction.transform(X_test)
        self.feature_extraction = feature_extraction
        # convert Y_train and Y_test values as integers
        self.Y_train = Y_train.astype('int')
        self.Y_test = Y_test.astype('int')
        
        return[self.X_train_features,self.X_test_features,self.Y_train,self.Y_test]
    
    def transform_data(self,input_mail):
        vector = self.feature_extraction.transform(input_mail)
        return vector 

    def train(self,clf, features, targets):
        clf.fit(features, targets)

    def predict(self,clf, features):
        return (clf.predict(features))

    def trained_models(self):
        #initialize multiple classification models
        svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
        knc = KNeighborsClassifier(n_neighbors=70)
        dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
        rfc = RandomForestClassifier(n_estimators=31, random_state=111)
        #create a dictionary of variables and models
        clfs = {'SVC' : svc,'KN' : knc, 'DT': dtc, 'RF': rfc}
        pred_scores_word_vectors = []
        for k,v in clfs.items():
            self.train(v, self.X_train_features, self.Y_train)
            pred = self.predict(v, self.X_test_features)
            pred_scores_word_vectors.append((k, [accuracy_score(self.Y_test , pred)]))
        return [clfs,pred_scores_word_vectors]

    def evaluate(self,models_score):
        clfs = models_score[0]
        scores = models_score[1]
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
        for k,v in clfs.items():
            y_score = v.predict_proba(self.X_test_features)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.Y_test, y_score,)
            print(f"{k}: {thresholds}")
            auc_score = auc(fpr, tpr)
            name = f"{k} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        return [fig,y_score,scores]
        
