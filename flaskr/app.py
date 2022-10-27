import json
from flask import Flask, jsonify, render_template, request
import pandas as pd
import pickle
import plotly
import plotly.express as px
from flaskr.src.modules.SPAM import Spam_Classifier

app = Flask(__name__,template_folder="./src/templates",static_folder="public")

# Loading the trained models
knn_model = pickle.load(open('./src/models/knn_model.sav', 'rb'))
svm_model = pickle.load(open('./src/models/svm_model.sav', 'rb'))
rf_model = pickle.load(open('./src/models/random_forest_model.sav', 'rb'))
dt_model = pickle.load(open('./src/models/descision_tree_model.sav', 'rb'))


# preprocessing of tha data and creating the TF-IDF vertorize for feature exraction 
data = pd.read_csv("./data/mail_data.csv")
spam_app = Spam_Classifier(data)
spam_app.process_data()



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test',methods=['POST'])
def testModel():
    
    email = request.get_json()['email']
    vector = spam_app.transform_data([email])

    svm_out = svm_model.predict(vector)
    knn_out = knn_model.predict(vector) 
    dt_out = dt_model.predict(vector)
    rf_out = rf_model.predict(vector)
    
    svm_class = 'spam' if svm_out[0]==0 else 'ham'
    knn_class = 'spam' if knn_out[0]==0 else 'ham'
    dt_class =  'spam'  if dt_out[0]==0 else 'ham'
    rf_class =  'spam'  if rf_out[0]==0 else 'ham'

    output  = {
        'svm' : svm_class,
        'knn' : knn_class,
        'dt' : dt_class,
        'rf' : rf_class,
    }

    return jsonify(output)





@app.route('/evaluate',methods=['POST'])
def evaluateModel():
    models_scores = spam_app.trained_models()
    evalution = spam_app.evaluate(models_scores)
    curve = evalution[0]

    json_plots = json.dumps([curve], cls=plotly.utils.PlotlyJSONEncoder)
    
    return json_plots

