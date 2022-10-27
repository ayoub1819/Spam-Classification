from src.modules.SPAM import Spam_Classifier
import pandas as pd
import pickle


data = pd.read_csv("./data/mail_data.csv")

app = Spam_Classifier(data)

app.process_data()

out = app.trained_models()
models = out[0]


svm_model = models["SVC"]
knn_model = models["KN"]
dt_model = models["DT"]
rf_model = models["RF"]

svm = 'svm_model.sav'
random_forest = 'random_forest_model.sav'
knn = 'knn_model.sav'
descision_tree = 'descision_tree_model.sav'

pickle.dump(svm_model, open(svm, 'wb'))
pickle.dump(rf_model, open(random_forest, 'wb'))
pickle.dump(knn_model, open(knn, 'wb'))
pickle.dump(dt_model, open(descision_tree, 'wb'))
