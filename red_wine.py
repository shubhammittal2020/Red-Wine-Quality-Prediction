import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%matplotlib inline

data = pd.read_csv("winequality-red.csv")

x = data.drop(["quality"], axis = 1)
y = data["quality"]
x_train, x_test, y_train, y_test = train_test_split(x, y)



# Predictions by Linear Regression
from sklearn.linear_model import LinearRegression

LinearModel = LinearRegression()
LinearModel.fit(x_train, y_train)
LinearPredictions = LinearModel.predict(x_test)
LinearModel.score(x_test, LinearPredictions)
np.sqrt((np.sum((LinearPredictions - y_test)**2)/1250))/2     #Calculate cost by cost function



# Predictions by Logistic Regressions
from sklearn.linear_model import LogisticRegression
LogisticModel = LogisticRegression()
LogisticModel.fit(x_train, y_train)
LogisticPredictions = LogisticModel.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, LogisticPredictions)
correct_predictions = 0+0+134+96+0+0
total = 400
score = (correct_predictions/total)*100
print(score)



# Prediction by Random Forest
from sklearn.tree import DecisionTreeClassifier
DecisionPredictor = DecisionTreeClassifier()
DecisionPredictor.fit(x_train, y_train)
pred = DecisionPredictor.predict(x_test)
#file = open('result.csv', mode='w')
#file.write(str(pred))
#pd.read_csv('result.csv')
confusion_matrix(y_test, pred)
DecisionPredictor.score(x_test, y_test)
from sklearn.ensemble import RandomForestClassifier
RandomDecisionPredictor = RandomForestClassifier(n_estimators= 100)
RandomDecisionPredictor.fit(x_train, y_train)
RandomDecisionPredictor.predict(x_test)
RandomDecisionPredictor.score(x_test, y_test)



# Support Vector Machine
#Linear Implementation
from sklearn import svm
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_std_train = sc_x.fit_transform(x_train)
c = 0.01 #1.0
clf = svm.SVC(kernel='linear', C = c)
clf.fit(x_std_train, y_train)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

###### Cross Validation within Train Dataset
res = cross_val_score(clf, x_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

y_train_pred = cross_val_predict(clf, x_std_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
print("Precision Score: \t {0:.4f}".format(precision_score(y_train, y_train_pred, average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train, y_train_pred, average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train, y_train_pred, average='weighted')))

#### Cross Validation within Test Dataset
y_test_pred = cross_val_predict(clf, sc_x.transform(x_test), y_test, cv=3)
confusion_matrix(y_test, y_test_pred)
print("Precision Score: \t {0:.4f}".format(precision_score(y_test, y_test_pred, average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test, y_test_pred, average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test, y_test_pred, average='weighted')))
