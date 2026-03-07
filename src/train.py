import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Load data
df = pd.read_csv('data/fraud_data.csv')
print(df.head())

#print features name
print(df.columns)

#seperate intput and label feature
X=df.drop('Class',axis=1)
y=df['Class']

#split data into train and test dataset
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#handle imbalanced data
model=RandomForestClassifier(class_weight='balanced')

#train model
model.fit(x_train, y_train)

#Make Predictions
y_pred =model.predict(x_test)
y_pred

#Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


#Save Model
joblib.dump(model, "models/fraud_model.pkl")