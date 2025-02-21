import pandas as pd
import joblib
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/learnersgalaxy/Downloads/Machine Learning/Exercises for Indigo/Tankering Exercise/Tankering.csv")

df_train, df_test = train_test_split(df, test_size=.3, random_state=2025, stratify=df.Tankering)

clf = tree.DecisionTreeClassifier(max_depth=3)
cols = ['Pax', 'Temperature', 'AirDist', 'FlightTime', 'TripFuel','DepFuelPrice', 'DestFuelPrice']
clf.fit(df_train[cols], df_train.Tankering)
train_predictions = clf.predict(df_train[cols])
test_predictions = clf.predict(df_test[cols])
print(test_predictions)

joblib.dump(clf, 'pred_tank.joblib')

