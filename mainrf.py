import numpy as np                              
import pandas as pd     
match = pd.read_csv('ipl.csv')  
match = match.drop(['mid', 'date', 'venue'], axis = 1)   
y = match['total']                                      
y.head(10)       
X = match.drop('total',axis=1) 
print(X.info)
#printing null values
print("checking for null values")
print(X.isnull().sum().sum())

from sklearn.preprocessing import LabelEncoder  
labeled = LabelEncoder()         

X['bat_team'] = labeled.fit_transform(X['bat_team'])
X['bat_team'].head(10)    

X['bowl_team'] = labeled.fit_transform(X['bowl_team'])
X['bowl_team'].head(10)

X['bowler'] = labeled.fit_transform(X['bowler'])
X['bowler'].head(10)
X['batsman'] = labeled.fit_transform(X['batsman'])
X['batsman'].head(10)

from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

from sklearn.preprocessing import StandardScaler    

scaler = StandardScaler()  

X_train = scaler.fit_transform(X_train)        
X_test = scaler.transform(X_test)   


from sklearn.ensemble import RandomForestRegressor      
model=RandomForestRegressor()      
model.fit(X_train,y_train) 
print("Accuracy :",model.score(X_test,y_test))
X_dataframe = X_train.tolist() 
X_dataframe = pd.DataFrame(X_train)
feature_important = model.feature_importances_
print("feature_important :",feature_important)
print(feature_important)
total = sum(feature_important)
new = [value * 100 / total for value in feature_important]
new = np.round(new,2)
keys = list(X_dataframe.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
print(feature_importances)
