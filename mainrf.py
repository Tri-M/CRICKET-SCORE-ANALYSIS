import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv('ipl.csv')
# print(sns.pairplot(df))
df.head()
x=df.drop(columns=["date","total"],axis=1)
y=df["total"]
df.isnull().sum()
from datetime import datetime as dt
df['date']=df['date'].apply(lambda x:dt.strptime(x,'%Y-%m-%d'))

print(df.describe())
final_df=df.drop(['venue'],axis=1)
final_df=pd.get_dummies(final_df)
y_train=final_df[final_df['date'].dt.year <=2016 ]['total']
y_test= final_df[final_df['date'].dt.year >2016 ]['total']
X_train=final_df.drop(['total'],axis=1)[final_df['date'].dt.year <=2016].drop(['date'],axis=1)
X_test=final_df.drop(['total'],axis=1)[final_df['date'].dt.year >2016].drop(['date'],axis=1)
print(X_train.isnull().sum())

regressor=RandomForestRegressor()
n_estimators=[50,100,150,200,250]
max_depth=[5,10,15,20,25,30,35]
max_features=['auto', 'sqrt']
min_samples_split=[2, 5, 10, 15, 100]
min_samples_leaf=[1, 2, 5, 10]
parameters={
     'n_estimators':n_estimators,
     'max_depth':max_depth,
     'min_samples_leaf':min_samples_leaf,
     'min_samples_split':min_samples_split
}

# rf=RandomizedSearchCV(estimator=regressor,param_distributions= parameters,n_iter=2,cv=5,n_jobs=-1)

# rf.fit(X_train, y_train)

# print(" Results from Random Search " )
# print("\n The best estimator across ALL searched params:\n", rf.best_estimator_)
# print("\n The best score across ALL searched params:\n", rf.best_score_)
# print("\n The best parameters across ALL searched params:\n", rf.best_params_)
sns.barplot(x='venue',y='total',data=df)
plt.show()

