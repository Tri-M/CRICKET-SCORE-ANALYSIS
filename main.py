import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from turtle import color
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


df=pd.read_csv('ipl.csv')
# print(sns.pairplot(df))
df.head()
x=df.drop(columns=["date","total"],axis=1)
y=df["total"]
df.isnull().sum()
print(df.describe())
# print(df.columns)
dropcols=["batsman","bowler","striker","non-striker","mid"]
df.drop(columns=dropcols, inplace=True, axis=1)
print(sns.pairplot(df,height=1.5))
plt.show()
main_teams= ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

print(df.groupby(["bat_team"])["total"].sum().sort_values(ascending=True))
print(df.groupby(["bowl_team"])["total"].sum().sort_values(ascending=True))
print(df.shape)

df.reset_index(drop=True, inplace=True)
print(df.shape)
dups=df.iloc[:,1:4]
dups.bat_team.unique()

OE=OrdinalEncoder(categories=[['De Beers Diamond Oval',                              
'Subrata Roy Sahara Stadium' ,                              
'Buffalo Park','OUTsurance Oval',                                             
'Holkar Cricket Stadium',                                  
'Barabati Stadium',                                        
'Newlands',                                                 
'Maharashtra Cricket Association Stadium',                  
'Shaheed Veer Narayan Singh International Stadium',         
'New Wanderers Stadium',                                    
'Dr DY Patil Sports Academy',                               
'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',      
'JSCA International Stadium Complex',                       
'Sharjah Cricket Stadium',                                 
"St George's Park",                                         
'Sheikh Zayed Stadium',                                     
'Dubai International Cricket Stadium',                      
'SuperSport Park' ,                                         
'Himachal Pradesh Cricket Association Stadium',            
'Punjab Cricket Association IS Bindra Stadium, Mohali' ,   
'Kingsmead',                                               
'Sardar Patel Stadium, Motera',                            
'Brabourne Stadium',                                       
'Rajiv Gandhi International Stadium, Uppal',               
'Sawai Mansingh Stadium',                              
'Punjab Cricket Association Stadium, Mohali' ,'MA Chidambaram Stadium, Chepauk',         
'Feroz Shah Kotla' ,                                      
'Eden Gardens',                                           
'Wankhede Stadium',                                       
'M Chinnaswamy Stadium'],["Sunrisers Hyderabad", 
'Delhi Daredevils',              
'Rajasthan Royals',               
'Kolkata Knight Riders',       
'Royal Challengers Bangalore',    
'Chennai Super Kings',            
'Kings XI Punjab',                 
'Mumbai Indians'],["Sunrisers Hyderabad",          
'Chennai Super Kings',            
'Kings XI Punjab',               
'Rajasthan Royals',                
'Mumbai Indians',                 
'Royal Challengers Bangalore',    
'Kolkata Knight Riders',          
'Delhi Daredevils']],handle_unknown='use_encoded_value', unknown_value=-1)

print(OE.fit(dups))
dups=OE.fit_transform(dups)
ordinalE=pd.DataFrame(dups,columns=["venue","bat_team","bowl_team"])
print(ordinalE.shape)
print(df.shape)
df.date=pd.to_datetime(df.date)
df.drop(columns=["venue","bat_team","bowl_team"], axis=1, inplace=True)
df["venue"]=ordinalE["venue"]
df["bat_team"]=ordinalE["bat_team"]
df["bowl_team"]=ordinalE["bowl_team"]
X_train=df.drop(columns=["total"], axis=1)[df.date.dt.year<=2016]
X_test=df.drop(columns=["total"], axis=1)[df.date.dt.year>=2017]
y_train=df.total[df.date.dt.year<=2016].values
y_test=df.total[df.date.dt.year>=2017].values
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred=regressor.predict(X_test)
plt.scatter(y_test, pred, color="blue")
z=np.polyfit(y_test, pred,1)
p=np.poly1d(z)
plt.plot(y_test, p(y_test),"r--")
plt.show()
cv_results=cross_val_score(regressor,X_test,y_test)
cv_results.mean()
pred[:10]
y_test[:10]
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
sns.displot(y_test-pred)
ridge=Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test)
plt.scatter(y_test, ridge_pred, color="red")
z=np.polyfit(y_test, ridge_pred,1)
p=np.poly1d(z)
plt.plot(y_test, p(y_test),"r--")
plt.show()

ridge.score(X_test, y_test)
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5,error_score="raise")
ridge_regressor.fit(X_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
best_model=ridge_regressor.best_estimator_
best_model.fit(X_train, y_train)
prediction=best_model.predict(X_test)
best_model.score(X_test,y_test)
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#lasso regression
lasso=Lasso()

# parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
parameters={'alpha':[0.0001,0.001,0.01,0.1,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
