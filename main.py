import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from turtle import color
from sklearn.preprocessing import OrdinalEncoder

df=pd.read_csv('ipl.csv')
df.head()
x=df.drop(columns=["date","total"],axis=1)
y=df["total"]
df.isnull().sum()
print(df.describe())
# print(df.columns)
dropcols=["batsman","bowler","striker","non-striker","mid"]
df.drop(columns=dropcols, inplace=True, axis=1)
# print(sns.pairplot(df,height=1.5))
# plt.show()
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