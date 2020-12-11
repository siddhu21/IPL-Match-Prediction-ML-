#Importing Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Passing CSV into matches
matches=pd.read_excel('C:/Users/nsidd/OneDrive/Desktop/IPL Datasets/matches_2020.xlsx',sep=",")
matches.head()

matches.isnull().sum()


#Dropping Some Unuseful Columns
matches=matches.drop(['id','date','umpire1','umpire2','umpire3','player_of_match','venue','result','dl_applied','win_by_runs','win_by_wickets','season','city','Avg 1st Innings Score','Avg 2nd Innings Score'],axis=1) 

matches.head()

# Renaming Columns
matches = matches.rename(columns = {'team1': 'A', 'team2': 'B'}, inplace = False)

matches.head()

# team A=0 team B=1
matches.isnull().sum()

matches['winner'].value_counts()

matches.dtypes

df=matches[['Pplay T1','pplay twick1','Pplay T2','pplay twick2','toss_winner','toss_decision']]

sns.pairplot(df)


# if Team A is winner then it encoded 0 else 1
matches.winner[matches.winner == matches.A] = 0
matches.winner[matches.winner == matches.B] = 1


#Encoding Team A Names
matches.A[matches.A == "Chennai Super Kings"] = 0 
matches.A[matches.A == "Delhi Capitals"] = 1
matches.A[matches.A == "Deccan Chargers"] = 2
matches.A[matches.A == "Gujarat Lions"] = 3
matches.A[matches.A == "Kolkata Knight Riders"] = 4
matches.A[matches.A == "Kochi Tuskers Kerala"] = 5
matches.A[matches.A == "Kings XI Punjab"] = 6
matches.A[matches.A == "Mumbai Indians"] = 7
matches.A[matches.A == "Pune Warriors"] = 8
matches.A[matches.A == "Royal Challengers Bangalore"] = 9
matches.A[matches.A == "Rising Pune Supergiant"] = 10
matches.A[matches.A == "Rajasthan Royals"] = 11
matches.A[matches.A == "Sunrisers Hyderabad"] = 12
matches.A[matches.A == "Delhi Daredevils"] = 13


#Encoding Team B Names
matches.B[matches.B == "Chennai Super Kings"] = 0 
matches.B[matches.B == "Delhi Capitals"] = 1
matches.B[matches.B == "Deccan Chargers"] = 2
matches.B[matches.B == "Gujarat Lions"] = 3
matches.B[matches.B == "Kolkata Knight Riders"] = 4
matches.B[matches.B == "Kochi Tuskers Kerala"] = 5
matches.B[matches.B == "Kings XI Punjab"] = 6
matches.B[matches.B == "Mumbai Indians"] = 7
matches.B[matches.B == "Pune Warriors"] = 8
matches.B[matches.B == "Royal Challengers Bangalore"] = 9
matches.B[matches.B == "Rising Pune Supergiant"] = 10
matches.B[matches.B == "Rajasthan Royals"] = 11
matches.B[matches.B == "Sunrisers Hyderabad"] = 12
matches.B[matches.B == "Delhi Daredevils"] = 13


matches.head()

#Encoding Pitch Type
matches['Pitch Type'][matches['Pitch Type'] == "Batting"] = 0 
matches['Pitch Type'][matches['Pitch Type']== "Both"] = 1
matches['Pitch Type'][matches['Pitch Type'] == "Batting & Spinner Friendly"] = 2
matches['Pitch Type'][matches['Pitch Type'] == "Bowling"] = 3

#Replacing bowl to field in toss_decision column..
matches['toss_decision']=matches['toss_decision'].str.replace('bowl','field')

matches['toss_decision'].value_counts()

#Encoding Toss Decision
matches['toss_decision'][matches['toss_decision'] == "field"] = 0 
matches['toss_decision'][matches['toss_decision'] == "bat"] = 1

#Encoding Toss winner Names
matches['toss_winner'][matches['toss_winner'] == "Chennai Super Kings"] = 0 
matches['toss_winner'][matches['toss_winner'] == "Delhi Capitals"] = 1
matches['toss_winner'][matches['toss_winner'] == "Deccan Chargers"] = 2
matches['toss_winner'][matches['toss_winner'] == "Gujarat Lions"] = 3
matches['toss_winner'][matches['toss_winner'] == "Kolkata Knight Riders"] = 4
matches['toss_winner'][matches['toss_winner'] == "Kochi Tuskers Kerala"] = 5
matches['toss_winner'][matches['toss_winner'] == "Kings XI Punjab"] = 6
matches['toss_winner'][matches['toss_winner']== "Mumbai Indians"] = 7
matches['toss_winner'][matches['toss_winner'] == "Pune Warriors"] = 8
matches['toss_winner'][matches['toss_winner']== "Royal Challengers Bangalore"] = 9
matches['toss_winner'][matches['toss_winner'] == "Rising Pune Supergiant"] = 10
matches['toss_winner'][matches['toss_winner'] == "Rajasthan Royals"] = 11
matches['toss_winner'][matches['toss_winner']== "Sunrisers Hyderabad"] = 12
matches['toss_winner'][matches['toss_winner'] == "Delhi Daredevils"] = 13

#Converting Object types to Int

matches['A']=matches['A'].astype('int64')
matches['B']=matches['B'].astype('int64')
matches['winner']=matches['winner'].astype('int64')
matches['Pitch Type']=matches['Pitch Type'].astype('int64')
matches['toss_winner']=matches['toss_winner'].astype('int64')
matches['toss_decision']=matches['toss_decision'].astype('int64')


#Model 1:-

y = matches.winner.values  #Adding Winner Column into y
X = matches.drop(columns=["winner"], axis=1).values # features column

#Test Size of 20%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Data Preprocessing
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

# Passing Logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predicting on Test Set
y_pred1=classifier.predict(X_test)

#Importing Accuracy Score from Metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred1)
print(accuracy)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)
cm1


# Model 2:-

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred2= classifier.predict(X_test)

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred2)
print(accuracy)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmm = confusion_matrix(y_test, y_pred2)
print(cmm)



#So, from Random Forrest Algorithm we achieved an accuracy of 64% on test data

#Model 3:- ( Decision Tree)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dtree.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred4)
print(cm2)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred4)
print(accuracy)

# Dtree also has a bad accuracy score.

#model 4:- (Gradient Boosting)

#Test Size of 30%

state=20

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=state)

#Passing different learning rates to find best learning_rate.

from sklearn.ensemble import GradientBoostingClassifier

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
	
from sklearn.metrics import classification_report, confusion_matrix	
	
gb_clf2 = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

#Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

#Classification Report
print("Classification Report")
print(classification_report(y_test, predictions))

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

### Accuracy obtained 72% in Gradient Boosting Algorithm

#############

#Model 5:- (XGBoost)

#using test_size 30%

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=state)


##


from xgboost import XGBClassifier
xgb=XGBClassifier(learning_rate =0.01,
              n_estimators=1000,
              max_depth=10,
              min_child_weight=4,
              gamma=0.1,
              subsample=0.3,
              colsample_bytree=0.8,
              objective= 'binary:logistic',
              nthread=4,
              scale_pos_weight=3,
              seed=27, 
              reg_alpha=0.001,
              
              n_jobs=4,
              )
xgb.fit(X_train,y_train)

y_pred6=xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred6)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xg = confusion_matrix(y_test, y_pred6)
print(cm_xg)

#with learning_rate of 0.01 and some other parameteres we achieved a better accuracy score compared to other models.
