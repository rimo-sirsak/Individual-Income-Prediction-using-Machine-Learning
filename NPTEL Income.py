import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

data=pd.read_csv('income.csv',na_values=(' ?'))
df=data.copy()
df.info()
df.isnull().sum()
df.describe()
# Categorical data describe
summary_cat=df.describe(include='O')
df['JobType'].unique()
missing=df[df.isnull().any(axis=1)]
df=df.dropna(axis=0)
correlation=df.corr()
gender_salsat=pd.crosstab(index=df['gender'],columns=df['SalStat'],\
                          margins=True, normalize='index')

sns.countplot(df['SalStat'])
sns.distplot(df['?age'],bins=10,kde=False)
sns.boxplot('SalStat','?age',data=df)
df.groupby('SalStat')['?age'].median()

# calculate interquartile range
q25, q75 = np.percentile(df['?age'], 25), np.percentile(df['?age'], 75)
iqr = q75 - q25
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in df['?age'] if x < lower or x > upper]

sns.countplot(y=df['JobType'],hue=df['SalStat'])
pd.crosstab(index=df['JobType'],columns=df['SalStat'],normalize='index').round(4)*100

sns.countplot(y=df['EdType'],hue=df['SalStat'])
pd.crosstab(index=df['EdType'],columns=df['SalStat'],normalize='index').round(4)*100

sns.countplot(y=df['occupation'],hue=df['SalStat'])
pd.crosstab(index=df['occupation'],columns=df['SalStat'],normalize='index').round(4)*100

sns.distplot(df['capitalgain'])
df['SalStat']=df['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})

df=pd.get_dummies(df,drop_first=(True))
df=df.rename(columns={'?age':'Age'})

column=df.columns
features=list(set(column)-set(['SalStat']))
Y=df['SalStat'].values
X=df[features].values

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=9)
X_train_scale=StandardScaler().fit_transform(X_train)
X_test_scale=StandardScaler().fit_transform(X_test)
#for c in this_c:
    
#Logistic Regression Model
model=LogisticRegression().fit(X_train,y_train)
model.coef_
model.score(X_test,y_test)

y_pred=model.predict(X_test)
score_logistic_train=model.score(X_train,y_train)
score_logistic_test=model.score(X_test,y_test)
confusion_matrix=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score( y_test, y_pred)
print(accuracy)
f1_logistic=f1_score(y_test,y_pred)  #0.65
print(f1_logistic)



#KNN Model:
X_train_scale=StandardScaler().fit_transform(X_train)
X_test_scale=StandardScaler().fit_transform(X_test)
KNN_model=KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
score_KNN_train=KNN_model.score(X_train,y_train)
score_KNN_test=KNN_model.score(X_test,y_test)
y_pred=KNN_model.predict(X_test)
confusion_matrix_KNN=confusion_matrix(y_test,y_pred)
accuracy_KNN=accuracy_score( y_test, y_pred)
print(accuracy_KNN)
f1_KNN=f1_score(y_test,y_pred)  #0.66


#SVM model:
# We will use the scaled data here
SVM_model=SVC(kernel='rbf',C=2,gamma='auto').fit(X_train_scale,y_train)
score_SVM_train=SVM_model.score(X_train_scale,y_train)
score_SVM_test=SVM_model.score(X_test_scale,y_test)
y_pred=SVM_model.predict(X_test_scale)
confusion_matrix_SVM=confusion_matrix(y_test,y_pred)
accuracy_SVM=accuracy_score( y_test, y_pred)
print(accuracy_KNN)
f1_SVM=f1_score(y_test,y_pred) # 0.64


# RandomForest Classifier
Random_model=RandomForestClassifier(n_estimators=40, max_features=10,random_state=(10)).fit(X_train,y_train)
score_RF_train=Random_model.score(X_train,y_train)
score_RF_test=Random_model.score(X_test,y_test)
y_pred=Random_model.predict(X_test)
accuracy_RF=accuracy_score( y_test, y_pred)
f1_RF=f1_score(y_test,y_pred) # 0.66
Random_confusion=confusion_matrix(y_test,y_pred)

#NaiveBayes Classifier
NV_model=GaussianNB().fit(X_train,y_train)
score_NV_train=NV_model.score(X_train,y_train)
score_NV_test=NV_model.score(X_test,y_test)
y_pred=NV_model.predict(X_test)
accuracy_NV=accuracy_score( y_test, y_pred)
f1_NV=f1_score(y_test,y_pred) # 0.659



#Gradient Boosting classifier
GB_model=GradientBoostingClassifier(n_estimators=40,learning_rate=0.5,random_state=(15)).fit(X_train,y_train)
score_GB_train=GB_model.score(X_train,y_train)
score_Gb_test=GB_model.score(X_test,y_test)
y_pred=GB_model.predict(X_test)
accuracy_GB=accuracy_score( y_test, y_pred)
f1_RF=f1_score(y_test,y_pred) # 0.659

#Based on f1 score we can conclude that Random Forest is the best model to fit.

clf=GridSearchCV(RandomForestClassifier(random_state=0),{'n_estimators':[5,10,15,20,30,40],
                'max_features':[5,10,15,20,30,50],
                'max_depth':[1,5,10,20,30]},cv=10,return_train_score=(False),scoring=('f1'))
clf.fit(X_train,y_train)
result_RF=pd.DataFrame(clf.cv_results_)
result_RF=result_RF[['param_n_estimators','param_max_features','param_max_depth','mean_test_score']] 
clf.best_score_
clf.best_params_

#We will Use the best Params to get the best Result
Random_model=RandomForestClassifier(n_estimators=40, max_features=30, max_depth=(20),random_state=(0)).fit(X_train,y_train)
score_RF_train=Random_model.score(X_train,y_train)
score_RF_test=Random_model.score(X_test,y_test)
y_pred=Random_model.predict(X_test)
accuracy_RF=accuracy_score( y_test, y_pred)
f1_RF=f1_score(y_test,y_pred) # 0.66
Random_confusion=confusion_matrix(y_test,y_pred)



#We saw many misclassification by confusion matrix
# We will check Feature Scores
df=data.copy()
df=df.dropna(axis=0)
df=df.rename(columns={'?age':'Age'})


sns.countplot(y=df['JobType'],hue=df['SalStat'])
pd.crosstab(index=df['JobType'],columns=df['SalStat'],normalize='index').round(4)*100
      
sns.countplot(y=df['gender'],hue=df['SalStat'])
pd.crosstab(index=df['gender'],columns=df['SalStat'],normalize='index').round(4)*100

sns.countplot(y=df['race'],hue=df['SalStat'])
pd.crosstab(index=df['race'],columns=df['SalStat'],normalize='index').round(4)*100

df=df.drop(['race'],axis=1)
column=df.columns


def prepare_targets(y_train,y_test):
    le=LabelEncoder()
    le.fit(y_train)
    y_train_enc=le.transform(y_train)
    y_test_enc=le.transform(y_test)
    return y_train_enc,y_test_enc

features=list(set(column)-set(['SalStat']))
Y=df['SalStat']
X=df[features]

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=9)

oe=OrdinalEncoder()
oe.fit(X_train[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']])
X_train[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']]=oe.transform(X_train[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']])

oe.fit(X_test[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']])
X_test[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']]=oe.transform(X_test[['Age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'gender','nativecountry']])
                                                                       
y_train,y_test=prepare_targets(y_train, y_test)  

fs=SelectKBest(score_func=chi2,k='all')
fs.fit(X_train,y_train)
X_train_fs_chi=fs.transform(X_train)
X_test_fs_chi=fs.transform(X_test)                                                                

for i in range(len(fs.scores_)):
    print('Feature %d: %f'%(i,fs.scores_[i]))
    
# We Saw that Native country and JobType has very low feature value. We will check by another
# function namely mutual-info-classif

fs=SelectKBest(score_func=mutual_info_classif,k='all')
fs.fit(X_train,y_train)
X_train_fs_mutual=fs.transform(X_train)
X_test_fs_mutual=fs.transform(X_test)                                                                

for i in range(len(fs.scores_)):
    print('Feature %d: %f'%(i,fs.scores_[i]))

#We Saw almost Similar Result in both the cases

#We will try to drop random combination
col=['nativecountry','gender','JobType']
X_train_try1=X_train.drop(col,axis=1)
X_test_try1=X_test.drop(col,axis=1)

Random_model=RandomForestClassifier(n_estimators=40, max_features=2, max_depth=(20),random_state=(0)).fit(X_train_try1,y_train)
score_RF_train=Random_model.score(X_train_try1,y_train)
score_RF_test=Random_model.score(X_test_try1,y_test)
y_pred=Random_model.predict(X_test_try1)
accuracy_RF=accuracy_score( y_test, y_pred)
f1_RF=f1_score(y_test,y_pred) # 0.90
Random_confusion=confusion_matrix(y_test,y_pred)

#We got a Higher F1 score which was our requirement
