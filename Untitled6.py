#!/usr/bin/env python
# coding: utf-8

# # Predictive Analysis for Heart Attack Risk

# For this project we will be taking a look at heart attack risk data and try to use it to build models that can accurately predict when a patient is at risk for a heart attack. The goal is to see if we are able to use the given data set to build an accurate enough model. The features we will use will be shown later in the notebook.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy as sp
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE


# #### Importing and Inspecting

# In[ ]:


df=pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/5561569/9199051/heart_attack_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240820%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240820T013232Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=69f7434f37763f780f8444c120fc852d9c1704ff52b335f0adcaad33953411f2a39b3ae3273c9f5abe48f02eee51a2dc04fed504b71be190544fcff68d687f85e4e52a1eba03e2f8a89e74572adaff00aca929c01589dc9a50846c552fc6ca5ca08545334b0b49523b4b6be31c8066c2dd99e197da2ae62e730e8accf6d12594973094b50982054fc14074ca5a458e3430a4c70d1b88e39ad7122382ccac67f834af178a3d295ce67c1214422843d07298aa080293ccdf61c44228dd04edf59039765ee87128448df5fe5a51c229b4b2ac8332ebc1fa572e4d79a45d634d938813a2ef3ac164f7a6d6c524e64758ad8ab53f48e5fff0164511e1da43107db220')

# Dropping Unwanted Features:
df=df.drop(columns=['Patient ID','Country','Continent','Hemisphere','Income'])
df=df.dropna()
#Checking datatypes to see if anything can be changed (it can!)
df.info()


# ### EDA

# #### Converting 'Blood Pressure' into two new columns: systolic and diastolic

# In[ ]:


df['Systolic'] = None
df['Diastolic'] = None
count=0
for i in df['Blood Pressure']:
        systolic, diastolic = i.split('/')
        df.at[count,'Systolic'] = float(systolic)
        df.at[count,'Diastolic'] = float(diastolic)
        count += 1
df=df.drop(columns='Blood Pressure')


# #### Changing the string values in the data to something easily measureable 

# In[ ]:



df['Diet'] = df['Diet'].astype(str)
df['Sex'] = df['Sex'].astype(str)

Dietmap= {'Healthy' : 0 , 'Average': 1, 'Unhealthy':2}
SM= {'Male':1, 'Female':0}

df['Diet']=df['Diet'].map(Dietmap)
df['Sex']=df['Sex'].map(SM)


# #### Splitting data into testing and training and also scaling 

# In[ ]:



y=df['Heart Attack Risk']
x=df.drop(columns=['Heart Attack Risk'])
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size=0.2, random_state=17, stratify=y)


# #### Features we will be using for our models

# One of the goals of this project is to use these features to try and develope a model that can accurately predict heart attack risk

# In[ ]:


print(x.columns)


# ### Logistic Regression

# In[ ]:


LRM = LogisticRegression(class_weight='balanced')
LRM.fit(xtrain,ytrain)
ypred=LRM.predict(xtest)


print('Model Accuracy:',accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))


# ##### ROC CURVE

# In[ ]:


ypprob = LRM.predict_proba(xtest)[:, 1]
fpr, tpr , na = roc_curve(ytest, ypprob) 
roc = auc(fpr, tpr)
plt.figure()  
plt.plot(fpr, tpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

print('Area Under Curve:',roc)


# As you can see, this is not great. It is just a little better than random guessing.

# ### Random Forest

# In[ ]:


RF=RandomForestClassifier(random_state=17,class_weight='balanced')
RF.fit(xtrain,ytrain)
RFpred=RF.predict(xtest)
print('Model Accuracy:',accuracy_score(ytest,RFpred))
print(confusion_matrix(ytest,RFpred))


# Slightly worse accuracy, though very similar.

# In[ ]:


Rypprob = RF.predict_proba(xtest)[:, 1]
Rfpr, Rtpr , na = roc_curve(ytest, Rypprob) 
Rroc = auc(Rfpr, Rtpr)
plt.figure()  
plt.plot(Rfpr, Rtpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()
print('Area Under Curve:',Rroc)


# Slightly better AUC value compared to the logistic regression model.

# ### Support Vector Machines

# In[ ]:



SVC1=SVC(random_state=17,class_weight='balanced')
SVC1.fit(xtrain, ytrain)

SVCpred=SVC1.predict(xtest)



print('SVC Accuracy:',accuracy_score(ytest,SVCpred))
print(confusion_matrix(ytest,SVCpred))


# ### Gradient Boosting

# In[ ]:


GB=GradientBoostingClassifier(random_state=17)
GB.fit(xtrain,ytrain)
GBpred=GB.predict(xtest)
print('Gradient Boosting Accuracy:',accuracy_score(ytest,GBpred))
print(confusion_matrix(ytest,GBpred))


# ### Retrying Logistic Regression using Feature Selection

# In[ ]:


LRF = LogisticRegression(random_state=17,class_weight='balanced')
bestacc=0
bestauc=0
bestfeaturesAcc=None
numFeaturesAcc=0
bfAUC=None
nfAUC=None
for i in range(1,x.shape[1] + 1):
    rfe = RFE(LRF, n_features_to_select=i) 
    rfe = rfe.fit(xtrain, ytrain)
    features = x.columns[rfe.support_]
    
    xnewtrainfeatures = xtrain[features]
    xnewtestfeatures = xtest[features]
    LRF.fit(xnewtrainfeatures,ytrain)

    
    PRED1=LRF.predict(xnewtestfeatures)
    acc1=accuracy_score(ytest,PRED1)

    if acc1> bestacc:
        bestacc=acc1
        numfeatures = len(features)
        bestfeaturesAcc=features
    LRFprob = LRF.predict_proba(xnewtestfeatures)[:, 1]
    fpr, tpr , na = roc_curve(ytest, LRFprob) 
    roc = auc(fpr, tpr)
    if roc > bestauc:
        bestauc=roc
        bfAUC=features
        nfAUC=len(features)
print('Best Accuracy,Features,and Number of Features',bestacc,bestfeaturesAcc,numFeaturesAcc)
print('     ')
print('Best AUC,Features,and Number of Feature',bestauc,bfAUC,nfAUC)


# Using the best features on a new random forest:

# In[ ]:


RF=RandomForestClassifier(random_state=17,class_weight='balanced')
xnewtrainfeatures = xtrain[bestfeaturesAcc]
xnewtestfeatures = xtest[bestfeaturesAcc]
RF.fit(xnewtrainfeatures,ytrain)
PRED1=RF.predict(xnewtestfeatures)
print(accuracy_score(ytest,PRED1))


# ## Conclusion 

# In this project, we attempted to accurately predict heart attack risk based on certain features from a data set containing health related data. During this, it was discovered that the models we used were not as accurate as one would hope. The most accurate was gradient boosting, while the random forest model had the best ROC curve. Given the AUC of the curve, it implied that the models was slightly better than just randomly guessing. Which was not what we wanted. Our most accurate model, gradient boosting, seems to have major issues with predicting the positive class, which can render the models useless because being able to predict the positive class, in this case and in most cases, is imperative.

# In[ ]:




