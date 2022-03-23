#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[3]:


df_train.shape


# In[4]:


df_test.shape


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# # Univariant Analysis

# In[7]:


df_train['Loan_Status'].value_counts() 


# In[8]:


df_train['Loan_Status'].value_counts().plot.bar()


# In[9]:


df_train['Loan_Status'].value_counts(normalize=True).plot.bar()
# normalize = True will give the probability in y-axis

plt.title("Loan Status")


# In[10]:


#Plots for Independent Categorical Variables
plt.figure()
plt.subplot(321)
df_train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Gender')

plt.subplot(322)
df_train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Married')

plt.subplot(323)
df_train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Education')

plt.subplot(324)
df_train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Self-Employed')

plt.subplot(325)
df_train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Credit_History')


# In[11]:


#Plots for Independent Ordinal Variables
plt.figure()
plt.subplot(121)
df_train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Dependents')

plt.subplot(122)
df_train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Property Area')


# In[12]:


#Plots for Independent Numerical Variables
plt.subplot(121)
sns.distplot(df_train['ApplicantIncome'])
plt.subplot(122)
df_train['ApplicantIncome'].plot.box(figsize=(20,5))


# In[13]:


df_train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")


# In[14]:


plt.subplot(121)
sns.distplot(df_train['CoapplicantIncome'])
plt.subplot(122)
df_train['CoapplicantIncome'].plot.box(figsize=(20,5))


# In[15]:


df=df_train.dropna()
plt.subplot(121)
sns.distplot(df['LoanAmount'])

plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(20,5))


# # Bivariant Analysis

# In[16]:


#Frequency Table for Gender and Loan Status
Gender=pd.crosstab(df_train['Gender'],df_train['Loan_Status']) 
Gender


# In[17]:


Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[18]:


#Frequency Table for Married and Loan Status
Married=pd.crosstab(df_train['Married'],df_train['Loan_Status']) 
Married


# In[19]:


Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[20]:


#Frequency Table for Dependents and Loan Status
Dependents=pd.crosstab(df_train['Dependents'],df_train['Loan_Status']) 
Dependents


# In[21]:


Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[22]:


#Frequency Table for Education and Loan Status
Education= pd.crosstab(df_train['Education'],df_train['Loan_Status'])
Education


# In[23]:


Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[24]:


#Frequency Table for Self Employed and Loan Status
Self_Employed= pd.crosstab(df_train['Self_Employed'],df_train['Loan_Status'])
Self_Employed


# In[25]:


Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[26]:


#Frequency Table for Credit History and Loan Status
Credit_History= pd.crosstab(df_train['Credit_History'],df_train['Loan_Status'])
Credit_History


# In[27]:


Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))


# In[28]:


#Frequency Table for Property Area and Loan Status
Property_Area=pd.crosstab(df_train['Property_Area'],df_train['Loan_Status'])
Property_Area


# In[29]:


Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True, figsize=(4,4))


# In[30]:


df_train.Loan_Status=df_train.Loan_Status.map({'Y':1,'N':0})
df_train['Dependents'].replace('3+', 3,inplace=True) 
df_test['Dependents'].replace('3+', 3,inplace=True) 


# In[31]:


Loan_status=df_train.Loan_Status
df_train.drop('Loan_Status',axis=1,inplace=True)
Loan_ID=df_test.Loan_ID
dataset=pd.concat([df_train,df_test])
dataset.head()


# In[32]:


dataset.describe()


# In[33]:


dataset.isnull().sum()


# In[34]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[35]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[36]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[37]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[38]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[39]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[40]:


dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(),inplace=True)


# In[41]:


dataset['LoanAmount_log'] = np.log(dataset['LoanAmount']) 
dataset['LoanAmount_log'].hist(bins=20) 


# In[42]:


dataset=dataset.drop('Loan_ID',axis=1)
dataset.head()


# In[43]:


dataset=dataset.drop('Gender',axis=1)


# In[44]:


dataset=dataset.drop('Dependents',axis=1)


# In[45]:


dataset=pd.get_dummies(dataset) 
dataset.head()


# In[46]:


train_X=dataset.iloc[:614,]
train_y=Loan_status
X_test=dataset.iloc[614:,]


# # Classification Algorithms

# In[47]:


#Applying Classification Models
from sklearn.model_selection import train_test_split
train_X,x_cv,train_y,y_cv=train_test_split(train_X,train_y,random_state=0)


# # Logistic Regression

# In[48]:


#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(train_X, train_y)


# In[49]:


pred_cv = model.predict(x_cv)


# In[50]:


pred_cv


# In[51]:


# Measuring Accuracy
print('The accuracy of Logistic Regression is: ', accuracy_score(pred_cv,y_cv))


# In[52]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[54]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# # Decesion Tree

# In[55]:


#Applying Decesion Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train_X, train_y)


# In[56]:


pred_cv = model.predict(x_cv)


# In[57]:


pred_cv


# In[58]:


# Measuring Accuracy
print('The accuracy of Decision Tree Classifier is: ', accuracy_score(pred_cv,y_cv))


# In[59]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[61]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# # Random Forest

# In[62]:


#Applying Random Forest
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier()
model.fit(train_X, train_y)


# In[63]:


pred_cv = model.predict(x_cv)


# In[64]:


pred_cv


# In[65]:


print('The accuracy of Random Forest Classification is: ', accuracy_score(pred_cv,y_cv))


# In[66]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[67]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[68]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# # K-NN

# In[69]:


#Applying K-NN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
model.fit(train_X, train_y)


# In[70]:


pred_cv = model.predict(x_cv)


# In[71]:


pred_cv


# In[72]:


print('The accuracy of KNN is: ', accuracy_score(pred_cv,y_cv))


# In[73]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[74]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[75]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# # SVM

# In[76]:


#Applying SVM
from sklearn.svm import SVC
model = SVC(kernel='linear',C=1.0,gamma='scale',shrinking=False)
model.fit(train_X, train_y)


# In[77]:


pred_cv = model.predict(x_cv)


# In[78]:


pred_cv


# In[79]:


print('The accuracy of SVM is: ', accuracy_score(pred_cv,y_cv))


# In[80]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[81]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[82]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# # Naive Bayes

# In[83]:


#Applying Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_X, train_y)


# In[84]:


pred_cv = model.predict(x_cv)


# In[85]:


pred_cv


# In[86]:


print('The accuracy of Naive Bayes is: ', accuracy_score(pred_cv,y_cv))


# In[87]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[88]:


from sklearn.metrics import classification_report
print(classification_report(y_cv,pred_cv))


# In[89]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
print ('R Squared =',r2_score(y_cv,pred_cv))
print ('MAE =',mean_absolute_error(y_cv,pred_cv))
print ('MSE =',mean_squared_error(y_cv,pred_cv))
print ('MAPE =',mean_absolute_percentage_error(y_cv,pred_cv))


# In[90]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=LogisticRegression()
svc.fit(train_X,train_y)
pred=svc.predict(x_cv)
print(accuracy_score(y_cv,pred))
print(confusion_matrix(y_cv,pred))
print(classification_report(y_cv,pred))


# In[91]:


df_output=pd.DataFrame()


# In[92]:


outp=svc.predict(X_test).astype(int)
outp


# In[93]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[94]:


df_output.head()


# In[95]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\User\Downloads\Loan Eligiblity Prediction\output.csv',index=False)


# # Results:

# The accuracy of Logistic Regression is:  83.76 %
# 
# The accuracy of KNN is:  63.63 %
# 
# The accuracy of SVM is:  83.11 %
# 
# The accuracy of Naive Bayes is:  82.46 %
# 
# The accuracy of Decision Tree Classifier is:  72.07 %
# 
# The accuracy of Random Forest Classification is:  81.16 %
