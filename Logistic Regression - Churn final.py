#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set_style('whitegrid')


# In[16]:


#IMPORT DATASETS
customer = pd.read_csv("/Users/akntw/Desktop/customer_data.csv")
internet = pd.read_csv("/Users/akntw/Desktop/internet_data.csv")
churn = pd.read_csv("/Users/akntw/Desktop/churn_data.csv")


# In[5]:


customer.head()


# In[6]:


#CHECKING DATA TYPES IN DATASET
customer.dtypes


# In[7]:


#DISTRIBUTION OF GENDER
sns.countplot(x='gender',data=customer)


# In[8]:


#WHAT PROPORTION OF OUR CUSTOMERS ARE SENIOR CITIZENS
sns.countplot(x='SeniorCitizen',data=customer)


# In[9]:


#HOW MANY OF OUR CUSTOMERS HAVE DEPENDENTS
sns.countplot(x='Dependents',data=customer)


# In[10]:


#INTERNET DATA SET 
internet.head()


# In[11]:


#CHECKING FOR DATA TYPES
internet.dtypes


# In[12]:


#HOW MANY OF OUR CUSTOMERS HAVE MULTIPLE LINES
sns.countplot(x='MultipleLines',data=internet)


# In[13]:


#DISTRIBUTION OF INTERNET SERVICE USED
sns.countplot(x='InternetService',data=internet)


# In[17]:


#CHURN DATA
churn.head()


# In[18]:


churn.dtypes


# In[19]:


#CONVERT TOTAL CHARGES COLUMN TO FLOAT
churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'],errors='coerce')
churn['TotalCharges'] = churn['TotalCharges'].astype('float')


# In[20]:


#ANALYZE DISTRIBUTION OF TENURE 
sns.distplot(churn['tenure'])


# In[21]:


#ANALYZE CONTRACT TYPE
sns.countplot(x='Contract',data=churn)


# In[22]:


#ANALYSIS OF PAYMENT METHOD
sns.countplot(x='PaymentMethod',data=churn)


# In[23]:


#DISTRIBUTION OF MONTHLY CHARGES 
sns.distplot(churn['MonthlyCharges'],kde=False,bins=30)


# In[24]:


#DISTRIBUTION OF CHURN (TARGET VARIABLE)
sns.countplot(x='Churn',data=churn)


# In[25]:


#ASSESS CORRELINEARITY BETWEEN NUMERICAL VARIABLES
sns.heatmap(churn.corr(),annot=True)


# In[26]:


#DROP TOTAL CHARGES
churn = churn.drop('TotalCharges',axis=1)


# In[27]:


#JOIN ALL DATASETS
df = pd.merge(customer,internet,how='inner',on='customerID')
telecom = pd.merge(df,churn,how='inner',on='customerID')


# In[28]:


#DROP CUSTOMER ID
identity = telecom['customerID']
telecom = telecom.drop('customerID',axis=1)


# In[29]:


#ENCODE CATEGORCIAL VARIABLES
telecom = pd.get_dummies(telecom)


# In[31]:


#DROPPING REDUNDANT COLUMNS
telecom.drop(['gender_Female','Partner_No','Dependents_No','PhoneService_No', 'PaperlessBilling_No','Churn_No'],axis=1,inplace=True)


# In[33]:


#SPLITTING DATA INTO PREDICTOR AND TARGET VARIABLES
y= telecom['Churn_Yes']
X = telecom.drop('Churn_Yes',axis=1)


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


#IMPLEMENTING TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


# In[36]:


#IMPORTING LOGISTIC REGRESSION MODEL
from sklearn.linear_model import LogisticRegression


# In[37]:


logmodel = LogisticRegression()


# In[38]:


#TRAINING THE MODEL
logmodel.fit(X_train,y_train)


# In[39]:


#PREDICTING
predictions = logmodel.predict(X_test)


# In[40]:


#EVALUATING RESULTS 
from sklearn.metrics import classification_report


# In[41]:


#CLASSIFICATION REPORT
print(classification_report(y_test,predictions))


# In[42]:


#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[ ]:




