#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


# loading the data 
titanic_data=pd.read_csv(r"C:\Users\Karanja\Downloads\titanic\train.csv")
# printing the first 5 rows of the data
titanic_data.head()


# In[7]:


# number of rows and columns
titanic_data.shape


# In[8]:


#information about the data 
titanic_data.info()


# In[9]:


#checking the number of missing values 
titanic_data.isnull().sum()


# In[11]:


#handling the null values 
titanic_data = titanic_data.drop(columns = 'Cabin', axis=1)


# In[12]:


# replacing the missing value in age value with mean value 
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[13]:


#finding the mode value of embarked column
print(titanic_data['Embarked'].mode())


# In[15]:


# replacing the mising value in the embarked value with the mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True )


# In[16]:


titanic_data.isnull().sum()


# In[17]:


# DATA ANALYSIS
# getting statistical measures about the data 
titanic_data.describe()


# In[18]:


# finding the number of pple survived and did not 
titanic_data['Survived'].value_counts()


# In[22]:


# making a count plot of survived column
sns.countplot(x='Survived', data=titanic_data)


# In[27]:


# finding the number of pple by gender 
titanic_data['Sex'].value_counts()


# In[28]:


# making a count plot of sex column
sns.countplot(x='Sex', data=titanic_data)


# In[30]:


# number of survivors genderwise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)


# In[31]:


# making a count plot of pclass column
sns.countplot(x='Pclass', data=titanic_data)


# In[32]:


# number of survivors pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# In[33]:


#encoding the categorical columns 
titanic_data['Sex'].value_counts()


# In[34]:


titanic_data['Embarked'].value_counts()


# In[37]:


#converting gategorical columns
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)


# In[38]:


titanic_data.head()


# In[40]:


#SEPARATING FEATURES & FEATURES
X=titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'], axis=1)
Y=titanic_data['Survived']


# In[41]:


print(X)


# In[42]:


print(Y)


# In[43]:


#spliting the data into training and testing data
X_train, X_test,Y_train, Y_test =train_test_split(X,Y, test_size=0.2,random_state=2)


# In[44]:


print(X.shape,X_train.shape,X_test.shape)


# In[45]:


#model training 
model = LogisticRegression()


# In[46]:


# training the model with the training data 
model.fit(X_train, Y_train)


# In[47]:


# model evaluation
X_train_prediction = model.predict(X_train)


# In[48]:


print(X_train_prediction )


# In[49]:


training_data_accuracy= accuracy_score(Y_train, X_train_prediction )


# In[50]:


print('Accuracy score of the training data is : ',training_data_accuracy )


# In[51]:


X_test_prediction = model.predict(X_test)


# In[52]:


print(X_test_prediction)


# In[54]:


test_data_accuracy= accuracy_score(Y_test, X_test_prediction )
print('Accuracy score of the test data is : ',test_data_accuracy )


# In[ ]:




