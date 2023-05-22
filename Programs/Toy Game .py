#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd


# In[66]:


df = pd.read_csv("E:\ML\DataSets\placement.csv")


# In[67]:


df.head()


# In[68]:


df.info()


# In[69]:


df.shape


# In[70]:


df = df.iloc[:,1:]


# In[71]:


df.head()


# In[72]:


import matplotlib.pyplot as plt


# In[73]:


plt.scatter(df['cgpa'],df['iq'],c=df['placement'])


# In[74]:


X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[75]:


X


# In[76]:


y.shape


# In[77]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


# In[78]:


X_train


# In[79]:


y_train


# In[80]:


X_test


# In[82]:


from sklearn.preprocessing import StandardScaler


# In[83]:


scaler = StandardScaler()


# In[84]:


X_train = scaler.fit_transform(X_train)


# In[31]:


X_train


# In[85]:


X_test = scaler.transform(X_test)


# In[86]:


X_test


# In[87]:


from sklearn.linear_model import LogisticRegression


# In[88]:


clf = LogisticRegression()


# In[89]:


clf.fit(X_train,y_train)


# In[90]:


y_pred = clf.predict(X_test)


# In[92]:


y_test


# In[93]:


from sklearn.metrics import accuracy_score


# In[94]:


accuracy_score(y_test,y_pred)


# In[96]:


from mlxtend.plotting import plot_decision_regions


# In[97]:


plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)


# In[ ]:




