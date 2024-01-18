#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[15]:


movies = pd.read_csv("advertising (1).csv")


# In[25]:


movies.head()


# In[26]:


movies.shape


# In[27]:


movies.describe()


# In[28]:


sns.pairplot(movies, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# In[39]:


movies['TV'].plot.hist(bins=10)
plt.show()


# In[38]:


movies['Radio'].plot.hist(bins=10, color="green", xlabel="Radio")
plt.show()


# In[40]:


movies['Newspaper'].plot.hist(bins=10,color="purple", xlabel="newspaper")
plt.show()


# In[42]:


sns.heatmap(movies.corr(),annot = True)
plt.show()


# In[49]:


x_train, x_test ,y_train ,y_test = train_test_split(movies[['TV']], movies[['Sales']], test_size = 0.3,random_state=0)


# In[50]:


print(x_train)


# In[51]:


print(y_train)


# In[52]:


print(x_test)


# In[53]:


print(y_test)


# In[54]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[55]:


res= model.predict(x_test)
print(res)


# In[56]:


model.coef_


# In[57]:


model.intercept_


# In[59]:


0.05473199* 69.2 + 7.14382225


# In[61]:


plt.plot(res)
plt.show()


# In[63]:


plt.scatter(x_test, y_test)
plt.plot(x_test, 7.14382225 + 0.05473199 * x_test, 'r')
plt.show()


# In[ ]:




