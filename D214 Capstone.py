#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import statistics
from scipy import stats
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"E:\Users\laisu\Downloads\kc_house_data.csv",dtype={'locationid':np.int64})


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['price'] = df['price'].astype(np.int64)
df['bathrooms'] = df['bathrooms'].astype(np.int64)
df['floors'] = df['floors'].astype(np.int64)
df['lat'] = df['lat'].astype(np.int64)
df['long'] = df['long'].astype(np.int64)


# In[6]:


df['date'] = df['date'].str.strip('0')
df['date'] = df['date'].str.strip('T')
df['date'] = df['date'].astype(np.int64)
df['date']


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


missing_percentages = df.isnull().sum() / len(df) * 100
missing_percentages = missing_percentages.sort_values(ascending=False)
print(missing_percentages)


# In[10]:


total_missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
print("Data Sparsity: {:.2f}%".format(total_missing_percentage))


# In[11]:


df.describe()


# In[12]:


df.duplicated()


# In[13]:


df.isnull().sum()


# In[14]:


boxplot=sns.boxplot(x= 'bedrooms',data=df)
df.plot.scatter(x = 'bedrooms', y = 'price')


# In[15]:


boxplot=sns.boxplot(x= 'bathrooms',data=df)
df.plot.scatter(x = 'bathrooms', y = 'price')


# In[16]:


boxplot=sns.boxplot(x= 'sqft_living',data=df)
df.plot.scatter(x = 'sqft_living', y = 'price')


# In[17]:


boxplot=sns.boxplot(x= 'sqft_lot',data=df)
df.plot.scatter(x = 'sqft_lot', y = 'price')


# In[18]:


boxplot=sns.boxplot(x= 'floors',data=df)
df.plot.scatter(x = 'floors', y = 'price')


# In[19]:


boxplot=sns.boxplot(x= 'sqft_above',data=df)
df.plot.scatter(x = 'sqft_above', y = 'price')


# In[20]:


df.boxplot(figsize=(15, 8))
plt.title("Box Plots of Variables")
plt.xticks(rotation=90)
plt.show()


# In[21]:


#drop unnecessary variables
df = df.drop(columns=['id', 'long', 'lat'])


# In[22]:


df.info()


# In[23]:


corr = df.corr().abs()
fig, ax=plt.subplots(figsize=(17,12))
fig.suptitle('Variable Correlations', fontsize=30, y=.95)
heatmap = sns.heatmap(corr, cmap='Reds', annot=True)


# In[24]:


df['price'] = np.log(df['price'])


# In[25]:


from scipy.stats import kstest, norm

np.random.seed(123)
sample = np.random.normal(loc=0, scale=1, size=100)
ks_stat, p_value = kstest(sample, norm.cdf)
print("Kolmogorov-Smirnov test:")
print("KS statistic:", ks_stat)
print("p-value:", p_value)


# In[26]:


df.columns


# In[27]:


y = df['price']
X = df[['date', 'bedrooms', 'bathrooms', 'sqft_living',
        'view', 'grade', 'sqft_above',
       'sqft_basement', 'sqft_living15',
          ]].assign(const=1)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# In[28]:


#checking for multicollinearity using VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[['date', 'bedrooms', 'bathrooms', 'sqft_living',
        'view', 'grade', 'sqft_above',
       'sqft_basement', 'sqft_living15',]].assign(const=1)
vif_df = pd.DataFrame()
vif_df["feature"] = X.columns

vif_df["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]

print(vif_df)


# In[29]:


#Removing variables with high VIF
y = df['price']
X = df[['bedrooms', 'bathrooms',
        'view', 'grade',
        'sqft_living15']].assign(const=1)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# In[30]:


model = sm.OLS(y, sm.add_constant(X)).fit()
y_pred = model.predict(sm.add_constant(X))
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[31]:


fig_qq = sm.graphics.qqplot(model.resid, line='45', fit=True,)
fig_qq.suptitle('QQ plot for residual normality check')


# In[32]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
scores = cross_val_score(reg, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
train, test = train_test_split(df,test_size=.25,shuffle=True)


# In[34]:


df_train = pd.DataFrame(X_train, y_train)
df_test = pd.DataFrame(X_test, y_test)


# In[36]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[37]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[38]:


train_mse = np.mean(y_train-y_pred_train)**2
test_mse = np.mean(y_test-y_pred_test)**2
print('Mean Squared Error Train:', train_mse)
print('Mean Squared Error Test:', test_mse)


# In[39]:


df.to_csv (r"E:\Users\laisu\Documents\data2142.csv")

