import pandas as pd
import numpy as np
from __future__ import division
import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_validate 

from xgboost import XGBRegressor

from scipy.stats import pearsonr

##############################################################################################
# Loading Data
##############################################################################################

df = pd.read_csv("kc_house_data.csv")
df.head(20)
df.dtypes
df.info            
df.describe()

# For missing values
df.isnull().sum() 

################################################################################################
# EDA (Exploratory Data Analysis)
##############################################################################################

features = df.iloc[:, 3:,].columns.tolist()
features

target = df.iloc[:, 2].name
target

correlation = {}

for f in features:
  data_temp = df[[f, target]]
  x1 = data_temp[f].values
  y1 = data_temp[target].values
  key = f + 'VS' + target
  correlation[key] = pearsonr(x1, y1)[0]
  
data_correlation = pd.DataFrame(correlation, index = ['Values']).T
data_correlation.loc[data_correlation['Values'].abs().sort_values(ascending = False).index]

# PLOTTING 2 BEST REGRESSOR JOINTLY

y = df.loc[:, ['sqft_living', 'grade', target]].sort_values(target, ascending = True).values

x = np.arange(y.shape[0])

%matplotlib inline

plt.subplot(3,1,1)
plt.plot(x,y[:,0])
plt.title("Sqrt & Grade VS Price")
plt.ylabel("Sqrt")
plt.subplot(3,1,2)
plt.plot(x,y[:, 1])
plt.ylabel("Grade")
plt.subplot(3,1,3)
plt.plot(x,y[:,2])
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)

sns.barplot(data=df,x='bedrooms',y='price')
sns.boxplot(x='bedrooms',y='price',data=df)

sns.barplot(data=df,x='grade',y='price', palette = 'rocket') # palatte = 'rocket'

# From the above graph it is clear that bedrooms and Grade affects the Price.
# Especially Grade as there is an exponential increase

sns.barplot(data = df, x = 'waterfront', y ='price')
sns.boxplot(x='waterfront',y='price',data=df)
# The houses having waterfront has higher price

sns.barplot(data=df,x='view',y='price',palette='vlag')

sns.countplot(data = df, x = 'waterfront')
# From this we can say that there are less number of waterfrony houses

sns.barplot(data=df,x='sqft_living',y='price')
sns.barplot(data=df,x='sqft_above',y='price')

###########################################################################################
# FEATURE ENGINEERING
###########################################################################################

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

df.head(5)
df.dtypes

plt.figure(figsize=(15,15))
sns.boxplot(x = 'year', y = 'price', data = df)

plt.figure(figsize=(15,15))
sns.boxplot(x = 'month', y = 'price', data = df)

df.groupby('month').mean()['price'].plot()

df = df.drop('id', axis = 1)
df = df.drop('date', axis = 1)
df = df.drop('zipcode', axis = 1)
df.head()

############################################################################################
# Defining X and Y
############################################################################################

X = df.drop('price', axis = 1)
y = df['price']

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train.shape
X_test = scaler.fit_transform(X_test)
X_test.shape

model = []
score = []

############################################################################################
# Making Models
############################################################################################

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_predict = lr.predict(X_test)
print("Score: ",r2_score(lr_predict,y_test))
model.append("Linear Regression")
score.append(r2_score(lr_predict,y_test))


mean_absolute_error(y_test,lr_predict)
mean_squared_error(y_test,lr_predict)
mean_squared_error(y_test,lr_predict)**0.5
explained_variance_score(y_test,lr_predict)


lf = Lasso()
lf.fit(X_train,y_train)
lf_pred = lf.predict(X_test)
print("Score: ",r2_score(lf_pred,y_test))
model.append("Lasso Regression")
score.append(r2_score(lf_pred,y_test))


rf = RandomForestRegressor(n_estimators=100, random_state = 0)
rf.fit(X_train,y_train)
rf_predict = rf.predict(X_test)
print("Score: ",r2_score(rf_predict,y_test))
model.append("Random Forest Regression")
score.append(r2_score(rf_predict,y_test))


xg = XGBRegressor()
xg.fit(X_train,y_train)
xg_predict = xg.predict(X_test)
print("Score: ",r2_score(xg_predict,y_test))
model.append("Xgboost Regression")
score.append(r2_score(xg_predict,y_test))

plt.subplots(figsize=(10, 15))
sns.barplot(y=score,x=model,palette = sns.cubehelix_palette(len(score)))
plt.xlabel("Score")
plt.ylabel("Regression")
plt.title('Regression Score')
plt.show()


############################################################################################
# ANN (Artificial Neural Network)
############################################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train.values,validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)

mean_squared_error(y_test,predictions)

mean_squared_error(y_test,predictions)**0.5

explained_variance_score(y_test,predictions)

# Our predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test,predictions)
# Perfect predictions
plt.plot(y_test,y_test,'r')
