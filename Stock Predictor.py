#!/usr/bin/env python
# coding: utf-8

# In[73]:


import yfinance as yf

msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")


# In[2]:


msft.head()


# In[18]:


msft_hist.plot.line(y="Close", use_index=True)


# In[5]:


msft.drop(columns=["Dividends", "Stock Splits"])


# In[19]:


data = msft_hist[["Close"]]
data = data.rename(columns={'Close':'Actual_Close'})
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]


# In[10]:


data.head()


# In[20]:


msft_prev = msft_hist.copy()


# In[21]:


msft_prev = msft_prev.shift(1)


# In[22]:


msft_prev.head()


# In[23]:


predictors = ["Close", "High", "Low", "Open", "Volume"]
data = data.join(msft_prev[predictors]).iloc[1:]


# In[24]:


data.head()


# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[75]:


model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])


# In[27]:


import pandas as pd
from sklearn.metrics import precision_score

pred = model.predict(test[predictors])

pred = pd.Series(pred, index = test.index)


# In[28]:


pred


# In[29]:


precision_score(test["Target"], pred)


# In[76]:


combined = pd.concat({"Target": test["Target"], "Predictions": pred}, axis=1)


# In[33]:


combined


# In[34]:


combined.plot()


# In[77]:


start = 1000
step = 750


def backtest(data, model, predictors, start=1000, step=50):
    predictions = []
    for i in range(start, data.shape[0], step):

        train = data.iloc[0:i].copy()
        test = data.iloc [i: (i+step)].copy() 
        model.fit(train[predictors], train["Target"])
        pred = model.predict_proba(test[predictors])[:,1]
        pred = pd.Series(pred, index=test.index) 
        pred[pred > .6] = 1 
        pred[pred <= .6] = 0 
        combined = pd.concat({"Target": test["Target"], "Predictions": pred}, axis=1) 
        predictions.append(combined)
        
    predictions = pd.concat(predictions)
    return predictions


# In[78]:


predictions["Predictions"].value_counts()


# In[79]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[80]:


weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()

weekly_trend = data.shift(1).rolling(7).mean()["Target"]


# In[81]:


data["weekly_mean"] = weekly_mean["Close"] / data["Close"] 
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]
data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data ["annual_mean"] / data ["quarterly_mean"]

data["weekly_trend"] = weekly_trend 

data["open_close_ratio"] = data["Open"] / data["Close"] 
data["high close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "weekly_trend", "open_close_ratio", "high close_ratio", "low_close_ratio"]


# In[82]:


predictions = backtest(data.iloc[365:], model, full_predictors)


# In[84]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[86]:


predictions["Predictions"].value_counts()


# In[87]:


data.tail()


# In[64]:


predictions


# In[88]:


predictions.tail(20)


# In[ ]:




