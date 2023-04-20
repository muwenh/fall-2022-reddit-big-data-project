# Databricks notebook source
import pandas as pd
df = pd.read_csv('../../data/csv/timeseries_subreddits_stocks.csv', index_col="date")
df.sort_index(inplace=True)
df

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# COMMAND ----------

from statsmodels.tsa.stattools import grangercausalitytests

df1 = pd.DataFrame(np.zeros((len(df.columns), len(df.columns))), columns=df.columns, index=df.columns)
for col in df1.columns:
    for row in df1.index:
        test_result = grangercausalitytests(df[[row, col]], maxlag=7, verbose=False)
        p_values = [round(test_result[i+1][0]["ssr_chi2test"][1], 4) for i in range(7)]
        p_value = min(p_values)
        df1.loc[row, col] = p_value
df1

# COMMAND ----------

# MAGIC %md
# MAGIC The table above shows the p-values of Granger’s Causality Test. The null hypothesis is that event A dones't cause event B. If the p-value is smaller than 0.05, we should reject the null and conclude that event A casues evnt B. We focus on causality of between each subreddit close price. "pokemontrades", "PokemonSwordAndShield", "PokemonMasters", "PokemonUnite" "PokemonLegendsArceus" have p-values bigger than 0.05, so they have cuaslity with close price. We will remove these columns for future analysis.

# COMMAND ----------

df.drop(columns=["pokemontrades", "PokemonSwordAndShield", "PokemonMasters", "PokemonUnite", "PokemonLegendsArceus"], inplace=True)
df

# COMMAND ----------

from statsmodels.tsa.vector_ar.vecm import coint_johansen
test_cointegration = coint_johansen(df,-1,5)
traces = test_cointegration.lr1
cvts = test_cointegration.cvt[:, 1]
df2 = pd.DataFrame(np.zeros((len(df.columns), 3)), columns=["Test Statistics", "Cointegration(95%)", "Significance"], index=df.columns)
for col, trace, cvt in zip(df.columns, traces, cvts):
    df2.loc[col, "Test Statistics"] = trace
    df2.loc[col, "Cointegration(95%)"] = cvt
    df2.loc[col, "Significance"] = trace > cvt
df2

# COMMAND ----------

df_train = df[:len(df) - 10]
df_test = df[len(df) - 10:]

# COMMAND ----------

df3 = pd.DataFrame(np.zeros((len(df_train.columns), 3)), columns=["p-value", "Decision", "Stationary"], index=df_train.columns)
alpha = 0.05
for col in df_train.columns:
    test_adfuller = adfuller(df_train[col])
    p_value = round(test_adfuller[1], 4)
    df3.loc[col, "p-value"] = p_value
    if p_value < alpha:
        df3.loc[col, "Decision"] = "reject null"
        df3.loc[col, "Stationary"] = True
    else:
        df3.loc[col, "Decision"] = "don't reject null"
        df3.loc[col, "Stationary"] = False
df3

# COMMAND ----------

df_1st_diff = df_train.diff().dropna()
df4 = pd.DataFrame(np.zeros((len(df_1st_diff.columns), 3)), columns=["p-value", "Decision", "Stationary"], index=df_1st_diff.columns)
alpha = 0.05
for col in df_1st_diff.columns:
    test_adfuller = adfuller(df_1st_diff[col])
    p_value = round(test_adfuller[1], 4)
    df4.loc[col, "p-value"] = p_value
    if p_value < alpha:
        df4.loc[col, "Decision"] = "reject null"
        df4.loc[col, "Stationary"] = True
    else:
        df4.loc[col, "Decision"] = "don't reject null"
        df4.loc[col, "Stationary"] = False
df4

# COMMAND ----------

model = VAR(df_1st_diff)

# COMMAND ----------

for i in range(1, 10):
    model_result = model.fit(i)
    result ={}
    result["AIC"] = model_result.aic
    result["BIC"] = model_result.bic
    print("Lag Order: ", i)
    print(result)

# COMMAND ----------

x = model.select_order(maxlags=12)
x.summary()

# COMMAND ----------

model_var = model.fit(8)
model_var.summary()

# COMMAND ----------

forecast_input = df_1st_diff.values[-model_var.k_ar:]
fc = model_var.forecast(y=forecast_input, steps=10)
df_forecast = pd.DataFrame(fc, index=df.index[-10:], columns=df.columns + '_1d')
df_forecast

# COMMAND ----------

df_fc = df_forecast.copy()
columns = df_train.columns
for col in columns:        
    # Roll back 1st Diff
    df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
df_fc = df_fc.iloc[:, len(columns):len(df_fc) + 2]
df_fc

# COMMAND ----------

fig, axes = plt.subplots(nrows=int(len(df_fc.columns)/2), ncols=2, figsize=(15,15), dpi=100)
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_fc[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-10:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# COMMAND ----------

# Plot forecast of future 10 days Nintendo stock price
import plotly.express as px
df_plot = df_fc.copy()
df_plot["Close"] = df_test["Close"]
df_plot.reset_index(inplace=True)

fig = px.line(df_plot, x="date", y=["Close", "Close_forecast"],
              hover_data={"date": "|%B %d, %Y"},
              title="Forecast of future 10 days Nintendo stock price")
fig.write_html("../../data/plots/stock_forecast.html")
fig.show()

# COMMAND ----------

df_result = pd.DataFrame(np.zeros((len(df_fc.columns), 6)), columns=["MAPE", "ME", "MAE", "MPE", "RMSE", "Correlation"], index=df_fc.columns)
for col in df_fc.columns:
    column = col.split("_")[0]
    actual = df_test[column]
    predicted = df_fc[col]
    df_result.loc[col, "MAPE"] = np.mean(np.abs(predicted - actual)/np.abs(actual))
    df_result.loc[col, "ME"] = np.mean(predicted - actual)
    df_result.loc[col, "MAE"] = np.mean(np.abs(predicted - actual))
    df_result.loc[col, "MPE"] = np.mean((predicted - actual)/actual)
    df_result.loc[col, "RMSE"] = np.sqrt(np.mean((predicted - actual)**2))
    df_result.loc[col, "Correlation"] = np.corrcoef(predicted, actual)[0,1]
df_result

# COMMAND ----------

# MAGIC %md
# MAGIC In order to investaige if it's possible to use number of posts in Pokémon subreddits to predict the stock price of Nintendo, we train a vector autoregressive (VAR) model to predict number of comments in Pokemon subreddits and stock price. The plot above shows the forecaset stock price by the model v.s. actual stock price and table shows multiple metrics to evaluate forecast results. The actual stock price fluctuates during the predicted 10-day period which has several sudden changes from Aug 23 to Aug 26. The model cannot predict these changes and can only capture the trend of the stock. The metrics table shows that the model has very poor ability to predict number of comments in subreddits and it's suprising to see that the accuracy of predicting stock price is much higher. This may indicate that the number of comments in Pokemon subreddits have more randomness than the stock price, so it's very difficult for time series model to capture patterns. The RMSE of the predicted stock price is about 0.4206 which is about 3.6% of total stock price. Although the error seems small, we think it's not good enough considering the short period of prediction. Thus, we do not reccomend using this model to guide the investment strategy.
