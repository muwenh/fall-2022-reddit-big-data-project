# Databricks notebook source
# MAGIC %md
# MAGIC ## Analysis external data
# MAGIC 
# MAGIC **The notebook in ipynb doesn't save plotly graphs. Please view html format for interactive graphs.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9. Can we use number of posts in Pokémon subreddits to predict the stock price of Nintendo?
# MAGIC Business goal: Perform time series analysis and build a time series model to predict Nintendo stock price using number of posts in Pokémon subreddits.
# MAGIC 
# MAGIC Technical proposal: Gather Nintendo stock data from January 2021 to August 2022 from Yahoo Finance. Full outer Join the stock data with the time series data created in Question 2. Fill missing values in joint data with linear interpolation. Train a VAR (vector autoregression) model to predict stock price. Present result by a graph showing actual vs. predicted data.

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import plotly.express as px
import numpy as np


# COMMAND ----------

# MAGIC %md
# MAGIC The external data we used is Nintendo stock prices from  January 2021 to August 2022. The data can be downloaded from Yahoo!Finance.

# COMMAND ----------

# read data from file system
df_stocks = pd.read_csv('/dbfs/FileStore/NTDOY.csv')
df_stocks.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Figure.1 Nintendo stock prices from January 2021 to August 2022
# MAGIC 
# MAGIC Plot candlestick for Nintendo stock prices. Also plot volume data alongside with the candlestick

# COMMAND ----------

# plot candlesticks
candlesticks = go.Candlestick(x=df_stocks["Date"], open=df_stocks["Open"], high=df_stocks["High"], 
                              low=df_stocks["Low"], close=df_stocks["Close"], showlegend=False)
# plot volume
volume_bars = go.Bar(x=df_stocks["Date"], y=df_stocks["Volume"], showlegend=False, marker={"color": "rgba(128,128,128,0.5)",})
# merge two plots togther using multi-axes
fig = go.Figure(candlesticks)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(candlesticks, secondary_y=True)
fig.add_trace(volume_bars, secondary_y=False)
fig.update_layout(title="Nintendo Stock Prices from January 2021 to August 2022")
fig.update_yaxes(title="Nintendo Stock Price", secondary_y=True, showgrid=True)
fig.update_yaxes(title="Volume", secondary_y=False, showgrid=False)
fig.write_html("../../data/plots/NTDOY.html")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The graph shows the stock price of Nintendo from Jan 2021 to Aug 2022. The price displays a decreasing trend in the recent year with occasional oscillations and dropped from 15.74 to 10.22. There does not exist much seasonality since the stock price would be largely affected by the overall trend of economic recession in the recent post-pandemic period and several unique corporate events.
# MAGIC 
# MAGIC On Feb 17, 2021, Nintendo hosted the longest Nintendo Direct presentation in recent years to share news on available games as well as upcoming games. That included Pokémon Snap which would come to Nintendo Switch. This droves the stock price up to a relatively high point. The same logic holds for the surge in June 2021 when Nintendo was preparing for its E3 Nintendo Direct on June 15th, 2021.
# MAGIC 
# MAGIC The steep decreasing trend from Jul 2021 to Dec 2021 is mainly due to decreasing demand in both hardware and software markets, which also caused a decrease in net sales. Also, main investor pulling back could be another possible reason for the decrease.
# MAGIC 
# MAGIC The stock gradually recovered from Dec 2021 to Mar 2022 with a series of releasing events happening during the last quarter of fiscal 2022. That also included several updates on the Pokémon series, such as Pokémon Legends: Arceus.
# MAGIC 
# MAGIC Further step: 1. Full outer join the stock price data with the time series data attained from Question 2. 2. Filling in missing values with linear interpolation. 3. Build time series models to see whether we can predict Nintendo stock price with the number of posts in Pokémon subreddits.

# COMMAND ----------

# MAGIC %md
# MAGIC Join the Nintendo stock data with our reddit data. We select 9 subreddits and compute their number of comments by days. The number of comments data is joined with stock data.

# COMMAND ----------

pokemon_submissions = spark.read.parquet("/FileStore/pokemon_submissions")
pokemon_submissions = pokemon_submissions.withColumn("date", f.from_unixtime(f.col("created_utc"), "yyyy-MM-dd"))
df_timeseries = pd.read_csv("../../data/csv/top3_subreddit_timeseires.csv")

# select top 3 reddits and those related to Nintendo games
# group by date and subreddit
# sum number of comments
# "pokemon","pokemontrades","pokemongo",
subreddits = ["PokemonTCG","PokemonSwordAndShield","PokemonUnite","PokemonMasters","PokemonLegendsArceus","PokemonBDSP"]
pokemon_submissions_date = pokemon_submissions.filter(f.col("subreddit").isin(subreddits)) \
                                              .groupBy("date", "subreddit") \
                                              .agg(f.sum(f.col("num_comments")).alias("num_comments_date")) \
                                              .sort(f.col("date").desc()) \
                                              .toPandas()

list_tcg = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonTCG"].num_comments_date.tolist()
list_swsd = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonSwordAndShield"].num_comments_date.tolist()
list_master = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonMasters"].num_comments_date.tolist()
df_timeseries = df_timeseries.assign(PokemonTCG=list_tcg, PokemonSwordAndShield=list_swsd,  PokemonMasters=list_master)
# thses subreddits have less than 608 days, thus use join
uniteDF = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonUnite"][['date','num_comments_date']].rename(columns={'num_comments_date':'PokemonUnite'})
arcuDF = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonLegendsArceus"][['date','num_comments_date']].rename(columns={'num_comments_date':'PokemonLegendsArceus'})
BDSPDF = pokemon_submissions_date[pokemon_submissions_date.subreddit == "PokemonBDSP"][['date','num_comments_date']].rename(columns={'num_comments_date':'PokemonBDSP'})

df_timeseries = df_timeseries.merge(uniteDF, how='outer')
df_timeseries = df_timeseries.merge(arcuDF, how='outer')
df_timeseries = df_timeseries.merge(BDSPDF, how='outer')
df_timeseries

# COMMAND ----------

# MAGIC %md
# MAGIC The result above shows that the data contains some missing values. Thus, we use linear interpolation to fill missing values.

# COMMAND ----------

# Merge reddit dataset and stock price dataset
df_stocks = df_stocks[['Date','Close','Volume']].rename(columns={'Date':'date'})
df_timeseries_stocks = df_timeseries.merge(df_stocks, how='outer')
# fill missing values by linear interpolation
df_timeseries_stocks = df_timeseries_stocks.interpolate(method='linear', limit_direction = 'forward')
# save data
df_timeseries_stocks.to_csv("../../data/csv/timeseries_subreddits_stocks.csv", index=False)
df_timeseries_stocks

# COMMAND ----------

# MAGIC %md
# MAGIC #### Figure.2 Number of comments in subreddits over time and Nintendo stock price
# MAGIC 
# MAGIC Plot number of comments in subreddits over time and Nintendo stock price. Plot the external data alongside with our reddit data.

# COMMAND ----------

fig = make_subplots(specs=[[{"secondary_y": True}]])
line_1 = go.Scatter(x=df_timeseries_stocks["date"], y=df_timeseries_stocks["pokemon"],name="pokemon")
line_2 = go.Scatter(x=df_timeseries_stocks["date"], y=df_timeseries_stocks["pokemongo"],name="pokemongo")
line_3 = go.Scatter(x=df_timeseries_stocks["date"], y=df_timeseries_stocks["Close"],name="close")
# reddit data share an axis
fig.add_trace(line_1, secondary_y=False)
fig.add_trace(line_2, secondary_y=False)
# stock price uses another axis
fig.add_trace(line_3, secondary_y=True)
# Add figure title
fig.update_layout(title_text="Number of comments in Pokémon subreddits and Nintendo stock price")
fig.write_html("../../data/plots/subreddits_stock.html")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There seems to exist a certain relationship between the number of comments in Pokemon subreddits and Nintendo stock price. When the number of comments in the top subreddits surged, such as on February 26th, 2021,  June 22nd, 2021, and Feb 28th, 2022, the stock price was also increasing or at a relatively high level. Suggesting that an increase in the number of comments might indicate a future increase in the stock price, although the relationship may be too weak to be detected. Thus, the next step would be using appropriate ML models to try to predict the stock price with the number of comments in top Pokemon subreddits. Usually, the day when Reddit has a high amount of discussion is when Pokémon releases a new game or version update. Therefore, the sentiment analysis of comments can also be one of the reasons that affect the stock price. We can introduce sentiment analysis to explore the impact of public praise on the company's stock price. 
