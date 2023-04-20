# Databricks notebook source
# MAGIC %md
# MAGIC ## Sentiment Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11. Is certain Pokémon influencing people's sentiments in Pokémon subreddits?
# MAGIC Business goal: Use machine learning models to predict people's sentiments based on present of certain Pokémon.
# MAGIC 
# MAGIC Technical proposal: Create dummy variables to indicate the present of popular Pokémon. Split sentiment dataset created by sentiment model into training data and test data. Build machine learning pipelines for logistic regression and random forest models. Evaluate models based on multiple metrics, including precision, recall, f-1 score, accuracy and ROC AUC. Plot heatmap for confusion matrix and create summary table for each model to present result.

# COMMAND ----------

import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# COMMAND ----------

pokemon_sentiments = spark.read.parquet("/FileStore/pokemon_comments_sentiment")
pokemon_sentiments.show(10)

# COMMAND ----------

# Create pokemon dummies by searching pokemon names
pokemon_sentiments = pokemon_sentiments.withColumn("Charizard", f.col("body").rlike("(?i)charmander|(?i)charmeleon|(?i)charizard").cast("int")) \
    .withColumn("Pikachu", f.col("body").rlike("(?i)pichu|(?i)pikachu|(?i)raichu").cast("int")) \
    .withColumn("Eevee", f.col("body").rlike("(?i)Vaporeon|(?i)Jolteon|(?i)Flareon|(?i)Espeon|(?i)Umbreon|(?i)Leafeon|(?i)Glaceon|(?i)Sylveon").cast("int")) \
    .withColumn("Weather_Trio", f.col("body").rlike("(?i)Kyogre|(?i)Groudon|(?i)Rayquaza").cast("int")) \
    .withColumn("Legendary_Beasts", f.col("body").rlike("(?i)Raikou|(?i)Entei|(?i)Suicune|(?i)Lugia|(?i)Ho-oh").cast("int")) \
    .withColumn("Creation_Trio", f.col("body").rlike("(?i)Dialga|(?i)Palkia|(?i)Giratina").cast("int"))

# Only select useful variables
pokemon_sentiments = pokemon_sentiments.select('subreddit', 'score', 'Charizard', 'Pikachu', 'Eevee', 'Weather_Trio', 'Legendary_Beasts', 'Creation_Trio', 'sentiment')

pokemon_sentiments.show(10)

# COMMAND ----------

# From nlp Part
# count number of sentiments by subreddit
sentiments_by_sub = pokemon_sentiments.groupby("subreddit", "sentiment").count().orderBy("subreddit", "sentiment").toPandas()
sentiments_by_sub

# COMMAND ----------

# From nlp Part
# Number of sentiments by subreddit
df_sentiment =  pd.DataFrame([sentiments_by_sub["count"][2::-1].tolist()], columns=["positive", "neutral", "negative"])
i = 1
for i in range(1,5):
    content = sentiments_by_sub["count"][2+3*i:2+3*(i-1):-1].tolist()
    df_sentiment.loc[i] = content
df_sentiment.index = ["PokemonSwordAndShield", "PokemonTCG", "pokemon", "pokemongo", "pokemontrades"]
df_sentiment.sort_values(by=["positive"], ascending=False, inplace=True)
df_sentiment

# COMMAND ----------

# From nlp Part
# Number of sentiments by Pokemon
sentiments_by_pokemon = pokemon_sentiments.groupBy("sentiment") \
    .agg(f.sum("Charizard").alias("Charizard"), f.sum("Pikachu").alias("Pikachu"), f.sum("Eevee").alias("Eevee"),
        f.sum("Weather_Trio").alias("Weather_Trio"), f.sum("Legendary_Beasts").alias("Legendary_Beasts"),
        f.sum("Creation_Trio").alias("Creation_Trio")).toPandas()

sentiments_by_pokemon

# COMMAND ----------

# Sum scores by sentiments
sentiment_score_sum = pokemon_sentiments.groupby('sentiment').agg(f.sum('score').alias('score_sum')).orderBy(f.col('score_sum').desc()).toPandas()
sentiment_score_sum

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split data into training and test sets

# COMMAND ----------

trainingData, testData = pokemon_sentiments.randomSplit([0.8, 0.2], 100)

# COMMAND ----------

# count the number of rows for each split
print("Number of training records: " + str(trainingData.count()))
print("Number of testing records : " + str(testData.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model pipelines

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, Model

# COMMAND ----------

trainingData.printSchema()

# COMMAND ----------

# Convert all the string fields to numeric indices
stringIndexer_sentiment = StringIndexer(inputCol="sentiment", outputCol="sentiment_ix")
stringIndexer_subreddit = StringIndexer(inputCol="subreddit", outputCol="subreddit_ix")

# Convert index variables that have more than two levels with OneHotEncoder
onehot_subreddit = OneHotEncoder(inputCol="subreddit_ix", outputCol="subreddit_vec")

# Combine all features together using the vectorAssembler method
vectorAssembler_features = VectorAssembler(
    inputCols=["subreddit_vec", "score", "Charizard", "Pikachu", "Eevee", "Weather_Trio", "Legendary_Beasts", "Creation_Trio"], 
    outputCol= "features")

# Models
rf_1 = RandomForestClassifier(labelCol="sentiment_ix", featuresCol="features", numTrees=50, maxDepth=4)
rf_2 = RandomForestClassifier(labelCol="sentiment_ix", featuresCol="features", numTrees=150, maxDepth=6)
lr_1 = LogisticRegression(labelCol="sentiment_ix", featuresCol="features", maxIter=20, regParam=0.3, elasticNetParam=0)
lr_2 = LogisticRegression(labelCol="sentiment_ix", featuresCol="features", maxIter=50, regParam=0.3, elasticNetParam=1)

# COMMAND ----------

# Inspect the indexed labels of sentiment
indexed_sentiment = stringIndexer_sentiment.fit(trainingData)
indexed_sentiment.labels

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra Credit: Apply pipelines to compare hyperparameters and model options

# COMMAND ----------

# Build pipelines
pipeline_rf_1 = Pipeline(stages=[stringIndexer_sentiment, 
                               stringIndexer_subreddit,
                               onehot_subreddit,
                               vectorAssembler_features,
                               rf_1])
pipeline_rf_2 = Pipeline(stages=[stringIndexer_sentiment, 
                               stringIndexer_subreddit,
                               onehot_subreddit,
                               vectorAssembler_features,
                               rf_2])

pipeline_lr_1 = Pipeline(stages=[stringIndexer_sentiment, 
                               stringIndexer_subreddit,
                               onehot_subreddit,
                               vectorAssembler_features,
                               lr_1])

pipeline_lr_2 = Pipeline(stages=[stringIndexer_sentiment, 
                               stringIndexer_subreddit,
                               onehot_subreddit,
                               vectorAssembler_features,
                               lr_2])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model training and evaluation 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

model_rf_1 = pipeline_rf_1.fit(trainingData)
pipelined_train_rf_1 = model_rf_1.transform(trainingData)
pipelined_train_rf_1.show(5)

# COMMAND ----------

# evaluate first hyperparameter set
predictions_1 = model_rf_1.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="sentiment_ix", rawPredictionCol="prediction", metricName="areaUnderROC")
rfAccuracy_1 = evaluator_acc.evaluate(predictions_1)
rfAUC_1 = evaluator_auc.evaluate(predictions_1)

print("Accuracy = %g" % rfAccuracy_1)
print("Test Error = %g" % (1.0 - rfAccuracy_1))
print("ROC AUC = %g" % rfAUC_1)

# COMMAND ----------

# # evaluate second hyperparameter set
model_rf_2 = pipeline_rf_2.fit(trainingData)
predictions_2 = model_rf_2.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="sentiment_ix", rawPredictionCol="prediction", metricName="areaUnderROC")
rfAccuracy_2 = evaluator_acc.evaluate(predictions_2)
rfAUC_2 = evaluator_auc.evaluate(predictions_2)

print("Accuracy = %g" % rfAccuracy_2)
print("Test Error = %g" % (1.0 - rfAccuracy_2))
print("ROC AUC = %g" % rfAUC_2)

# COMMAND ----------

# MAGIC %md
# MAGIC The second random forest has higher accuracy and ROC AUC. Thus, we should choose the second hyperparameter sets.

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

y_pred = predictions_2.select("prediction").collect()
y_orig = predictions_2.select("sentiment_ix").collect()

cm = confusion_matrix(y_orig, y_pred).tolist()
class_names = ['positive', 'negative', 'neutral']

# change each element of cm to type string for annotations
cm_text = [[str(y) for y in x] for x in cm]
# set up figure 
fig = ff.create_annotated_heatmap(cm, x=class_names, y=class_names, annotation_text=cm_text, colorscale='Viridis')
# add title
fig.update_layout(title_text='<i><b>Confusion matrix of Random Forest</b></i>')
# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.15, showarrow=False, text="Predicted value", xref="paper", yref="paper"))
# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=-0.075, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))
# adjust margins to make room for yaxis title
# fig.update_layout(margin=dict(t=50, l=200))
# add colorbar
fig['data'][0]['showscale'] = True
fig.write_html("../../data/plots/sentiment_classification_rf.html")
fig.show()

# COMMAND ----------

# Performance on each class
from sklearn.metrics import classification_report
report_rf = classification_report(y_orig, y_pred, target_names=class_names, zero_division=0)
print(report_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC To answer our business question, we decide to build machine learning models to predict comment sentiments using Pokémon dummy variables. The first model we trained is random forest. This model predicts most of comments as positive which lead to the high f-1 score in positive and very low f-1 scores in negative and neutral. The ROC AUC value of 0.5335 also indicates this model has almost no ability to predict comment sentiments. The worst prediction is about the neutral sentiment. However, this may not be a problem of the classification model, because neutral sentiment is a very small part of the overall datasets and it's inherently indistinguishable and unbiased. This model shows very little ability to predict sentiments based on Pokémon. However, by changing hyperparameter sets to create more complicted random forest, model performace increases. Thus, we think further tuning the random forest can improve the model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic regression (softmax)

# COMMAND ----------

model_lr_1 = pipeline_lr_1.fit(trainingData)
pipelined_train_lr_1 = model_lr_1.transform(trainingData)
pipelined_train_lr_1.show(5)

# COMMAND ----------

# evaluate first hyperparameter set
predictions_lr_1 = model_lr_1.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="sentiment_ix", rawPredictionCol="prediction", metricName="areaUnderROC")
lrAccuracy_1 = evaluator_acc.evaluate(predictions_lr_1)
lrAUC_1 = evaluator_auc.evaluate(predictions_lr_1)

print("Accuracy = %g" % lrAccuracy_1)
print("Test Error = %g" % (1.0 - lrAccuracy_1))
print("ROC AUC = %g" % lrAUC_1)

# COMMAND ----------

# evaluate second hyperparameter set
model_lr_2 = pipeline_lr_2.fit(trainingData)
predictions_lr_2 = model_lr_2.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="sentiment_ix", rawPredictionCol="prediction", metricName="areaUnderROC")
lrAccuracy_2 = evaluator_acc.evaluate(predictions_lr_2)
lrAUC_2 = evaluator_auc.evaluate(predictions_lr_2)

print("Accuracy = %g" % lrAccuracy_2)
print("Test Error = %g" % (1.0 - lrAccuracy_2))
print("ROC AUC = %g" % lrAUC_2)

# COMMAND ----------

y_pred = predictions_lr_1.select("prediction").collect()
y_orig = predictions_lr_1.select("sentiment_ix").collect()

cm = confusion_matrix(y_orig, y_pred).tolist()
class_names = ['positive', 'negative', 'neutral']

# change each element of cm to type string for annotations
cm_text = [[str(y) for y in x] for x in cm]
# set up figure 
fig = ff.create_annotated_heatmap(cm, x=class_names, y=class_names, annotation_text=cm_text, colorscale='Viridis')
# add title
fig.update_layout(title_text='<i><b>Confusion matrix of Logistic Regression</b></i>')
# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.15, showarrow=False, text="Predicted value", xref="paper", yref="paper"))
# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=-0.075, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))
# adjust margins to make room for yaxis title
# fig.update_layout(margin=dict(t=50, l=200))
# add colorbar
fig['data'][0]['showscale'] = True
fig.write_html("../../data/plots/sentiment_classification_lr.html")
fig.show()

# COMMAND ----------

# Performance on each class
print(classification_report(y_orig, y_pred, target_names=class_names, zero_division=0))

# COMMAND ----------

# MAGIC %md
# MAGIC The second model we build is the logistic regression model or SoftMax regression to compare with random forest. By changing parameters, we transform it to ridge regression and lasso regression. The performance of logistic regression is very similar to the random forest. It also predicts almost all comments as positive. The ROC AUC value of ridge regression is 0.500024 and that of lasso regression is 0.5 which are even worse than the random forest. Both negative and neutral has f-1 score of 0, so this model cannot make correct prediction of these two sentiments. We can conclude that the logistic regression model has no ability to predict comments sentiment based on Pokémon. Because the performance of logistic regression is so bad, we think further tuning may not be useful to improve prediction results. We think the logistic regression model may not be suitable for this classification task.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model selection

# COMMAND ----------

summary_df = pd.DataFrame({'model': ['random forest', 'logistic regression'], 
                           'accuracy': [0.62, 0.62], 
                           'precision (weighted)': [0.57, 0.52], 
                           'recall (weighted)': [0.62, 0.62], 
                           'f1-score (weighted)': [0.54, 0.47],
                           'ROC AUC': [rfAUC_2, lrAUC_1]})
summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC In this milestone, we focus on applying machine learning models to achieve two classification tasks: predicting sentiments of comments based on present of Pokémon and classifying submission titles to subreddits. For the first task, the reason of performing this task is to determine if certain Pokémon can influence people’s sentiments, so The Pokémon Company can make advertising strategies based on the result. We train two machine learning models, random forest and logistic regression to use present of certain Pokémon including, Charizard, Pikachu, Eevee, Weather Trio, Legendary Beasts, Creation Trio to predict sentiments of comments in Pokémon subreddits. Both models don't show promising ability to predict sentiments (Table 1). We think there are some factors may lead to this terrible model performance. First, number of present of these Pokémon is relatively small compared to all comments, so we should focus on comments containing these Pokémon instead of all comments. Second, sentiment labels are created by sentiment model which may not be accurate. The gap between sentiment generation process and prediction process may lead to huge error. Third, at current stage, both models are far from fine-tuned, so we believe further tuning models can improve their performance. Fourth, using more complicated models like deep learning might generate better results. However, due to hardware limitation, we cannot perform deep learning on the databricks cluster (no GPU). We think this is a path worth further investigation. For the conclusion, our current model cannot predict sentiments of comments using present of Pokémon.
