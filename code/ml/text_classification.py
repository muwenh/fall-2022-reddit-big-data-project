# Databricks notebook source
# MAGIC %md
# MAGIC ## Text Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10. Can we classify submissions to their subreddit based on submission contents?
# MAGIC Business goal: Build classification models to predict the subreddits of the submissions based on the submission contents.
# MAGIC 
# MAGIC Technical proposal: Use submissions from selected subreddits related to Pokémon. Perform tokenization, remove stop words, remove any external links, punctuations, and special characters, and transform the submission contents into a document term matrix. Use StringIndexer to encode the label column to label indices. Split the data into training and test sets. Build Spark ML pipeline. Use models such as Naive Bayes and random forest to perform text classification. Evaluate model performances with accuracies, confusion matrices, and ROC curves.

# COMMAND ----------

import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# COMMAND ----------

pokemon_submissions = spark.read.parquet("/FileStore/pokemon_submissions")

# select top 5 subreddits
top5_subreddit = ['pokemongo','pokemon','pokemontrades','PokemonTCG','PokemonSwordAndShield']
top_pokemon = pokemon_submissions.filter(f.col("subreddit").isin(top5_subreddit)).select("subreddit", "title")

top_pokemon.show(5)

# COMMAND ----------

top_pokemon.printSchema()

# COMMAND ----------

top_pokemon.groupby('subreddit') \
    .count() \
    .orderBy(f.col('count').desc()) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model pipeline 

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="title", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["http","https","amp","rt","t","c","the"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

label_stringIdx = StringIndexer(inputCol = "subreddit", outputCol = "label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(top_pokemon)
dataset = pipelineFit.transform(top_pokemon)
dataset.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split data into training and test sets

# COMMAND ----------

# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model training and evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic regression (softmax)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)

# COMMAND ----------

# MAGIC %md
# MAGIC #### LR Model evaluation

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
lrAccuracy = evaluator_acc.evaluate(predictions)
lrAUC = evaluator_auc.evaluate(predictions)

print("Test Accuracy = %g" % lrAccuracy)
print("Test Error = %g" % (1.0 - lrAccuracy))
print("ROC AUC = %g" % lrAUC)

# COMMAND ----------

# Cross validation & Parameter tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
             .addGrid(lr.maxIter, [10, 20]) #Number of iterations
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator_auc, \
                    numFolds=5)
cvModel = cv.fit(trainingData)
# get best model
bestModel = cvModel.bestModel

# COMMAND ----------

# best model hyperparameter sets
print('Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam())
print('Best Param (regParam): ', bestModel._java_obj.getRegParam())
print('Best Param (MaxIter): ', bestModel._java_obj.getMaxIter())

# COMMAND ----------

# evaluate the best model by cross validation
predictions_best = bestModel.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
lrAccuracy_best = evaluator_acc.evaluate(predictions_best)
lrAUC_best = evaluator_auc.evaluate(predictions_best)

print("Test Accuracy = %g" % lrAccuracy_best)
print("Test Error = %g" % (1.0 - lrAccuracy_best))
print("ROC AUC = %g" % lrAUC_best)

# COMMAND ----------

# MAGIC %md
# MAGIC The best model selected by cross validation has slightly lower accuracy and ROC AUC compared to the first model. Thus, we still choose the first hyperparameter set.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the LR model

# COMMAND ----------

lrModel.write().overwrite().save('/FileStore/my_folder/fitted_models/lrModel')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Evaluation

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

y_pred = predictions.select("prediction").collect()
y_orig = predictions.select("label").collect()

cm = confusion_matrix(y_orig, y_pred).tolist()
class_names = ['pokemongo', 'pokemon', 'pokemontrades', 'PokemonTCG', 'PokemonSwordAndShield']

# change each element of cm to type string for annotations
cm_text = [[str(y) for y in x] for x in cm]
# set up figure 
fig = ff.create_annotated_heatmap(cm, x=class_names, y=class_names, annotation_text=cm_text, colorscale='Viridis')
# add title
fig.update_layout(title_text='<i><b>Confusion matrix of Logistic Regression</b></i>')
# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.15, showarrow=False, text="Predicted value", xref="paper", yref="paper"))
# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=-0.18, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))
# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))
# add colorbar
fig['data'][0]['showscale'] = True
fig.write_html("../../data/plots/text_classification_lr.html")
fig.show()

# COMMAND ----------

# Performance on each class
from sklearn.metrics import classification_report
print(classification_report(y_orig, y_pred, target_names=class_names))

# COMMAND ----------

# MAGIC %md
# MAGIC In order to answer our business question, we decide to build machine learning models to classify submission titles to their subreddits. Submission titles are text data which cannot be directly used as input for models, so we should further process submisson titles data. We tokenize submission titles with regular expression, removing stop words and vectorize tokens by term frequency. The first model we build is multinominal logistic regression or softmax regression. The text classification creates a large number of features, so it's possible that our model will suffer from multicollinearity. To resolve this problem, we decide to use Ridge regression which has elastic net of 0 to train model. The confusion matrix and evaluation metrics including accuracy, precision, recall and f-1 score are shown above. The PokemonTCG has the highest precision and f-1 score and pokemontrades is the second. The PokemonTCG is about the trading card game which is distinctive from other Pokemon products, so we speculate that special contents of this subreddit make it easy for the model to predict. The model has the worst ability to predict PokemonSwordAndShield among all subreddits. The PokemonSwordAndShield has a lot of overlaps with pokemon and pokemongo, making it similar with other subreddits. Thus, the model fail to distinguish PokemonSwordAndShield from other subreddits. To improve the performance of the model, we perform 5-fold cross validation using different combination of parameters. The result model has similar performance compared to the model we first trained. This indicates that the model we trained is good enough and further tuning may not improve model performance significantly.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Naive Bayes

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the NB model

# COMMAND ----------

# save model
model.write().overwrite().save("/FileStore/my_folder/fitted_models/nb_model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### NB Model evaluation

# COMMAND ----------

predictions_nb = model.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
nbAccuracy = evaluator_acc.evaluate(predictions_nb)
nbAUC = evaluator_auc.evaluate(predictions_nb)

print("Test Accuracy = %g" % nbAccuracy)
print("Test Error = %g" % (1.0 - nbAccuracy))
print("ROC AUC = %g" % nbAUC)

# COMMAND ----------

y_pred = predictions_nb.select("prediction").collect()
y_orig = predictions_nb.select("label").collect()

cm = confusion_matrix(y_orig, y_pred).tolist()
class_names = ['pokemongo', 'pokemon', 'pokemontrades', 'PokemonTCG', 'PokemonSwordAndShield']

# change each element of cm to type string for annotations
cm_text = [[str(y) for y in x] for x in cm]
# set up figure 
fig = ff.create_annotated_heatmap(cm, x=class_names, y=class_names, annotation_text=cm_text, colorscale='Viridis')
# add title
fig.update_layout(title_text='<i><b>Confusion matrix of Naive Bayes</b></i>')
# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.15, showarrow=False, text="Predicted value", xref="paper", yref="paper"))
# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14), x=-0.18, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))
# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))
# add colorbar
fig['data'][0]['showscale'] = True
fig.write_html("../../data/plots/text_classification_nb.html")
fig.show()

# COMMAND ----------

# Performance on each class
from sklearn.metrics import classification_report
print(classification_report(y_orig, y_pred, target_names=class_names))

# COMMAND ----------

# MAGIC %md
# MAGIC The confusion matrix and table above shows the classification result of naive bayes model. This model has similar overall accuracy compared to the random forest. The result also shows the similar pattern as the random forest that PokemonTCG has the highest prediction accuracy and PokemonSwordAndShield has the lowest. The naive bayes model has better ability to predict pokemongo and pokemon, but has lower prediction accuracy in pokemontrades. Considering the fact that pokemongo and pokemon are top two common labels in the training data, we think the ability to predict pokemon and pokemongo is more important than predicting pokemontrades. Although the accuracy of predicting PokemonSwordAndShield is low, the naive bayes has higher f-1 score which is 0.44 than that of the logistic regression which is 0.36. This indicates that the naive bayes model has higher overall accuracy in predicting PokemonSwordAndShield. Thus, we think the naive bayes model has better ability to predict subreddit than the logistic regression model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf_1 = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=150, maxDepth=6)
rfModel_1 = rf_1.fit(trainingData)

rf_2 = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200, maxDepth=10)
rfModel_2 = rf_2.fit(trainingData)

# COMMAND ----------

# MAGIC %md
# MAGIC #### RF Model evaluation

# COMMAND ----------

predictions_rf_1 = rfModel_1.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
rfAccuracy_1 = evaluator_acc.evaluate(predictions_rf_1)
rfAUC_1 = evaluator_auc.evaluate(predictions_rf_1)

print("Test Accuracy = %g" % rfAccuracy_1)
print("Test Error = %g" % (1.0 - rfAccuracy_1))
print("ROC AUC  = %g" % rfAUC_1)

# COMMAND ----------

predictions_rf_2 = rfModel_2.transform(testData)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
rfAccuracy_2 = evaluator_acc.evaluate(predictions_rf_2)
rfAUC_2 = evaluator_auc.evaluate(predictions_rf_2)

print("Test Accuracy = %g" % rfAccuracy_2)
print("Test Error = %g" % (1.0 - rfAccuracy_2))
print("ROC AUC  = %g" % rfAUC_2)

# COMMAND ----------

# MAGIC %md
# MAGIC The second hyperparameter sets have higher accuracy and ROC AUC. Thus, we choose the second hyperparameter set.

# COMMAND ----------

# Performance on each class
from sklearn.metrics import classification_report
y_pred = predictions_rf_2.select("prediction").collect()
y_orig = predictions_rf_2.select("label").collect()
print(classification_report(y_orig, y_pred, target_names=class_names, zero_division=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model selection  

# COMMAND ----------

summary_df = pd.DataFrame({'model': ['logistic regression', 'naive bayes', 'random forest'], 
                           'accuracy': [0.73, 0.73, 0.54], 
                           'precision (weighted)': [0.74, 0.73, 0.71], 
                           'recall (weighted)': [0.73, 0.73, 0.54], 
                           'f1-score (weighted)': [0.71, 0.72, 0.51],
                           'ROC AUC': [lrAUC, nbAUC, rfAUC_2]})
summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC Compared with the three models, the performance of random forest is obviously worse than the other two. The results of logistic regression and Naive Bayes are very close, and  Naive Bayes is slightly better in accuracy. Both models can better classify the subreddit by content. 
# MAGIC Through these two models, we can effectively classify the subreddit by content. Especially on the pokemon subreddit widely discussed by the players, users sometimes cannot accurately distinguish the content they need to browse. With this model, we can archive the content published in the pokemon subreddit to the corresponding subreddit. We can also label the posts in pokemon subreddit to help users classify and improve their browsing experience.
