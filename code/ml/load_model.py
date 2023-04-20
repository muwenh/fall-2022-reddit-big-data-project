# Databricks notebook source
import pyspark.sql.functions as f
pokemon_submissions = spark.read.parquet("/FileStore/pokemon_submissions")

# select top 5 subreddits
top5_subreddit = ['pokemongo','pokemon','pokemontrades','PokemonTCG','PokemonSwordAndShield']
top_pokemon = pokemon_submissions.filter(f.col("subreddit").isin(top5_subreddit)).select("subreddit", "title")

top_pokemon.show(5)

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

from  pyspark.ml.classification import LogisticRegressionModel
loaded_lrModel = LogisticRegressionModel.load('/FileStore/my_folder/fitted_models/lrModel')
lr_predictions = loaded_lrModel.transform(dataset)
lr_predictions.show(5)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lrAccuracy = evaluator.evaluate(lr_predictions)

print("Training Accuracy = %g" % lrAccuracy)
print("Training Error = %g" % (1.0 - lrAccuracy))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
lrAUC = evaluator.evaluate(predictions)

print("ROC AUC = %g" % lrAUC)

# COMMAND ----------

# load model
from pyspark.ml.classification import  NaiveBayesModel
loaded_nb_model = NaiveBayesModel.load("/FileStore/my_folder/fitted_models/nb_model")
nb_predictions = loaded_nb_model.transform(dataset)
nb_predictions.show(5)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nbAccuracy = evaluator.evaluate(nb_predictions)

print("Training Accuracy = %g" % nbAccuracy)
print("Training Error = %g" % (1.0 - nbAccuracy))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
nbAUC = evaluator.evaluate(nb_predictions)

print("ROC AUC = %g" % nbAUC)
