# Databricks notebook source
# MAGIC %md
# MAGIC ### Natural language processing.
# MAGIC 
# MAGIC - Find out most common words overall.
# MAGIC 
# MAGIC - Find out distribution of text length.
# MAGIC 
# MAGIC - Find out important words using TF-IDF

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
# MAGIC We use top 5 most popular Pokemon subreddits to perform NLP tasks, including "pokemongo", "pokemon", "pokemontrades", "PokemonTCG", "PokemonSwordAndShield". According to the result in EDA, these 5 subreddits contain about 11 million comments.

# COMMAND ----------

pokemon_comments = spark.read.parquet("/FileStore/pokemon_comments")

# select top 5 subreddits
top5_subreddit = ['pokemongo','pokemon','pokemontrades','PokemonTCG','PokemonSwordAndShield']
top_pokemon = pokemon_comments.filter(f.col("subreddit").isin(top5_subreddit)).select("subreddit", "body", "created_utc", "score")

# COMMAND ----------

# MAGIC %md
# MAGIC Count common words overall.

# COMMAND ----------

# split sentence into words and count word occurrence
words_count = top_pokemon.withColumn('word', f.explode(f.split(f.col('body'), ' '))) \
    .groupBy('word') \
    .agg(f.count('word').alias('word_count')) \
    .sort('word_count', ascending=False)
words_count.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The table above shows the most common words in comments data. The top common words are "the", "a", "to" which are very common in English but don't contain any useful meaning. Thus, the table above doesn’t display any useful results. In the following step, we remove these stopwords and count again.

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover

tokenizer = Tokenizer(inputCol="body", outputCol="words_token")
pokemon_tokenized = tokenizer.transform(top_pokemon).select("body","words_token", "score")
# remove stop words
stopwordsRemover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
pokemon_clean = stopwordsRemover.transform(pokemon_tokenized).select("body", "words_clean")

pokemon_clean.show()

# COMMAND ----------

# count words after removing stop words
words_count = pokemon_clean.withColumn('word', f.explode(f.col('words_clean'))) \
    .groupBy('word') \
    .agg(f.count('word').alias('word_count')) \
    .sort('word_count', ascending=False) \
    .filter((f.col('word') != " ") & (f.col('word') != "-") & (f.col('word') != "*") & (f.col('word') != ""))
words_count = words_count.toPandas()
words_count.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC After removing stopwords, the word count result is much more meaningful. Words like "like", "trading", "pokémon", "post" are very common. We should expect to see them in future analysis.
# MAGIC 
# MAGIC Compute the length distribution of the comments

# COMMAND ----------

# compute length distribution of the comments
comments_length = top_pokemon.withColumn('comments_length', f.size(f.split(f.col('body'), ' '))) \
    .groupby('comments_length') \
    .agg(f.count('comments_length').alias('count')) \
    .sort('count', ascending=False)
comments_length.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Figure.1 Comments length distribution
# MAGIC 
# MAGIC Plot comments length with bar graph.

# COMMAND ----------

comments_length = comments_length.toPandas()
# Plot comments length distribution
plt.figure(figsize=(15,9))
# select only comments length with more than 200 occurrences
plot_1 = sns.barplot(x="comments_length", y="count", data=comments_length[comments_length['count'] > 200])
plot_1.set_title("Comments length distribution", fontsize=25)
plot_1.set_xlabel("Comments length", fontsize=15)
plot_1.set_ylabel("Number of comments", fontsize=15)
plot_1.set_xticklabels([])

plot_fpath = os.path.join("../../data/plots", "comments_length.png")
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Before proceeding to the NLP cleaning steps, let us examine the distribution of comments length. As comment length becomes longer, the counts gradually drop. Indicating that there exists a negative relationship between the number of comments and comment length. That suggests that the majority of comments are relatively short and people seldom write long comments in online social communities. The fact augments the difficulty for further tasks such as text classification. Since there might not exist enough context information for the model to decide which class the comments belong to.     

# COMMAND ----------

# MAGIC %md
# MAGIC Use TF-IDF to identify important words in comments.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer
# perform TF-IDF
tokenizer = Tokenizer(inputCol="body", outputCol="words")
df_token = tokenizer.transform(top_pokemon)
# compute term frequency
cv = CountVectorizer(inputCol="words", outputCol="tf", vocabSize=10000, minDF=5)
cv_model=cv.fit(df_token)
df_tf = cv_model.transform(df_token)
# compute TF-IDF
idf=IDF(inputCol="tf", outputCol="tf_idf")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)
df_tfidf.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Find out top 3 most important words for each comment. Due to the CountVectorizer, some words are not counted. Thus, its's possible that a comment doesn't contain any important words.

# COMMAND ----------

# vocabulary set
vocab = cv_model.vocabulary
# function to extract the most important words
def max_tfidf(features):
    text = features['body']
    vector = features['tf_idf']
    top3_index = np.argsort(vector)[::-1][:3]
    return (text, [vocab[index] for index in top3_index])
# show most important words by TF-IDF
tfidf_words = df_tfidf.select('body','tf_idf').rdd.map(lambda x: max_tfidf(x)).toDF(["body", "important_words"])
tfidf_words.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NLP Cleaning
# MAGIC 
# MAGIC For NLP cleaning, we perform following procedures.
# MAGIC 
# MAGIC  - Remove URL links
# MAGIC  - Remove ('s), ('ll), ('ve), ('m), ('re)
# MAGIC  - Remove punctuation and special characters
# MAGIC  - Remove extra white space
# MAGIC  - Tokenization
# MAGIC  - Normalize word frequencies
# MAGIC  - Remove stopwords
# MAGIC  - Lemmatization

# COMMAND ----------

from pyspark.ml import Pipeline
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
import pyspark.sql.functions as f

# COMMAND ----------

pokemon_comments = spark.read.parquet("/FileStore/pokemon_comments")

# select top 5 subreddits
top5_subreddit = ["pokemongo",'pokemon','pokemontrades','PokemonTCG','PokemonSwordAndShield']
top_pokemon = pokemon_comments.filter(f.col("subreddit").isin(top5_subreddit)).select("subreddit", "body", "created_utc", "score")

# COMMAND ----------

# MAGIC %md
# MAGIC Use regex to perform cleaning procedures.

# COMMAND ----------

# regex to identify most of url links
url = r'''(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'''
# remove url links
top_pokemon = top_pokemon.withColumn("body_clean", f.regexp_replace(f.col("body"), url, ""))
# remove 's, 'll, 've, 'm, 're
top_pokemon = top_pokemon.withColumn("body_clean", f.regexp_replace(f.col("body_clean"), r"'s\b|'ll\b|'ve\b|'m\b|'re\b", ""))
# remove special characters
top_pokemon = top_pokemon.withColumn("body_clean", f.regexp_replace(f.col("body_clean"), r"[\t\n\r\*\@\-\/]", " "))
# remove extra white spaces
top_pokemon = top_pokemon.withColumn("body_clean", f.regexp_replace(f.col("body_clean"), r"\s+", " "))

top_pokemon.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Build sentiment model pipeline.

# COMMAND ----------

# build sentiment model pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("body_clean") \
    .setOutputCol("document")
# tokenization
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
# normalize word frequencies
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("token_normalized")
# remove stop words
stopwords_remover = StopWordsCleaner() \
    .setInputCols("token_normalized") \
    .setOutputCol("token_clean") \
    .setCaseSensitive(False)
# lemmatization
lemmatization = LemmatizerModel.pretrained('lemma_antbnc') \
    .setInputCols(["token_clean"]) \
    .setOutputCol("lemmatization")
# word embedding
wordEmbedding = WordEmbeddingsModel.pretrained() \
    .setInputCols(["document", "lemmatization"]) \
    .setOutputCol("word_embeddings") \
    .setCaseSensitive(False)
# sentence embedding 
sentenceEmbedding = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
    .setInputCols(["document", "word_embeddings"])\
    .setOutputCol("sentence_embeddings")
# sentiment model
sentimentModel = SentimentDLModel.pretrained(name="sentimentdl_use_twitter", lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")
# build pipeline
sentimentPipeline = Pipeline(
    stages = [documentAssembler,
              tokenizer,
              normalizer,
              stopwords_remover,
              lemmatization,
              wordEmbedding,
              sentenceEmbedding,
              sentimentModel])

# COMMAND ----------

empty_df = spark.createDataFrame([['']]).toDF("body_clean")
pipelineModel = sentimentPipeline.fit(empty_df)

# transform data
sentiment_result = pipelineModel.transform(top_pokemon)

# COMMAND ----------

# MAGIC %md
# MAGIC Extract sentiment result and join to the comments data

# COMMAND ----------

# extract sentiment result and join to the comments data
pokemon_sentiment = sentiment_result.select("subreddit", "body", "body_clean", "created_utc", "score", 
                                            f.explode("sentiment.result").alias("sentiment"))
pokemon_sentiment.show(10)

# COMMAND ----------

# save filtered sentiment result to file system
pokemon_sentiment.write.parquet("/FileStore/pokemon_comments_sentiment")
