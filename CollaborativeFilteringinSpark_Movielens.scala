// Databricks notebook source exported at Sun, 6 Sep 2015 03:13:40 UTC
//Notebook illustrating Collaborative Filtering in Spark and alrogirthsm for item similarity
//Inspiration is Machine Learning with Spark by Nick Pentreath
// http://mlnick.github.io/blog/2013/04/01/movie-recommendations-and-more-with-spark/

// COMMAND ----------

// Replace with your values
import java.net.URLEncoder
val AccessKey = "***"
val SecretKey = "***"
val EncodedSecretKey = URLEncoder.encode(SecretKey, "UTF-8")
val AwsBucketName = "dataraj"
val MountName = "mnt/raj"


// COMMAND ----------

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

// COMMAND ----------

//Import Movie Data
val movies = sc.textFile(s"/$MountName/ml-100k/u.item")
movies.take(5)

// COMMAND ----------

//Lets get the titles together
 val titles = movies.map(line => line.split("\\|").take(2)).map(array
   => (array(0).toInt,array(1))).collectAsMap()
titles(50)

// COMMAND ----------

//Grabs the raw data
val rawData = sc.textFile(s"/$MountName/ml-100k/u.data")
rawData.first()

// COMMAND ----------

//Takes the first three values, user, item, and rating
val rawRatings = rawData.map(_.split("\t").take(3))
rawRatings.take(3)

// COMMAND ----------

//Map out the ratings
val ratings = rawRatings.map{ case Array(user, movie, rating) =>
  Rating(user.toInt, movie.toInt, rating.toDouble) }
   ratings.cache

// COMMAND ----------

//ALS Model, train with rank of 50, 10 iterations, and a lambda parameter of 0.01
val alsModel = ALS.train(ratings, 40, 12, 0.1)
alsModel.userFeatures

// COMMAND ----------

//Check what is in the model
alsModel.userFeatures.count

// COMMAND ----------

//Check what is in the model
alsModel.productFeatures.count

// COMMAND ----------

//Lets do a prediction
val predictedRating = alsModel.predict(689, 123)

// COMMAND ----------

// top 10 recommended items for user 689:
val userId = 689
val K = 15
val topKRecs = alsModel.recommendProducts(userId, K)

// COMMAND ----------



// COMMAND ----------

//Top rated movies for the chosen user
val moviesForUser = ratings.keyBy(_.user).lookup(689)
moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.
   product), rating.rating)).foreach(println)

// COMMAND ----------

//Top recommenendations for user
topKRecs.map(rating => (titles(rating.product), rating.rating)).
   foreach(println)


// COMMAND ----------

///Movie Similarity Analysis

// COMMAND ----------

val PRIOR_COUNT = 10
val PRIOR_CORRELATION = 0
val TRAIN_FILENAME = "ua.base"
val TEST_FIELNAME = "ua.test"
val MOVIES_FILENAME = "u.item"

// COMMAND ----------

// get movie names keyed on id
 val movies = sc.textFile(s"/$MountName/ml-100k/u.item")
      .map(line => {
        val fields = line.split("\\|")
        (fields(0).toInt, fields(1))
    })
 val movieNames = movies.collectAsMap()      // for local use to map id <-> movie name for pretty-printing

// COMMAND ----------

// extract (userid, movieid, rating) from ratings data
    val ratings = sc.textFile(s"/$MountName/ml-100k/ua.base")
      .map(line => {
        val fields = line.split("\t")
        (fields(0).toInt, fields(1).toInt, fields(2).toInt)
    })

// COMMAND ----------

// get num raters per movie, keyed on movie id
 val numRatersPerMovie = ratings
      .groupBy(tup => tup._2)
      .map(grouped => (grouped._1, grouped._2.size))

// COMMAND ----------

 // join ratings with num raters on movie id
 val ratingsWithSize = ratings
      .groupBy(tup => tup._2)
      .join(numRatersPerMovie)
      .flatMap(joined => {
        joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
    })


// COMMAND ----------

// ratingsWithSize now contains the following fields: (user, movie, rating, numRaters).

    // dummy copy of ratings for self join
    val ratings2 = ratingsWithSize.keyBy(tup => tup._1)

    // join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
    val ratingPairs =
      ratingsWithSize
      .keyBy(tup => tup._1)
      .join(ratings2)
      .filter(f => f._2._1._2 < f._2._2._2)

// COMMAND ----------

//Functions for similarity measures
 def correlation(size : Double, dotProduct : Double, ratingSum : Double,
                  rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = scala.math.sqrt(size * ratingNormSq - ratingSum * ratingSum) *
      scala.math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }

  /**
   * Regularize correlation by adding virtual pseudocounts over a prior:
   *   RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
   * where w = # actualPairs / (# actualPairs + # virtualPairs).
   */
  def regularizedCorrelation(size : Double, dotProduct : Double, ratingSum : Double,
                             rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double,
                             virtualCount : Double, priorCorrelation : Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  /**
   * The cosine similarity between two vectors A, B is
   *   dotProduct(A, B) / (norm(A) * norm(B))
   */
  def cosineSimilarity(dotProduct : Double, ratingNorm : Double, rating2Norm : Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }

  /**
   * The Jaccard Similarity between two sets A, B is
   *   |Intersection(A, B)| / |Union(A, B)|
   */
  def jaccardSimilarity(usersInCommon : Double, totalUsers1 : Double, totalUsers2 : Double) = {
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon / union
  }

// COMMAND ----------

  // compute raw inputs to similarity metrics for each movie pair
    val vectorCalcs =
      ratingPairs
      .map(data => {
        val key = (data._2._1._2, data._2._2._2)
        val stats =
          (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
            data._2._1._3,                // rating movie 1
            data._2._2._3,                // rating movie 2
            math.pow(data._2._1._3, 2),   // square of rating movie 1
            math.pow(data._2._2._3, 2),   // square of rating movie 2
            data._2._1._4,                // number of raters movie 1
            data._2._2._4)                // number of raters movie 2
        (key, stats)
      })
      .groupByKey()
      .map(data => {
        val key = data._1
        val vals = data._2
        val size = vals.size
        val dotProduct = vals.map(f => f._1).sum
        val ratingSum = vals.map(f => f._2).sum
        val rating2Sum = vals.map(f => f._3).sum
        val ratingSq = vals.map(f => f._4).sum
        val rating2Sq = vals.map(f => f._5).sum
        val numRaters = vals.map(f => f._6).max
        val numRaters2 = vals.map(f => f._7).max
        (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
      })

    // compute similarity metrics for each movie pair
    val similarities =
      vectorCalcs
      .map(fields => {
        val key = fields._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
        val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
          ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
        val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
        val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

        (key, (corr, regCorr, cosSim, jaccard))
      })


// COMMAND ----------

 // test a few movies out (substitute the contains call with the relevant movie name
    val sample = similarities.filter(m => {
      val movies = m._1
      (movieNames(movies._1).contains("Star Wars (1977)"))
    })

// COMMAND ----------

    // collect results, excluding NaNs if applicable
    val result = sample.map(v => {
      val m1 = v._1._1
      val m2 = v._1._2
      val corr = v._2._1
      val rcorr = v._2._2
      val cos = v._2._3
      val j = v._2._4
      (movieNames(m1), movieNames(m2), corr, rcorr, cos, j)
    }).collect().filter(e => !(e._4 equals Double.NaN)).sortBy(elem => -elem._4).take(10)    // test for NaNs must use equals rather than ==
   // .sortBy(elem => elem._4).take(10)

// COMMAND ----------

    // print the top 10 out
result.foreach(r => println(r._1 + " | " + r._2 + " | " + r._3.formatted("%2.4f") + " | "
  + r._4.formatted("%2.4f")
  + " | " + r._5.formatted("%2.4f") + " | " + r._6.formatted("%2.4f")))

// COMMAND ----------


