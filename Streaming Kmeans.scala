// Databricks notebook source exported at Fri, 4 Sep 2015 01:35:05 UTC
// MAGIC %md # **Streaming POC**
// MAGIC 
// MAGIC This is a POC of doing clustering/anomaly detection using Spark Streaming on 1 second batches of streaming data. 
// MAGIC 
// MAGIC It uses an in-memory random number generator as a dummy source for streaming data. 

// COMMAND ----------

// Replace with your values
import java.net.URLEncoder
val AccessKey = "***"
val SecretKey = "***"
val EncodedSecretKey = URLEncoder.encode(SecretKey, "UTF-8")
val AwsBucketName = "dataraj"
val MountName = "mnt/raj"

//dbutils.fs.mount(s"s3n://$AccessKey:$EncodedSecretKey@$AwsBucketName", s"/$MountName")

// COMMAND ----------

//Example of Labeled points to use in the model
val myRDD = sc.textFile(s"/$MountName/kmeans/kmeanssample4.csv").map(LabeledPoint.parse)
myRDD.take(5)
val parsed = myRDD.map{x => (x.features)}.map(p => new Person(p(0), p(1))).toDF()
parsed.take(5)
parsed.show()

// COMMAND ----------

//Example of training data 
import org.apache.spark.mllib.linalg.Vectors
import sqlContext.implicits._
case class Person(x: Double, y: Double)
val myRDD2 = sc.textFile(s"/$MountName/kmeans/kmeanstrain.csv")
myRDD2.take(5)
val parsed = myRDD2.map {x => Vectors.dense(x.split(',').slice(0,2).map(_.toDouble))}
parsed.take(5)
//val df1 = myRDD2.map(line => line.split(",")).map(p => new Person(p(0).trim.tryGetInt, p(1).trim.toInt)).toDF()
val df1 = parsed.map(p => new Person(p(0), p(1))).toDF()
  df1.registerTempTable("df1")
  df1.printSchema()
//df1.select("y").show()
df1.show()

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.storage._
import org.apache.spark.streaming._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.clustering.StreamingKMeansModel




// COMMAND ----------

// MAGIC %md ## Configurations
// MAGIC 
// MAGIC Configurations that control the streaming app in the notebook

// COMMAND ----------


// === Configuration to control the flow of the application ===
val stopActiveContext = true	 
// "true"  = stop if any existing StreamingContext is running;              
// "false" = dont stop, and let it run undisturbed, but your latest code may not be used

// === Configurations for Spark Streaming ===
val batchIntervalSeconds = 1 
val eventsPerSecond = 10    // For the dummy source

// Verify that the attached Spark cluster is 1.4.0+
require(sc.version.replace(".", "").toInt >= 140, "Spark 1.4.0+ is required to run this notebook. Please attach it to a Spark 1.4.0+ cluster.")


// COMMAND ----------

// MAGIC %md ## Setup: This is the function for the StreamingContext
// MAGIC 
// MAGIC It does two things. 
// MAGIC * Define a custom receiver as the dummy source 
// MAGIC * Define the function that creates and sets up the streaming computation (this is the main logic)

// COMMAND ----------

// This is the dummy source implemented as a custom receiver. No need to understand this.

import scala.util.Random
import org.apache.spark.streaming.receiver._

class DummySource(ratePerSec: Int) extends Receiver[String](StorageLevel.MEMORY_AND_DISK_2) {

  def onStart() {
    // Start the thread that receives data over a connection
    new Thread("Dummy Source") {
      override def run() { receive() }
    }.start()
  }

  def onStop() {
   // There is nothing much to do as the thread calling receive()
   // is designed to stop by itself isStopped() returns false
  }

  /** Create a socket connection and receive data until receiver is stopped */
  //Creates labeled points with in two dimensions x1 and x2
  private def receive() {
    while(!isStopped()) {      
      store(Random.nextInt(3) + "," + Random.nextInt(8) + " " + Random.nextInt(8))
      //store ("1,0 3")
      Thread.sleep((1000.toDouble / ratePerSec).toInt)
    }
  }
}

// COMMAND ----------

// MAGIC %md This section trains a K-Means and then does predictions on the streaming data

// COMMAND ----------

var newContextCreated = true      // Flag to detect whether new context was created or not
case class Person(x: Double, y: Double)
case class Person2(x: Double, y: Double)

// Function to create a new StreamingContext and set it up
def creatingFunc(): StreamingContext = {
    
  // Create a StreamingContext
  val ssc = new StreamingContext(sc, Seconds(batchIntervalSeconds))
  
  // Create a stream that generates 1000 lines per second
  val stream = ssc.receiverStream(new DummySource(eventsPerSecond)).map(LabeledPoint.parse)
  val streamtest = stream.map(x => (x.features))
  
 // stream.foreachRDD { rdd =>
    //System.out.println("# events = " + rdd.count())
   // System.out.println("\t " + rdd.take(10).mkString(", ") + ", ...")
//  }
 //val trainingData = ssc.textFileStream((s"/$MountName/kmeans/kmeanssample5.csv")).map {x=>Vectors.dense(x.split(',').slice(0,2).map(_.toDouble))}
  val testData = ssc.textFileStream((s"/$MountName/kmeans/kmeanssample4.csv")).map(LabeledPoint.parse)
  val trainingData = ssc.textFileStream((s"/$MountName/kmeans/kmeanstrain.csv")).map {x=>Vectors.dense(x.split(',').slice(0,2).map(_.toDouble))}
  
  val numDimensions = 2
  val numClusters = 3
  val model = new StreamingKMeans()
  .setK(numClusters)
  .setDecayFactor(1.0)
  .setRandomCenters(numDimensions, 0.0)
  model.trainOn(trainingData)
 // model.predictOnValues(stream.map(lp => (lp.label, lp.features))).print()

   // Create temp table at every batch interval
  
//trainingData.foreachRDD(rdd => {rdd.map(p => Person(p(0), p(1)))}.toDF().registerTempTable("training"))
streamtest.foreachRDD(rdd => {rdd.map(t => Person2(t(0), t(1)))}.toDF().registerTempTable("testing"))
  //map(p => Person2(p(0), p(1)))
  //.foreachRDD(rdd => {rdd.map(p => Person2(p(0), p(1)))})
  //.toDF().registerTempTable("test")) 
  //  
  
  ssc.remember(Minutes(1))  // To make sure data is not deleted by the time we query it interactively
  
  println("Creating function called to create new StreamingContext")
  newContextCreated = true  
  ssc
}

// COMMAND ----------

// MAGIC %md ## Start/Restart: Stop existing StreamingContext if any and start/restart the new one
// MAGIC 
// MAGIC Here we are going to use the configuration at the top of the notebook to decide whether to stop any existing StreamingContext, and start a new one, or recover one from existing checkpoints.

// COMMAND ----------

// Stop any existing StreamingContext 
if (stopActiveContext) {	
  StreamingContext.getActive.foreach { _.stop(stopSparkContext = true) }
} 
val ssc = StreamingContext.getActiveOrCreate(creatingFunc)
if (newContextCreated) {
  println("New context created from currently defined creating function") 
} else {
  println("Existing context running or recovered from checkpoint, may not be running currently defined creating function")
}

ssc.start()
ssc.awaitTerminationOrTimeout(batchIntervalSeconds * 5 * 1000)


// COMMAND ----------

//Use this to stop the streaming context
StreamingContext.getActive.foreach { _.stop(stopSparkContext = false) }

// COMMAND ----------

// MAGIC %md ## Interactive Querying
// MAGIC 
// MAGIC Now let's try querying the table. You can run this command again and again, you will find the numbers changing.

// COMMAND ----------

// MAGIC %sql select * from train

// COMMAND ----------


