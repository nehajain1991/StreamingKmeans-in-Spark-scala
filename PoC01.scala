// To run this code create two folders train and test and put the training data file in the train folder and put the test data in the test folder
// Execute this script using :load PoC01.scala in spark-shell or spark-shell -i PoC01.scala
// to view the UI output check http://10.0.2.15:4040/streaming/ 
// You need to run sc.stop() before running this script
// Importing libraries
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming.{Seconds, StreamingContext}

//As you add text to `train.txt` the clusters will continuously update.
//Anytime you add text to `test.txt`, you'll see predicted labels using the current model.
//We are reading the stream in batches taken in every 3 seconds

//Generating spark content and configuration
val conf = new SparkConf().setAppName("StreamingKMeansExample")
val ssc = new StreamingContext(conf, Seconds(3))

//Convert input data to Vectors and LabeledPoints for the MLLib functions we will use
val trainingData = ssc.textFileStream("/home/user/train/").map(Vectors.parse).cache()
val testData = ssc.textFileStream("/home/user/test/").map(LabeledPoint.parse)
println()

//See the training data as it is received 
trainingData.print()

//Build a K-means clustering model for 3 clusters and 2 features 
val model = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(2, 0.0)
println()

//Training and testing the model 
model.trainOn(trainingData)
//As test data is received, we'll keep refining our clustering model and printing out the
//results.
model.predictOnValues(testData.map(lp => (lp.label.toInt, lp.features))).print()

//Start the process
ssc.checkpoint("/home/user/kmeans/")
ssc.start()
ssc.awaitTermination()
println()
