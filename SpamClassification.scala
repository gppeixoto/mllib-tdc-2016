import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by gpp on 6/29/16.
  */
object SpamClassification {
  val SPAM = "spam"
  val HAM = "ham"

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf())
    val sqlContext = new SQLContext(sc)

    // Dataset available at:
    // https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    val dataRDD:RDD[(Long, String, Double)] = sc
      .textFile("./input/SMSSpamCollection")
      .zipWithIndex
      .map({case (line, index) =>
        val label = if (line.startsWith(SPAM)) 1.0 else 0.0
        val text = line.substring(if (line.startsWith(SPAM)) 4 else 3)
        (index, text, label)
      })

    val Array(trainingDF, testDF) = sqlContext
      .createDataFrame(dataRDD)
      .toDF("id", "text", "label")
      .randomSplit(Array(0.8, 0.2))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val estimator = new LogisticRegression()
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, estimator))
    val model = pipeline.fit(trainingDF)

    val accuracy = 1.0 * model.transform(testDF)
      .filter("prediction = label")
      .count / testDF.count

    println("Accuracy: %.3f on %d training samples and %d test samples"
      .format(accuracy, trainingDF.count, testDF.count)
    )
  }
}
