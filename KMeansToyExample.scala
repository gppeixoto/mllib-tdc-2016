import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by gpp on 6/29/16.
  */
object KMeansToyExample {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf())

    val moonsDataset = sc
      .textFile("./input/kmeans_data.txt")
      .map(s => Vectors.dense(s.split(",").map(_.toDouble)))
      .cache

    val kmeans = KMeans.train(moonsDataset, 2, 200)
    val wssse = kmeans.computeCost(moonsDataset)
    println(s"Within Set SSE: $wssse")
  }
}
