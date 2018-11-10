//Secion
import org.apache.spark.sql.SparkSession
//reportar errores reciduos
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//instancia de la secion
val spark = SparkSession.builder().getOrCreate()
// libreria KMeans
import org.apache.spark.ml.clustering.KMeans
//se carga el dataset
 val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")

//seleccionar columnas
val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")



   //importar VectorAssembler
   import org.apache.spark.ml.feature.VectorAssembler
   import org.apache.spark.ml.linalg.Vectors

    val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

val training_data = assembler.transform(feature_data).select("features")



 val kmeans = new KMeans().setK(3).setSeed(1L)
 val model = kmeans.fit(training_data)


 val WSSE = model.computeCost(training_data)
println(s"Within set sum of Squared Errors = $WSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
