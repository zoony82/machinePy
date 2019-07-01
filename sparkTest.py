import pyspark
spark = pyspark.sql.SparkSession.builder.appName("pysaprk_python").getOrCreate()
data = [1,2,3,4,5]

df = spark.read.text("bagOfWords_2.py")
df.show()

List(1,2,3)
