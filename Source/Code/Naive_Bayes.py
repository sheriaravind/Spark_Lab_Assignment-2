from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import os
os.environ["SPARK_HOME"] = r"V:\Softwares\Spark\spark-2.3.1-bin-hadoop2.7"
os.environ["HADOOP_HOME"]= r"V:\Softwares\Spark\bin\winutils.exe"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_161"

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

data = spark.read.load(r"A:\Summer-18\Big Data\Spark\Lab-4\Absenteeism_at_work_AAA\Absenteism_at_work.csv", format="csv", header=True, delimiter=";")
data = data.withColumn("MOA", data["Month of absence"] - 0).withColumn("label", data['Seasons'] - 0). \
    withColumn("ROA", data["Reason for absence"] - 0). \
    withColumn("distance", data["Distance from Residence to Work"] - 0). \
    withColumn("BMI", data["Body mass index"] - 0)
#data.show()
assem = VectorAssembler(inputCols=["BMI", "distance"], outputCol='features')

data = assem.transform(data)
# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")


# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
#predictions.show()


# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test set accuracy = " + str(accuracy))