from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from collections import namedtuple
import os

os.environ["SPARK_HOME"] = r"V:\Softwares\Spark\spark-2.3.1-bin-hadoop2.7"
os.environ["HADOOP_HOME"]= r"V:\Softwares\Spark\bin\winutils.exe"

def Word_Count():
    sc = SparkContext(appName="PysparkStreaming")
    wordcount = {}
    ssc = StreamingContext(sc, 5)

    lines = ssc.socketTextStream("localhost", 5555)

    fields = ("word", "count")
    Tweet = namedtuple('Text', fields)

    # lines = socket_stream.window(20)
    counts = lines.flatMap(lambda text: text.split(" "))\
        .map(lambda x: (x, 1))\
        .reduceByKey(lambda a, b: a + b).map(lambda rec: Tweet(rec[0], rec[1]))
    counts.pprint()
    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":
    Word_Count()
