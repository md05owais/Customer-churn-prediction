import numpy as np
import pandas as pd
import findspark
findspark.init()
from pyspark import SparkConf
from pyspark.sql import SparkSession

def _intialize_spark() -> SparkSession:
    conf = SparkConf().setAppName('Customers Churn Prediction').setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext
