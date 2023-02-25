import numpy as np
import base64
import math
from utils import _intialize_spark
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassificationModel
spark, _ = _intialize_spark()
# import findspark
# findspark.init()
# import pyspark
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
# from pyspark.ml.classification import RandomForestClassificationModel

# importing pipline and final_model
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml import PipelineModel
pipeline = PipelineModel.load('./model/pipeline')
model = CrossValidatorModel.read().load('./model/final_model')
rf_model = RandomForestClassificationModel.read().load('./model/rf_model')

def onlineDataTosparkDataFrame(data, col):
     df=spark.createDataFrame(data,col)
     return df
def dfToSparkDataFrame(data):
    df = spark.createDataFrame(data)
    return df
def predict_churn(df):
    # df=spark.createDataFrame(data,col)
    model_df = pipeline.transform(df)
    pred = model.transform(model_df)
    prob=pred.select('probability').collect()[0][0][0]
    prob = math.ceil(prob * 100)/100
    output = pred.select('prediction').collect()[0][0]
    return (output, prob)
def modelBuilding(data):
    from pyspark.sql.functions import expr, substring, when, col
    df = spark.createDataFrame(data)
    df1=df.drop('id')
    df1=df1.withColumn('area_code', expr("substring(area_code,11,12)"))
    model_df = pipeline.transform(df1)
    final_res = model.transform(model_df)

    predicted_df = final_res.select('state','account_length','area_code','international_plan','voice_mail_plan','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge','total_eve_minutes','total_eve_calls','total_eve_charge','total_night_minutes','total_night_calls','total_night_charge','total_intl_minutes','total_intl_calls','total_intl_charge','number_customer_service_calls','prediction')
    predicted_df = predicted_df.withColumnRenamed('prediction', 'churn')
    predicted_df = predicted_df.withColumn('churn', when(predicted_df['churn'] == 0, 'no').otherwise('yes'))
    return (predicted_df,df)
def countNullValues(df):
    from pyspark.sql.functions import isnan, count, when, col, expr, substring
    total_col = df.columns
    df_with_total_missing_cell = df.select([count(when( col(c).contains('NA') | col(c).contains('None') \
                      | (col(c) == "NaN") | (col(c) == '') | (col(c) == 'NULL')\
                      | col(c).isNull() | isnan(c) ,c)).alias(c) for c in total_col])
    return df_with_total_missing_cell
def showDuplicatesRecords(df):
    from pyspark.sql.functions import isnan, count, when, col, expr, substring
    df_duplicates = df.groupBy(df.columns).count().filter(col("count")>1)
    return df_duplicates
