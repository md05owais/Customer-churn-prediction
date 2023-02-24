import streamlit as st
import base64
import pandas as pd
# from predict import predict_churn
# from predict import read_data
import numpy as np
import base64
import math
# rf_model = RandomForestClassificationModel.read().load('./model/rf_model')

def predict_churn(data, col):
    import findspark
    findspark.init()
    import pyspark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
    # from pyspark.ml.classification import RandomForestClassificationModel

    # importing pipline and final_model
    from pyspark.ml.tuning import CrossValidatorModel
    from pyspark.ml import PipelineModel
    pipeline = PipelineModel.load('./model/pipeline')
    model = CrossValidatorModel.read().load('./model/final_model')
    df=spark.createDataFrame(data,col)
    model_df = pipeline.transform(df)
    pred = model.transform(model_df)
    prob=pred.select('probability').collect()[0][0][0]
    prob = math.ceil(prob * 100)/100
    output = pred.select('prediction').collect()[0][0]

    return (output, prob)
def read_data(data):
    import findspark
    findspark.init()
    import pyspark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
    # from pyspark.ml.classification import RandomForestClassificationModel

    # importing pipline and final_model
    from pyspark.ml.tuning import CrossValidatorModel
    from pyspark.ml import PipelineModel
    pipeline = PipelineModel.load('./model/pipeline')
    model = CrossValidatorModel.read().load('./model/final_model')
    from pyspark.sql.functions import expr, substring, when, col
    df = spark.createDataFrame(data)
    df=df.drop('id')
    df=df.withColumn('area_code', expr("substring(area_code,11,12)"))
    model_df = pipeline.transform(df)
    final_res = model.transform(model_df)

    predicted_df = final_res.select('state','account_length','area_code','international_plan','voice_mail_plan','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge','total_eve_minutes','total_eve_calls','total_eve_charge','total_night_minutes','total_night_calls','total_night_charge','total_intl_minutes','total_intl_calls','total_intl_charge','number_customer_service_calls','prediction')
    predicted_df = predicted_df.withColumnRenamed('prediction', 'churn')
    predicted_df = predicted_df.withColumn('churn', when(predicted_df['churn'] == 0, 'no').otherwise('yes'))
    return predicted_df

# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('./data/true.png')

def main():
    # st.title("Customer Churn Prediction")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:yellow;text-align:center;">Customer Churn Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    choice = st.sidebar.selectbox("How do you want to predict", ['Online', 'Batch'])
    st.sidebar.subheader("Designed By")
    st.sidebar.text("Md Owais")
    if(choice == 'Online'):
        state = st.selectbox('Select State', ['AZ','SC','LA','MN','NJ','DC','OR','VA','RI','WY','KY','NH','MI','NV','WI','ID','CA','NE','CT','MT','NC','VT','MD','DE','MO','IL','ME','WA','ND','MS','AL','IN','OH','TN','IA','NM','PA','SD','NY','TX','WV','GA','MA','KS','FL','CO','AK','AR','OK','UT','HI'])
        account_length = st.slider('Select total active month', 1, 250)
        area_code = st.selectbox('Enter area code', ['415','510','408'])
        international_plan = st.selectbox('Is User has international plan ?', ['yes', 'no'])
        voice_mail_plan = st.selectbox('Is user has voice mail plan ?', ['yes', 'no'])
        number_vmail_messages = st.number_input('Enter Total Number of voice mail',min_value=0)
        total_day_minutes = st.number_input('Enter Total day minutes',min_value=0.0)
        total_day_calls = st.number_input('Enter Total day calls',min_value=0)
        total_day_charge = st.number_input('Enter Total day charge',min_value=0.0)
        total_eve_minutes = st.number_input('Enter Total evening minutes',min_value=0.0)
        total_eve_calls = st.number_input('Enter Total evening calls',min_value=0)
        total_eve_charge = st.number_input('Enter Total evening charge',min_value=0.0)
        total_night_minutes = st.number_input('Enter Total night minutes',min_value=0.0)
        total_night_calls = st.number_input('Enter Total night calls',min_value=0)
        total_night_charge = st.number_input('Enter Total night charge',min_value=0.0)
        total_intl_minutes = st.number_input('Enter Total international minutes',min_value=0.0)
        total_intl_calls = st.number_input('Enter Total international calls',min_value=0)
        total_intl_charge = st.number_input('Enter Total international charge',min_value=0.0)
        number_customer_service_calls = st.number_input('Enter Total customer service calls',min_value=0)

        data = [(state, account_length, area_code, international_plan, voice_mail_plan, number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge, total_eve_minutes,total_eve_calls, total_eve_charge, total_night_minutes,total_night_calls,total_night_charge,total_intl_minutes,total_intl_calls,total_intl_charge,number_customer_service_calls)]

        col = ['state','account_length','area_code','international_plan','voice_mail_plan','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge','total_eve_minutes','total_eve_calls','total_eve_charge','total_night_minutes','total_night_calls','total_night_charge','total_intl_minutes','total_intl_calls','total_intl_charge','number_customer_service_calls']

        # df = spark.createDataFrame(data, col)
        churn_html = """  
                <div style="background-color:#F6EFEF;padding:10px >
                <h2 style="color:green;text-align:center;"> A churn customer</h2>
                </div>
                """
        no_churn_html = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> not a churn customer</h2>
                </div>
                """
        recommendation = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> recommendation: </h2>
                </div>
                """
        recommendation1 = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> Give some offer on International plan</h2>
                </div>
                """
        recommendation2 = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> Network coverage is poor in this state Improve network coverage</h2>
                </div>
                """
        recommendation3 = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> Improve the service of call center, by taking descrete feedback from the customers</h2>
                </div>
                """
        recommendation4 = """  
                <div style="background-color:#F0F6EF;padding:10px >
                <h2 style="color:green ;text-align:center;"> Improve the network quality in this area</h2>
                </div>
                """

        if st.button('Predict'):
            result = predict_churn(data, col)
            output = result[0]
            prob = result[1]
            # st.success('The probability of customer being churned is {}'.format(prob))
            churn_state = ['NJ', 'CA', 'WA', 'MD', 'MT', 'OK', 'NV', 'SC', 'TX', 'MS', 'ME', 'MN']
            if output == 1.0:
                st.markdown(churn_html, unsafe_allow_html= True)
                st.markdown(recommendation,unsafe_allow_html=True)
                if international_plan == 'yes':
                    st.markdown(recommendation1, unsafe_allow_html=True)
                elif (churn_state.__contains__(state)):
                    st.markdown(recommendation2, unsafe_allow_html=True)
                elif(number_customer_service_calls >=4 ) :
                    st.markdown(recommendation3, unsafe_allow_html=True)
                else:
                    st.markdown(recommendation4, unsafe_allow_html=True)

            else:
                st.markdown(no_churn_html, unsafe_allow_html= True)
                st.balloons()
    else:
        data = st.file_uploader("Upload Dataset", type='CSV')
        if data is not None:
            df = pd.read_csv(data)
            result = read_data(df)
            st.dataframe(result) 
            csv_result = result.toPandas().to_csv(index=False).encode('utf-8')
            st.download_button(label='Download File', data = csv_result,file_name='Predicted_result_file.csv')


main()
