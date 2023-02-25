import streamlit as st
import pandas as pd
from utils import _intialize_spark
from predict import predict_churn
from predict import modelBuilding
import numpy as np
import base64
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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
# def uplaodData(data):
#     df = pd.read_csv(data)
#     result = read_data(df)
#     return result

def main():
    # st.title("Customer Churn Prediction")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:yellow;text-align:center;">Customers Churn Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activites=['EDA', 'Plot', 'Prediction','About']
    choices = st.sidebar.selectbox('Select Activity',activites)
    
    if(choices=='Prediction'):
        choice = st.sidebar.selectbox("How would you like to predict:", ['Online', 'Batch'])
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
                    <div style="background-color:#F6EFEF;padding:10px;" >
                    <h2 style="color:red;text-align:center;"> A churn customer</h2>
                    </div>
                    """
            no_churn_html = """  
                    <div style="background-color:#F0F6EF;padding:10px;" >
                    <h2 style="color:green;text-align:center;"> not a churn customer</h2>
                    </div>
                    """
            recommendation = """  
                    <div style="background-color:#F0F6EF;padding:10px;" >
                    <h2 style="color:green;font-family:Algerian;font-size: 20px;"><u> recommendation: </u></h2>
                    </div>
                    """
            recommendation1 = """  
                    <div style="background-color:#F0F6EF;padding:10px;margin-top:-25px;" >
                    <h2 style="color:black ;text-align:center; font-size:20px;"> Give some offer on International plan and reduce some charges of calling also</h2>
                    </div>
                    """
            recommendation2 = """  
                    <div style="background-color:#F0F6EF;padding:10px;margin-top:-25px;" >
                    <h2 style="color:black ;text-align:center;font-size:20px;"> Network coverage is poor in this state Improve network coverage</h2>
                    </div>
                    """
            recommendation3 = """  
                    <div style="background-color:#F0F6EF;padding:10px;margin-top:-25px;" >
                    <h2 style="color:black ;text-align:center;font-size:20px;"> Improve the service of call center, by taking descrete feedback from the customers</h2>
                    </div>
                    """
            recommendation4 = """  
                    <div style="background-color:#F0F6EF;padding:10px;margin-top:-25px;" >
                    <h2 style="color:black ;text-align:center;font-size:20px;"> Improve the network quality in this area</h2>
                    </div>
                    """

            if st.button('Predict'):
                from predict import onlineDataTosparkDataFrame
                spark_df = onlineDataTosparkDataFrame(data, col)
                result = predict_churn(spark_df)
                output = result[0]
                prob = result[1]
                # st.success('The probability of customer being churned is {}'.format(prob))
                churn_state = ['NJ', 'CA', 'WA', 'MD', 'MT', 'OK', 'NV', 'SC', 'TX', 'MS', 'ME', 'MN']
                if output == 1.0:
                    st.markdown(churn_html, unsafe_allow_html= True)
                    st.write(recommendation,unsafe_allow_html=True)
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
                res = modelBuilding(df)
                result = res[0]
                # st.dataframe(result) 
                st.write(result.toPandas())
                csv_result = result.toPandas().to_csv(index=False).encode('utf-8')
                st.download_button(label='Download File', data = csv_result,file_name='Predicted_result_file.csv')
    elif(choices=='EDA'):
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Upload Dataset", type='CSV')
        if data is not None:
            from predict import dfToSparkDataFrame
            df = pd.read_csv(data)
            res = dfToSparkDataFrame(df)
            result = res
            st.write(result.toPandas())
            csv_result = result.toPandas().to_csv(index=False).encode('utf-8')
            if st.checkbox('Show shape'):
                st.write((result.count(), len(result.columns)))
            if st.checkbox('Show Columns'):
                st.write(result.columns)
            if st.checkbox('Show Summary'):
                st.write(result.toPandas().describe())
            if st.checkbox('Show total missing cell column wise:'):
                from predict import countNullValues
                output = countNullValues(result)
                st.write(output.toPandas())
            if st.checkbox('Show Total duplicate Records'):
                from predict import showDuplicatesRecords
                output = showDuplicatesRecords(result)
                if(output.count() > 0):
                    st.write('total duplicate records = ', result.count()-output.count())
                    st.write(output.toPandas())
                else :
                    st.write('No any duplicates record in this dataset')
    elif choices=='Plot':
        st.subheader("Data Visualization")
        data = st.file_uploader("Upload Dataset", type='CSV')
        if data is not None:
            from predict import dfToSparkDataFrame
            df = pd.read_csv(data)
            res = dfToSparkDataFrame(df)
            result = res
            st.write(result.toPandas())
            if st.checkbox('Correlation'):
                plt.figure(figsize=(15,12))
                st.write(sns.heatmap(result.toPandas().corr(),annot=True,cmap='inferno'))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            if st.checkbox("Pie Chart"):
                all_columns = result.columns
                columns_to_plot = st.selectbox("Select Column",all_columns)
                pie_plot = px.pie(result.groupBy(columns_to_plot).count().toPandas(), 'count')
                st.write(pie_plot)
                st.pyplot()
            all_columns = result.columns
            type_of_plot=st.selectbox('Select type of plot',['bar','histogram','scatter','box'])
            selected_columns = st.multiselect('Select columns to Plot',all_columns, default= 'churn' if all_columns.count('churn') >0 else None)
            if st.button('Generate Plot'):
                st.success('Generating {} plot for {}'.format(type_of_plot,selected_columns))
                if type_of_plot == 'histogram':
                #    column_selected = result.select(selected_columns).toPandas()
                #    st.bar_chart(column_selected)
                    choosen_columns = []
                    for val in selected_columns:
                        if val == 'chrun':
                            pass
                        else :
                            choosen_columns.append(val)
                    fig = px.histogram(result.toPandas(),choosen_columns, color='churn' if selected_columns.count('churn')>0 else None, barmode='group',text_auto=True,color_discrete_sequence = ['green','red'])
                    st.write(fig)
                    st.pyplot()
                if type_of_plot == 'bar':
                #    column_selected = result.select(selected_columns).toPandas()
                #    st.bar_chart(column_selected)
                    choosen_columns = []
                    for val in selected_columns:
                        if val == 'chrun':
                            pass
                        else :
                            choosen_columns.append(val)
                    fig = px.bar(result.toPandas(),choosen_columns, color='churn' if selected_columns.count('churn')>0 else None, barmode='group',text_auto=True,color_discrete_sequence = ['green','red'])
                    st.write(fig)
                    st.pyplot()
                if type_of_plot=='scatter':
                    fig = px.scatter(result.toPandas(), selected_columns,color_discrete_sequence = ['green','red'])
                    st.write(fig)
                    st.pyplot()
                if type_of_plot == 'box':
                    choosen_columns = []
                    for val in selected_columns:
                        if val == 'chrun':
                            pass
                        else :
                            choosen_columns.append(val)
                    fig = px.box(result.toPandas(), choosen_columns, points="all")
                    if len(choosen_columns)==1:
                        fig = px.box(result.toPandas(), x=choosen_columns[0], points="all")
                    if(len(choosen_columns)==2):
                        fig = px.box(result.toPandas(), x=choosen_columns[0], y=choosen_columns[1], points="all")
                    st.write(fig)
                    st.pyplot()



    st.sidebar.subheader("Designed By")
    st.sidebar.text("Md Owais")
            

if __name__=='__main__':
    main()
