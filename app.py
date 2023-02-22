import streamlit as st
import numpy as np
def main():
    st.title("Prediction of churn customers")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:yellow;text-align:center;">Churn Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.subheader("Designed By")
    st.sidebar.text("Md Owais")

    State = st.selectbox('Select State', ['AZ','SC','LA','MN','NJ','DC','OR','VA','RI','WY','KY','NH','MI','NV','WI','ID','CA','NE','CT','MT','NC','VT','MD','DE','MO','IL','ME','WA','ND','MS','AL','IN','OH','TN','IA','NM','PA','SD','NY','TX','WV','GA','MA','KS','FL','CO','AK','AR','OK','UT','HI'])

    account_length = st.slider('Select total active month', 1, 250)

    area_code = st.selectbox('Enter area code', [415,510,408])
    # Geo = int(le1_pik.transform([Geography]))

    international_plan = st.selectbox('Is User has international plan ?', ['yes', 'no'])
    # Gen = int(le_pik.transform([Gender]))
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
    

    churn_html = """  
              <div style="background-color:#F6EFEF;padding:10px >
               <h2 style="color:red;text-align:center;"> A churn customer</h2>
               </div>
            """
    no_churn_html = """  
              <div style="background-color:#F0F6EF;padding:10px >
               <h2 style="color:green ;text-align:center;"> not a churn customer</h2>
               </div>
            """
    recommendation = """  
              <div style="background-color:#F0F6EF;padding:10px >
               <h2 style="color:green ;text-align:center;"> Give some offer on International plan</h2>
               </div>
            """

    if st.button('Predict'):
        # output = predict_churn(CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        output = 0.6
        st.success('The probability of customer being churned is {}'.format(output))
        st.balloons()

        if output >= 0.5:
            st.markdown(churn_html, unsafe_allow_html= True)
            if international_plan == 'yes':
                st.markdown(recommendation, unsafe_allow_html=True)

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)

main()
