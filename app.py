# 1 good(low risk) , 0 bad(high risk)

import pandas as pd
import streamlit as st  
import joblib

model=joblib.load('extratrees_credit_model.pkl')
encoders={col:joblib.load(f'le_{col}.pkl') for col in ['Sex','Housing','Saving accounts','Checking account','Purpose']}

st.title('Credit Risk Prediction app')
st.write('Enter the details below to predict credit risk:')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex',['male', 'female'])
job = st.number_input('Job (0-3)',min_value=0,max_value=3,value=1)
housing = st.selectbox('Housing',['own', 'free', 'rent'])
saving_accounts = st.selectbox('Saving accounts',['little', 'moderate', 'quite rich', 'rich'])
checking_account = st.selectbox('Checking account',['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit amount', min_value=0, value=1000)
duration = st.number_input('Duration (in months)', min_value=1, value=12)
purpose = st.selectbox('Purpose',['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'vacation/others', 'retraining'])

input_df = pd.DataFrame({
    'Sex': [encoders['Sex'].transform([sex])[0]],
    'Job': [job],
    'Housing': [encoders['Housing'].transform([housing])[0]],
    'Saving accounts': [encoders['Saving accounts'].transform([saving_accounts])[0]],
    'Checking account': [encoders['Checking account'].transform([checking_account])[0]],
    'Purpose': [encoders['Purpose'].transform([purpose])[0]],
    'Age': [age],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    
})

if st.button('Predict risk'):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success('The predicted credit risk is :  LOW (Good credit)') 
    else:
        st.error('The predicted credit risk is : HIGH (Bad credit)')