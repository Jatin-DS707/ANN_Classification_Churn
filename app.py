import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import pickle



## Load the trained Model
model=load_model('model.h5')

## Load Encoders and Scaler
with open('lbl_encoder.pkl','rb') as file:  ## Read Byte
     lbl_encoder=pickle.load(file)

with open('ohe.pkl','rb') as file:  ## Read Byte
     ohe=pickle.load(file)

with open('scaler.pkl','rb') as file:  ## Read Byte
     scaler=pickle.load(file)


## Streamlit App Developmnt
st.title("Customer Churn Prediction")

## User Input
CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography',ohe.categories_[0]) ## Need to check why added
Gender = st.selectbox('Gender',lbl_encoder.classes_) ## Need to check why added
Age = st.slider('Age',18,92)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input('Balance')
Num_Of_Products = st.slider('Number of Products',1,4)
HasCrCard = st.selectbox('Has Credit Card',[0,1])
IsActiveMember = st.selectbox('Is Active Member',[0,1])
EstimatedSalary = st.number_input('Estimated Salary')


## Prepare the input Data into Dictionary
input_data = pd.DataFrame({
     'CreditScore' : [CreditScore],
     'Geography' : [Geography],
     'Gender' : [lbl_encoder.transform([Gender])[0]],
     'Age' : [Age],
     'Tenure' : [Tenure],
     'Balance' : [Balance],
     'NumOfProducts' : [Num_Of_Products],
     'HasCrCard' : [HasCrCard],
     'IsActiveMember' : [IsActiveMember],
     'EstimatedSalary' : [EstimatedSalary]
})


## One-hot encode 'Geography'
encoded_array = ohe.transform(input_data[['Geography']])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['Geography']))



## Concatenating the data
input_data.drop('Geography',axis=1,inplace=True)
input_data = pd.concat([input_data, encoded_df], axis=1)


## Scaling the Data
input_data = scaler.transform(input_data)


## Prediction
prediction = model.predict(input_data)
prediction_prob = prediction[0][0]

st.write(prediction_prob)

if prediction_prob > 0.5:
    st.write("Customer is likely to churn.")
else :
    st.write("Customer is not likelt to churn.")



