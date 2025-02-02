# import libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# load the trained model
model = tf.keras.models.load_model('model.h5')

# load encoders and scalers
with open('label_en_gender.pkl', 'rb') as file:
    label_en_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Estimate Salary Prediction')

# user input
# user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_en_gender.classes_)
age = st.slider('age', 18, 90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# preprocess the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],  
    'Gender': [label_en_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [exited]
})

# one hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with the input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

# predict estimated salary
prediction = model.predict(input_data_scaled)
prediction = prediction[0][0]

st.write(f'Estimated Salary: ${prediction:.2f}')

