import streamlit as st
import joblib
import numpy as np

model = joblib.load('best_model_for_customer_purchase.pkl')
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Purchase Behavior Prediction")
st.write("This application predicts whether a customer is a **High Spender** or **Low Spender** "
    "based on their purchase behavior.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
income = st.number_input("Annual Income", min_value=0, value=50000)
purchase_frequency = st.number_input("Purchase Frequency", min_value=1, value=5)

if st.button('Predict Customer Type'):
  data = np.array([[age,income,purchase_frequency]])
  predictions = model.predict(data)[0]

  if predictions[0] == 1:
    st.success("The customer is a **High Spender**")
    st.write('Customer is likely to spend more so they can be targeted for premium offers!')
  else:
    st.success("The customer is a **Low Spender**")
    st.write('Customer is likely to spend less so they can be targeted for regular offers!')
