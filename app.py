import streamlit as st  
import sklearn
import pickle 
import numpy as np 

# Loading the model from the storage to the code
with open('svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Title of the web app
st.title("Drug Prediction App")

age = st.slider("Age", min_value=1, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male","Female"])
bp = st.selectbox("Blood Pressure", ["Low","Normal","High"])
# bp = "Normal"
cl = st.selectbox("cholestrol level", ["Low","High"])
na_to_k = st.slider("na_to_k", min_value=1.0, max_value=30.0, value=15.0)


sex = 0 if sex == "Female" else 1
# if sex == "Female":
#     sex = 0
# else:
#     sex = 1
bp_mapping = {"Low":0, "Normal":1, "High": 2}
# bp = "Normal"
bp = bp_mapping[bp]
# bp =  1
cl_mapping = {"Low":0, "High": 1}
cl = cl_mapping[cl]

drug_mapping = {0:'drugA',1: 'drugB',
                2: 'drugC',3: 'drugX',4: 'drugY'}
if st.button("Predict button"):
    st.write(f"User input data is {age,sex,bp,cl,na_to_k}")
    # Age,Sex,BP,Cholesterol,Na_to_K
    drug = model.predict([[age,sex,bp,cl,na_to_k]])
    st.write(f"The suitable drug for patient is {drug[0]}")
    st.write(f"The suitable drug for patient is {drug_mapping[drug[0]]}")
    