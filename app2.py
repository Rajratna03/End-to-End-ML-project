import streamlit as st
import pickle
import pandas as pd

# Load the trained Decision Tree model from the pickle file
with open('svc_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess user inputs
def preprocess_input(salary, gender, savings):
    # Perform any necessary preprocessing, such as label encoding
    # For gender, you might encode it as 0 for female and 1 for male
    gender_encoded = 1 if gender == 'Male' else 0
    return salary, gender_encoded, savings

# Function to make predictions
def predict_loan_acceptance(salary, gender, savings):
    # Preprocess the input
    processed_input = preprocess_input(salary, gender, savings)
    # Reshape the input for prediction (reshape as per your model's requirements)
    input_array = pd.DataFrame([processed_input], columns=['Salary', 'Gender', 'Savings'])
    # Make predictions
    prediction = model.predict(input_array)
    return prediction[0]  # Return the predicted label

# Streamlit app
def main():
    st.title('Loan Acceptance Prediction App')
    st.write('Enter employee details to predict loan acceptance')

    # Input fields for user
    salary = st.number_input('Salary', min_value=0)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    savings = st.number_input('Savings', min_value=0)

    # Predict when user clicks the button
    if st.button('Predict'):
        prediction = predict_loan_acceptance(salary, gender, savings)
        st.write(f'Loan acceptance prediction: {prediction}')

if __name__ == '__main__':
    main()
