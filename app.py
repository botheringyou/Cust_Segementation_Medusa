import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('customer_segmentation_model.pkl'))

# Create the Streamlit app
st.title('Customer Segmentation Predictor')

# Input form
st.header('Customer Details')
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
graduated = st.selectbox('Graduated', ['Yes', 'No'])
profession = st.selectbox('Profession', ['Artist', 'Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Doctor', 'Homemaker', 'Executive', 'Marketing', 'Unemployed'])
work_experience = st.number_input('Work Experience (years)', min_value=0, max_value=30)
family_size = st.number_input('Family Size', min_value=1, max_value=10)
spending_score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])

# Preprocessing function (simplified from your original code)
def preprocess_input(data):
    # Encode categorical variables
    data['Gender'] = 1 if data['Gender'] == 'Male' else 0
    data['Ever_Married'] = 1 if data['Ever_Married'] == 'Yes' else 0
    data['Graduated'] = 1 if data['Graduated'] == 'Yes' else 0

    # Profession one-hot encoding (simplified)
    professions = ['Artist', 'Healthcare', 'Engineer', 'Lawyer', 'Entertainment',
                  'Doctor', 'Homemaker', 'Executive', 'Marketing', 'Unemployed']
    for prof in professions:
        data[f'Profession_{prof}'] = 1 if data['Profession'] == prof else 0

    # Spending score encoding
    if data['Spending_Score'] == 'Low':
        data['Spending_Score_Low'] = 1
        data['Spending_Score_Average'] = 0
    elif data['Spending_Score'] == 'Average':
        data['Spending_Score_Low'] = 0
        data['Spending_Score_Average'] = 1
    else:
        data['Spending_Score_Low'] = 0
        data['Spending_Score_Average'] = 0

    # Drop original columns
    data.drop(['Profession', 'Spending_Score'], axis=1, inplace=True)

    return data

# Prediction button
if st.button('Predict Segmentation'):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ever_Married': [ever_married],
        'Graduated': [graduated],
        'Profession': [profession],
        'Work_Experience': [work_experience],
        'Family_Size': [family_size],
        'Spending_Score': [spending_score]
    })

    # Preprocess input
    processed_data = preprocess_input(input_data)

    # Ensure all expected columns are present
    expected_columns = ['Age', 'Gender', 'Ever_Married', 'Graduated', 'Work_Experience',
                      'Family_Size', 'Profession_Artist', 'Profession_Healthcare',
                      'Profession_Engineer', 'Profession_Lawyer', 'Profession_Entertainment',
                      'Profession_Doctor', 'Profession_Homemaker', 'Profession_Executive',
                      'Profession_Marketing', 'Profession_Unemployed', 'Spending_Score_Low',
                      'Spending_Score_Average']

    # Add missing columns with 0
    for col in expected_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0

    # Reorder columns
    processed_data = processed_data[expected_columns]

    # Make prediction
    prediction = model.predict(processed_data)

    # Map prediction to segment name
    segment_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    segment = segment_map.get(prediction[0], 'Unknown')

    st.success(f'Predicted Customer Segment: {segment}')
