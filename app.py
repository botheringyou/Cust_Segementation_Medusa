import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan encoder
model = joblib.load("customer_segmentation_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Customer Segmentation Predictor", layout="wide")
st.title("🔍 Customer Segmentation Prediction App")

# Sidebar menu
menu = st.sidebar.selectbox("Pilih Mode", ["Upload File", "Input Manual"])

# Input kolom
input_columns = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
                 'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1']

if menu == "Upload File":
    st.header("Prediksi dari File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)
        pred = model.predict(df)
        df['Predicted_Segment'] = label_encoder.inverse_transform(pred)

        st.success("Prediksi berhasil!")
        st.dataframe(df.head())

        # Visualisasi
        st.subheader(" Distribusi Segmentasi yang Diprediksi")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Predicted_Segment', order=sorted(df['Predicted_Segment'].unique()), ax=ax)
        st.pyplot(fig)

        # Unduh hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Unduh Hasil Prediksi", csv, "prediksi_segmentasi.csv", "text/csv")

elif menu == "Input Manual":
    st.header("✍️ Prediksi dari Input Manual")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            age = st.slider("Age", 18, 90, 30)
        with col2:
            graduated = st.selectbox("Graduated", ["Yes", "No"])
            profession = st.selectbox("Profession", ["Healthcare", "Engineer", "Lawyer", "Artist", "Executive", "Doctor", "Entertainment", "Marketing"])
            work_exp = st.number_input("Work Experience", 0.0, 30.0, step=1.0)
        with col3:
            spending_score = st.selectbox("Spending Score", ["Low", "Average", "High"])
            family_size = st.number_input("Family Size", 1.0, 20.0, step=1.0)
            var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                'Gender': gender,
                'Ever_Married': ever_married,
                'Age': age,
                'Graduated': graduated,
                'Profession': profession,
                'Work_Experience': work_exp,
                'Spending_Score': spending_score,
                'Family_Size': family_size,
                'Var_1': var_1
            }])

            pred = model.predict(input_df)
            segment = label_encoder.inverse_transform(pred)[0]

            st.success(f"Prediksi Segmentasi: 🧩 **{segment}**")
