import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Aadhaar Drishti",
    layout="wide"
)

st.title("🔍 Aadhaar Drishti Dashboard")
st.markdown("AI + ML based Aadhaar Analytics and Fraud Detection System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("Select Column for Visualization")

        selected_col = st.selectbox(
            "Choose Numeric Column",
            numeric_cols
        )

        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna(), bins=20)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

        st.subheader("Correlation Matrix")
        corr = df[numeric_cols].corr()
        st.dataframe(corr)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        cax = ax2.matshow(corr)
        fig2.colorbar(cax)
        st.pyplot(fig2)

    st.subheader("Fraud Detection Summary")

    suspicious_rows = df[df.isnull().sum(axis=1) > 2]

    st.write(f"Potential suspicious records detected: {len(suspicious_rows)}")

    if len(suspicious_rows) > 0:
        st.dataframe(suspicious_rows.head())

else:
    st.info("Please upload a dataset to begin analysis.")