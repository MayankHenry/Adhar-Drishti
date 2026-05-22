import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Aadhaar Drishti",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("🔍 Aadhaar Drishti Dashboard")
st.markdown(
    "AI + ML based Aadhaar Analytics, Multi-CSV Pattern Analysis and Fraud Detection System"
)

# =========================
# MULTIPLE FILE UPLOAD
# =========================

uploaded_files = st.file_uploader(
    "Upload Multiple CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

# =========================
# PROCESS FILES
# =========================

if uploaded_files:

    dfs = []

    for file in uploaded_files:
        try:
            df = pd.read_csv(file)

            # add source filename
            df["source_file"] = file.name

            dfs.append(df)

        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    # =========================
    # MERGE DATASETS
    # =========================

    combined_df = pd.concat(dfs, ignore_index=True)

    st.success(f"{len(uploaded_files)} files uploaded successfully!")

    # =========================
    # DATA PREVIEW
    # =========================

    st.subheader("📄 Combined Dataset Preview")
    st.dataframe(combined_df.head())

    # =========================
    # DATASET INFO
    # =========================

    st.subheader("📊 Dataset Information")

    st.write("Shape of Dataset:")
    st.write(combined_df.shape)

    st.write("Columns:")
    st.write(combined_df.columns.tolist())

    st.write("Statistical Summary:")
    st.write(combined_df.describe())

    # =========================
    # MISSING VALUES
    # =========================

    st.subheader("⚠ Missing Values Analysis")

    missing_values = combined_df.isnull().sum()

    st.dataframe(missing_values)

    # =========================
    # NUMERIC COLUMNS
    # =========================

    numeric_cols = combined_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # =========================
    # VISUALIZATION
    # =========================

    if numeric_cols:

        st.subheader("📈 Data Visualization")

        selected_col = st.selectbox(
            "Choose Numeric Column",
            numeric_cols
        )

        # histogram

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            combined_df[selected_col].dropna(),
            bins=20
        )

        ax.set_title(f"Distribution of {selected_col}")

        st.pyplot(fig)

        # =========================
        # CORRELATION MATRIX
        # =========================

        st.subheader("🔗 Correlation Matrix")

        corr = combined_df[numeric_cols].corr()

        st.dataframe(corr)

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        cax = ax2.matshow(corr)

        fig2.colorbar(cax)

        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_yticks(range(len(corr.columns)))

        ax2.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax2.set_yticklabels(corr.columns)

        st.pyplot(fig2)

        # =========================
        # Z-SCORE ANOMALY DETECTION
        # =========================

        st.subheader("🚨 Z-Score Fraud Detection")

        clean_numeric_df = combined_df[numeric_cols].fillna(0)

        z_scores = np.abs(
            zscore(clean_numeric_df)
        )

        threshold = 3

        anomalies = (z_scores > threshold).any(axis=1)

        fraud_df_zscore = combined_df[anomalies]

        st.write(
            f"Potential anomalies detected using Z-Score: {len(fraud_df_zscore)}"
        )

        if len(fraud_df_zscore) > 0:
            st.dataframe(fraud_df_zscore.head())

        # =========================
        # ISOLATION FOREST
        # =========================

        st.subheader("🤖 ML-Based Fraud Detection")

        try:

            iso_model = IsolationForest(
                contamination=0.02,
                random_state=42
            )

            combined_df["anomaly"] = iso_model.fit_predict(
                clean_numeric_df
            )

            fraud_df_ml = combined_df[
                combined_df["anomaly"] == -1
            ]

            st.write(
                f"Suspicious records detected using Isolation Forest: {len(fraud_df_ml)}"
            )

            if len(fraud_df_ml) > 0:
                st.dataframe(fraud_df_ml.head())

        except Exception as e:
            st.error(f"Isolation Forest Error: {e}")

        # =========================
        # SOURCE FILE ANALYSIS
        # =========================

        st.subheader("📁 Source File Analysis")

        source_counts = combined_df["source_file"].value_counts()

        st.dataframe(source_counts)

        fig3, ax3 = plt.subplots(figsize=(8, 4))

        source_counts.plot(
            kind="bar",
            ax=ax3
        )

        ax3.set_title("Records Per Uploaded File")

        st.pyplot(fig3)

    else:
        st.warning(
            "No numeric columns found for ML analysis."
        )

else:
    st.info(
        "Please upload one or more CSV files to begin analysis."
    )