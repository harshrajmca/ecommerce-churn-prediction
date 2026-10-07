import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

DATA_PATH = "ecommerce_churn_data.csv"
RF_MODEL_PATH = "churn_rf_model.pkl"
KMEANS_PATH = "churn_kmeans.pkl"
PREPROCESS_PATH = "churn_preprocessing.pkl"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Tenure" in df.columns:
        df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
    return df


@st.cache_resource
def load_models():
    if not (os.path.exists(RF_MODEL_PATH) and os.path.exists(KMEANS_PATH) and os.path.exists(PREPROCESS_PATH)):
        return None, None, None
    return (
        joblib.load(RF_MODEL_PATH),
        joblib.load(KMEANS_PATH),
        joblib.load(PREPROCESS_PATH),
    )


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Churn", "CustomerID"], errors="ignore")


def get_training_column_names(preprocessing):
    cat_cols = list(preprocessing.transformers_[0][2])
    num_cols = list(preprocessing.transformers_[1][2])
    cat_encoded = preprocessing.named_transformers_["categorial"].get_feature_names_out(cat_cols)
    return np.concatenate([cat_encoded, num_cols])


def prepare_clustered_df(features: pd.DataFrame, preprocessing, kmeans) -> pd.DataFrame:
    transformed = preprocessing.transform(features)
    feature_names = get_training_column_names(preprocessing)
    processed = pd.DataFrame(transformed, columns=feature_names)
    cluster_cols = ["OrderCount", "DaySinceLastOrder"]
    clusters = kmeans.predict(processed[cluster_cols])
    processed["Cluster"] = clusters
    return processed


def retention_label(value: int) -> str:
    return "Retained" if value == 0 else "Churned"


df = load_data()
rf_model, kmeans, preprocessing = load_models()

st.title("Customer Segmentation and Retention Dashboard")
st.caption("Built from your churn analysis notebook and trained model artifacts.")

if rf_model is None:
    st.error(
        "Trained models were not found. Please run `python churn_2_fixed.py` once to generate model files."
    )
    st.stop()

features = build_feature_frame(df)
processed_with_cluster = prepare_clustered_df(features, preprocessing, kmeans)
predictions = rf_model.predict(processed_with_cluster)
df["PredictedChurn"] = predictions
df["RetentionStatus"] = df["PredictedChurn"].apply(retention_label)
df["Segment"] = processed_with_cluster["Cluster"].values

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Actual Churn Rate", f"{df['Churn'].mean() * 100:.2f}%")
col3.metric("Predicted Churn Rate", f"{df['PredictedChurn'].mean() * 100:.2f}%")
col4.metric("Average Tenure", f"{df['Tenure'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["Overview", "Segmentation", "Single Customer Prediction"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Actual Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Churn", palette="Set2", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained", "Churned"])
        ax.set_ylabel("Customers")
        st.pyplot(fig)

    with right:
        st.subheader("Churn by Preferred Login Device")
        fig, ax = plt.subplots(figsize=(7, 4))
        churn_by_device = (
            df.groupby("PreferredLoginDevice")["Churn"].mean().sort_values(ascending=False) * 100
        )
        churn_by_device.plot(kind="bar", ax=ax, color="#ef4444")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)

    st.subheader("Retention Drivers Snapshot")
    drivers = (
        df.groupby("Churn")[["OrderCount", "DaySinceLastOrder", "CashbackAmount", "SatisfactionScore"]]
        .mean()
        .rename(index={0: "Retained", 1: "Churned"})
    )
    st.dataframe(drivers.round(2), use_container_width=True)

with tab2:
    st.subheader("Customer Segments (KMeans Clusters)")
    seg_cols = ["Segment", "OrderCount", "DaySinceLastOrder", "CashbackAmount", "Churn"]
    segment_summary = (
        df[seg_cols]
        .groupby("Segment")
        .agg(
            Customers=("Churn", "count"),
            AvgOrderCount=("OrderCount", "mean"),
            AvgRecencyDays=("DaySinceLastOrder", "mean"),
            AvgCashback=("CashbackAmount", "mean"),
            ChurnRate=("Churn", "mean"),
        )
    )
    segment_summary["ChurnRate"] = segment_summary["ChurnRate"] * 100
    st.dataframe(segment_summary.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="OrderCount",
        y="DaySinceLastOrder",
        hue="Segment",
        palette="viridis",
        alpha=0.65,
        ax=ax,
    )
    st.pyplot(fig)

with tab3:
    st.subheader("Predict Churn for a New Customer")
    st.write("Adjust the inputs and click predict.")

    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
    device = c2.selectbox(
        "Preferred Login Device",
        sorted(df["PreferredLoginDevice"].dropna().unique().tolist()),
    )
    age = c3.slider("Age", 18, 80, 30)

    c4, c5, c6 = st.columns(3)
    tenure = c4.slider("Tenure (months)", 0, 60, 12)
    city_tier = c5.selectbox("City Tier", [1, 2, 3])
    warehouse = c6.slider("Warehouse To Home", 1, 50, 15)

    c7, c8, c9 = st.columns(3)
    app_hours = c7.slider("Hours Spent On App", 0.0, 5.0, 2.5)
    devices_reg = c8.slider("Registered Devices", 1, 10, 2)
    satisfaction = c9.slider("Satisfaction Score", 1, 5, 3)

    c10, c11, c12 = st.columns(3)
    addresses = c10.slider("Number Of Addresses", 1, 20, 2)
    complain = c11.selectbox("Complain", [0, 1])
    order_count = c12.slider("Order Count", 0, 20, 5)

    c13, c14 = st.columns(2)
    days_since_order = c13.slider("Days Since Last Order", 0, 60, 10)
    cashback = c14.slider("Cashback Amount", 0.0, 400.0, 120.0)

    if st.button("Predict Churn"):
        new_customer = pd.DataFrame(
            {
                "Gender": [gender],
                "PreferredLoginDevice": [device],
                "Age": [age],
                "Tenure": [tenure],
                "CityTier": [city_tier],
                "WarehouseToHome": [warehouse],
                "HoursSpentOnApp": [app_hours],
                "NumberOfDevicesRegistered": [devices_reg],
                "SatisfactionScore": [satisfaction],
                "NumberOfAddress": [addresses],
                "Complain": [complain],
                "OrderCount": [order_count],
                "DaySinceLastOrder": [days_since_order],
                "CashbackAmount": [cashback],
            }
        )

        transformed = preprocessing.transform(new_customer)
        feature_names = get_training_column_names(preprocessing)
        new_processed = pd.DataFrame(transformed, columns=feature_names)
        cluster_cols = ["OrderCount", "DaySinceLastOrder"]
        new_cluster = int(kmeans.predict(new_processed[cluster_cols])[0])
        new_processed["Cluster"] = new_cluster
        pred = int(rf_model.predict(new_processed)[0])
        probability = float(rf_model.predict_proba(new_processed)[0][1])

        st.write(f"Assigned Segment: **{new_cluster}**")
        st.write(f"Predicted Churn Probability: **{probability * 100:.2f}%**")
        if pred == 1:
            st.error("Prediction: Customer is likely to churn.")
        else:
            st.success("Prediction: Customer is likely to stay.")
