import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Logistic Regression - Telco Churn",
    layout="centered"
)


# --------------------------------------------------
# Load CSS safely
# --------------------------------------------------
def load_css(file):
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path, file)

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>Logistic Regression – Telco Churn</h1>
    <p>Predict whether a customer is likely to churn</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load dataset (DEPLOY SAFE)
# --------------------------------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return pd.read_csv(file_path)

df = load_data()


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
st.subheader("Data Preprocessing")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.mean(numeric_only=True), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop("customerID", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

st.success("Data preprocessing completed ✔")


# --------------------------------------------------
# Features & target
# --------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

st.success("Logistic Regression model trained successfully ✔")


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.subheader("Model Evaluation")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy:.3f}")

st.text("Classification Report")
st.code(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)


# --------------------------------------------------
# Simple User Prediction
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Churn")

tenure = st.slider(
    "Tenure (months)",
    0,
    int(df["tenure"].max()),
    12
)

monthly_charges = st.slider(
    "Monthly Charges",
    float(df["MonthlyCharges"].min()),
    float(df["MonthlyCharges"].max()),
    70.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# Create input dataframe
user_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract
}])

# One-hot encode & align
user_df = pd.get_dummies(user_df)
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Scale
user_scaled = scaler.transform(user_df)

# Predict
probability = model.predict_proba(user_scaled)[0][1]
prediction = 1 if probability >= 0.3 else 0   # lowered threshold

# Risk label
if probability < 0.3:
    risk = "Low Risk"
elif probability < 0.6:
    risk = "Medium Risk"
else:
    risk = "High Risk"

st.markdown(
    f"""
    <div class="prediction-box">
        <h3>Prediction Result</h3>
        <p>
            <b>Churn:</b> {"Yes" if prediction == 1 else "No"}<br>
            <b>Churn Probability:</b> {probability:.2f}<br>
            <b>Risk Level:</b> {risk}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
