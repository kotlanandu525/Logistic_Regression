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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config("Logistic Regression - Telco Churn", layout="centered")


def load_css(file):
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path, file)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# -----------------------------
# Title
# -----------------------------
st.markdown("""
<div class="card">
    <h1>Logistic Regression</h1>
    <p>Predict <b>Customer Churn</b> using Telco Dataset</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Load data (FIXED)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (2).csv")

df = load_data()


# -----------------------------
# Dataset preview
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Preprocessing
# -----------------------------
st.subheader("Data Preprocessing")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

st.success("Data preprocessing completed ✔")


# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

st.success("Logistic Regression model trained successfully ✔")


# -----------------------------
# Evaluation
# -----------------------------
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
# -----------------------------
# Simple User Prediction
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Churn (Simple Input)")

# --- User Inputs ---
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

# --- Create minimal input dataframe ---
user_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract
}])

# --- One-hot encode ---
user_df = pd.get_dummies(user_df)

# --- Align with training features ---
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# --- Scale ---
user_scaled = scaler.transform(user_df)

# --- Predict ---
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# --- Output ---
st.markdown(
    f"""
    <div class="prediction-box">
        <h3>Prediction Result</h3>
        <p>
            <b>Churn:</b> {"Yes" if prediction == 1 else "No"}<br>
            <b>Churn Probability:</b> {probability:.2f}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
