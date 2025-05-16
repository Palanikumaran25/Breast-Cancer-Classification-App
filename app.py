import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set up page
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# Title
st.title("🔬 Breast Cancer Classification App")

# Load data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

# Load and prepare data
X, y, labels = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Sidebar inputs
st.sidebar.header("📝 Enter Tumor Features")
input_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]
user_input = [st.sidebar.number_input(f"{feat.title()}", min_value=0.0, value=0.0) for feat in input_features]

# Prediction
if st.sidebar.button("🧠 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_padded = np.pad(input_array, ((0, 0), (0, 30 - len(input_features))), mode='constant')
    input_scaled = scaler.transform(input_padded)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    st.subheader(f"📌 Prediction: **{labels[pred]}**")
    st.metric("🟢 Probability (Benign)", f"{prob[1]:.2f}")
    st.metric("🔴 Probability (Malignant)", f"{prob[0]:.2f}")

# Model Performance
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
st.write(f"**Test Accuracy:** {accuracy:.2f}")

# Feature Importance
st.subheader("📈 Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

st.dataframe(feature_importance.head(10))

# Batch Prediction
st.subheader("📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload your breast cancer data CSV", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_upload)

    # Handle missing values
    df_upload.fillna(0, inplace=True)

    # Ensure input has all required columns
    expected_cols = X.columns
    for col in expected_cols:
        if col not in df_upload.columns:
            df_upload[col] = 0

    df_upload = df_upload[expected_cols]

    # Make predictions
    X_upload_scaled = scaler.transform(df_upload)
    predictions = model.predict(X_upload_scaled)
    probabilities = model.predict_proba(X_upload_scaled)
    
    df_upload['Prediction'] = [labels[p] for p in predictions]
    df_upload['Probability (Benign)'] = probabilities[:, 1]
    df_upload['Probability (Malignant)'] = probabilities[:, 0]

    st.success("✅ Batch Prediction Complete!")
    st.dataframe(df_upload[['Prediction', 'Probability (Benign)', 'Probability (Malignant)']])

st.markdown("---")
st.markdown("✅ Built with Streamlit & Scikit-learn | 🧠 Powered by AI")
