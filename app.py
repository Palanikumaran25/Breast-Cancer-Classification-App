import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set up page
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide") 

# Streamlit UI
st.title("🔬 Breast Cancer Classification App")
st.markdown("""
This AI-powered app uses a **Machine Learning Model Classifier** trained on the sklearn Breast Cancer dataset
for predicting whether a tumor is **malignant** or **benign**.
""")   

# Load and cache data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

X, y, labels = load_data()

# Preprocess function
def preprocess_data(X_raw, scaler_type="standard"):
    missing_values = X_raw.isnull().sum().sum()
    X_imputed = X.fillna(0)
    X_unique = X_imputed.nunique()

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, scaler, missing_values, X_unique, X_imputed

# Sidebar: Settings
st.sidebar.header("⚙️ Settings")
scaler_choice = st.sidebar.selectbox("Scaler Type", ["standard", "minmax", "robust"])
test_size = st.sidebar.slider("Test Size (Train/Test Split)", 0.1, 0.5, 0.2, 0.05)
X_scaled, scaler, missing_values, X_unique, _ = preprocess_data(X, scaler_type=scaler_choice)

# Preprocessing insights
st.sidebar.markdown(f"📉 Missing Values: `{missing_values}`")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Sidebar: Model Selection
st.sidebar.header("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
selected_model = models[selected_model_name]

# Train the selected model
selected_model.fit(X_train, y_train)
y_pred = selected_model.predict(X_test)
y_proba = selected_model.predict_proba(X_test)[:, 1] if hasattr(selected_model, "predict_proba") else None

# Sidebar: Input Features
st.sidebar.header("📝 Enter Tumor Features")
input_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]
user_input = [st.sidebar.number_input(f"{feat.title()}", min_value=0.0, value=0.0) for feat in input_features]

if st.sidebar.button("🧠 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_padded = np.pad(input_array, ((0, 0), (0, 30 - len(input_features))), mode='constant')
    input_scaled = scaler.transform(input_padded)
    pred = selected_model.predict(input_scaled)[0]
    
    st.subheader(f"📌 Prediction using **{selected_model_name}**: **{labels[pred]}**")
    
    if hasattr(selected_model, "predict_proba"):
        prob = selected_model.predict_proba(input_scaled)[0]
        st.metric("🟢 Probability (Benign)", f"{prob[1]:.2f}")
        st.metric("🔴 Probability (Malignant)", f"{prob[0]:.2f}")
    else:
        st.info("Selected model does not support probability scores.") 

st.markdown("---")

# Model Evaluation
st.subheader(f"📊 Evaluation: **{selected_model_name}**")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

with col2:
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose()) 

st.markdown("---")

# ROC Curve
if y_proba is not None:
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
else:
    st.warning("Selected model does not support ROC curve.") 
    
st.markdown("---")

# Batch Prediction with Uploaded CSV
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
            df_upload[col] = 0  # Add missing columns

    df_upload = df_upload[expected_cols]  # Reorder columns

    # Preprocess and predict
    X_upload_scaled = scaler.transform(df_upload)
    batch_predictions = selected_model.predict(X_upload_scaled)
    df_upload['Prediction'] = [labels[p] for p in batch_predictions]

    if hasattr(selected_model, "predict_proba"):
        batch_probs = selected_model.predict_proba(X_upload_scaled)
        df_upload['Probability (Benign)'] = batch_probs[:, 1]
        df_upload['Probability (Malignant)'] = batch_probs[:, 0]
    else:
        df_upload['Probability (Benign)'] = "N/A"
        df_upload['Probability (Malignant)'] = "N/A"

    st.success("✅ Batch Prediction Complete!")
    st.dataframe(df_upload[['Prediction', 'Probability (Benign)', 'Probability (Malignant)']])

st.markdown("---")
st.markdown("✅ Built with Streamlit & Scikit-learn | 🧠 Powered by AI")