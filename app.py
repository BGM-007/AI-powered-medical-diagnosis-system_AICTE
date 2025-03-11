import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load Dataset Function
def load_data(condition):
    if condition == "Heart Disease":
        data = pd.read_csv('datasets/heart.csv')
    elif condition == "Blood Sugar Level":
        data = pd.read_csv('datasets/diabetes.csv')
    elif condition == "Parkinson's Disease":
        data = pd.read_csv('datasets/parkinsons.csv')
    else:
        data = pd.DataFrame()  # Empty DataFrame as fallback
    return data

# Train and Save Models
def train_model(data, model_type):
    X = data.drop(['target', 'Outcome', 'status'], axis=1, errors='ignore')
    y = data['target'] if 'target' in data else (data['Outcome'] if 'Outcome' in data else data['status'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "SVM":
        model = SVC()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return scaler, model, accuracy

# Prediction Function
def predict(input_data, scaler, model):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return prediction[0]

# Main Streamlit App
def main():
    st.title("AI-Powered Medical Diagnosis System")
    st.sidebar.title("Select Disease for Prediction")

    condition = st.sidebar.selectbox("Disease", ["Heart Disease", "Blood Sugar Level", "Parkinson's Disease"])
    model_choice = st.sidebar.selectbox("Model", ["SVM", "Logistic Regression", "Random Forest"])

    data = load_data(condition)
    if data.empty:
        st.error("No data available for the selected condition.")
        return

    scaler, model, accuracy = train_model(data, model_choice)
    st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

    st.markdown("### Enter Patient Details Below")
    input_data = []
    for col in data.drop(['target', 'Outcome', 'status'], axis=1, errors='ignore').columns:
        placeholder_value = 0 if col.lower() == 'sex' else 0.0
        value = st.number_input(f"{col}", value=placeholder_value)
        input_data.append(value)

    if st.button("Predict"):
        prediction = predict(input_data, scaler, model)
        result = "Positive" if prediction == 1 else "Negative"
        st.success(f"Prediction: **{result}**")

if __name__ == "__main__":
    main()
