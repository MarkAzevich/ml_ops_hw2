import streamlit as st
import requests
import pickle
import numpy as np
from models import ModelManager

API_URL = "http://fastapi:8000"


st.title("ML Model Management Dashboard")


st.header("Upload Dataset to MinIO")
uploaded_file = st.file_uploader("Upload Dataset", type=["pkl"])
if st.button("Upload"):
    if uploaded_file:
        response = requests.post(f"{API_URL}/upload_dataset/", files={"file": uploaded_file})
        st.write(response.json())

st.header("Download Dataset from MinIO")
file_name = st.text_input("Enter Dataset Filename to Download")
if st.button("Download"):
    response = requests.get(f"{API_URL}/download_dataset/{file_name}")
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        st.write(f"File {file_name} downloaded successfully.")
    else:
        st.write(response.json())


st.header("Train a New Model")
model_type = st.selectbox("Select Model Type", ["logistic_regression", "random_forest"])
hyperparameters_file = st.file_uploader("Upload Hyperparameters (.pkl)", type=["pkl"])
training_data_file = st.file_uploader("Upload Training Data (.pkl)", type=["pkl"])
target_data_file = st.file_uploader("Upload Target Data (.pkl)", type=["pkl"])
model_name = st.text_input("Enter Model Name (optional)")
if st.button("Train"):
    if hyperparameters_file and training_data_file and target_data_file:
        hyperparameters = pickle.load(hyperparameters_file)
        training_data = pickle.load(training_data_file)
        target_data = pickle.load(target_data_file)
        # Convert numpy arrays to lists if necessary
        if isinstance(training_data, np.ndarray):
            training_data = training_data.tolist()
        if isinstance(target_data, np.ndarray):
            target_data = target_data.tolist()

        print(hyperparameters)
        print(training_data)
        print(target_data)


        response = requests.post(f"{API_URL}/train", json={
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "training_data": training_data,
            "target_data": target_data,
            "name": model_name
        })
        st.write(response.json())



        response_ = requests.post(f"{API_URL}/upload_dataset/", files={"file": training_data_file})
        st.write(response_.json())


    else:
        st.error("Please upload hyperparameters, training data, and target data files.")

st.header("List All Trained Models")
if st.button("List Models"):
    response = requests.get(f"{API_URL}/models/info")
    st.write(response.json())

st.header("Predict with a Model")
model_id = st.text_input("Enter Model ID for Prediction")
prediction_data_file = st.file_uploader("Upload Prediction Data (.pkl)", type=["pkl"])
predictions_filename = st.text_input("Enter Filename for Saving Predictions (.pkl)")
if st.button("Predict"):
    if prediction_data_file and predictions_filename:
        prediction_data = pickle.load(prediction_data_file)
        # Convert numpy arrays to lists if necessary
        if isinstance(prediction_data, np.ndarray):
            prediction_data = prediction_data.tolist()
        response = requests.post(f"{API_URL}/predict", json={
            "model_id": model_id,
            "data": prediction_data,
            "filename": predictions_filename
        })
        st.write(response.json())
    else:
        st.error("Please upload prediction data file and specify filename for predictions.")

st.header("Delete a Model")
delete_model_id = st.text_input("Enter Model ID to Delete")
if st.button("Delete"):
    response = requests.delete(f"{API_URL}/models/{delete_model_id}")
    st.write(response.json())

st.header("Retrain a Model")
retrain_model_id = st.text_input("Enter Model ID to Retrain")
retrain_data_file = st.file_uploader("Upload Retrain Data (.pkl)", type=["pkl"])
retrain_target_file = st.file_uploader("Upload Retrain Target (.pkl)", type=["pkl"])
retrain_predictions_filename = st.text_input("Enter Filename for Saving Retrain Predictions (.pkl)")
if st.button("Retrain"):
    if retrain_data_file and retrain_target_file and retrain_predictions_filename:
        retrain_data = pickle.load(retrain_data_file)
        retrain_target = pickle.load(retrain_target_file)
        # Convert numpy arrays to lists if necessary
        if isinstance(retrain_data, np.ndarray):
            retrain_data = retrain_data.tolist()
        if isinstance(retrain_target, np.ndarray):
            retrain_target = retrain_target.tolist()


        response = requests.post(f"{API_URL}/retrain", json={
            "model_id": retrain_model_id,
            "data": retrain_data,
            "target": retrain_target,
            "filename": retrain_predictions_filename
        })
        st.write(response.json())


        response_ = requests.post(f"{API_URL}/upload_dataset/", files={"file": training_data_file})
        st.write(response_.json())
        
    else:
        st.error("Please upload retrain data, retrain target files, and specify filename for retrain predictions.")
