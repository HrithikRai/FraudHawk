# Finding anomalies based on Isolation Forest
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
import requests
import time
from datetime import datetime, timedelta
import base64

# Setting up Streamlit
st.set_page_config(page_title="FraudHawk - Real-Time Transaction Monitoring for Fraud Detection", layout="wide")
st.markdown("<style>{}</style>".format(open("style.css").read()), unsafe_allow_html=True)

backend_url = "http://127.0.0.1:8000"

# Load data and reset session on new upload
def load_data(file):
    data = pd.read_csv(file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Train model for anomaly detection
def train_anomaly_model(data):
    model_data = data[['amount']]
    model = IsolationForest(contamination=0.05, random_state=42)
    with st.spinner("Training model..."):
        model.fit(model_data)
    st.success("Model training complete!")
    return model

# Generate synthetic data for a dummy user
def generate_dummy_data():
    timestamps = pd.date_range(end=datetime.now(), periods=50, freq='H')
    data = {
        "timestamp": timestamps,
        "amount": np.random.uniform(10, 500, len(timestamps)),
        "category": np.random.choice(["groceries", "electronics", "dining", "entertainment"], len(timestamps)),
        "online": np.random.choice(["online", "onsite"], len(timestamps))
    }
    df = pd.DataFrame(data)
    return df

# Helper to clear session and refresh with new user data
def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]

# Button to spawn a dummy user CSV
if st.button("Spawn Dummy User"):
    dummy_data = generate_dummy_data()
    csv = dummy_data.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()  # encode to base64
    st.download_button(label="Download Dummy User Transactions", data=csv, file_name='dummy_user.csv', mime='text/csv')

# File uploader for transaction history
uploaded_file = st.file_uploader("Upload User Transaction Data CSV", type=["csv"])
if uploaded_file:
    clear_session()
    data = load_data(uploaded_file)
    st.session_state['data'] = data
    st.session_state['user_id'] = f"user_{int(time.time())}"
    st.session_state['model'] = train_anomaly_model(data)  # Train and store model in session state
    st.session_state['transactions'] = data  # Store original transactions

# Display user transaction history and analysis
if 'data' in st.session_state:
    st.subheader("Transaction History and Anomaly Detection")
    data_with_anomalies = st.session_state['data']
    fig = px.scatter(data_with_anomalies, x='timestamp', y='amount',
                     title="Transactions")
    st.plotly_chart(fig, use_container_width=True)

# Simulatng the purchase session
if st.button("Start Purchasing"):
    st.write("Monitoring transactions...")
    transactions = []
    current_time = datetime.now()
    model = st.session_state.get('model')

    # Simulate 10 transactions
    for i in range(10):
        amount = np.random.uniform(10, 1000)
        category = np.random.choice(["groceries", "electronics", "dining", "entertainment", "luxury"])
        online = np.random.choice(["online", "onsite"])

        # Create new transaction
        new_transaction = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "category": category,
            "online": online
        }

        # Detect anomaly using trained model
        anomaly_label = model.predict([[amount]])[0]
        new_transaction['anomaly'] = "Fraudulent" if anomaly_label == -1 else "Normal"

        # Append transaction to transactions list
        transactions.append(new_transaction)
        transaction_df = pd.DataFrame(transactions)  # Create DataFrame with 'anomaly' column

        # Display real-time transaction alert
        color = "red" if new_transaction['anomaly'] == "Fraudulent" else "black"
        st.write(
            f"Transaction Alert: {new_transaction['timestamp']} | Amount: ${new_transaction['amount']:.2f} | {new_transaction['anomaly']}",
            color=color
        )

        # backend updation with transactions
        user_id = st.session_state['user_id']
        requests.post(f"{backend_url}/transactions/{user_id}/", json=new_transaction)
        current_time += timedelta(hours=1)
        time.sleep(1)  

    # Update the chart after all transactions are added
    with st.container():
        fig = px.scatter(transaction_df, x='timestamp', y='amount', color='anomaly',
                         title="Transaction Amount with Anomalies Highlighted",
                         color_discrete_map={"Fraudulent": "red", "Normal": "green"})  # Set custom colors
        st.plotly_chart(fig, use_container_width=True)