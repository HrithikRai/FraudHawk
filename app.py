# Finding anomalies based on reconstruction error from autoencoders
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import time
from datetime import datetime, timedelta
import random

# Set up Streamlit
st.set_page_config(page_title="FraudHawk - Real-Time Transaction Monitoring for Fraud Detection", layout="wide")
st.markdown("<style>{}</style>".format(open("style.css").read()), unsafe_allow_html=True)

backend_url = "http://127.0.0.1:8000"

# Load data and reset session on new upload
def load_data(file):
    data = pd.read_csv(file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Embedding model setup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Autoencoder Definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(data):
    # Embed categorical data
    data['category_embed'] = data['category'].apply(lambda x: embedding_model.encode(x))
    data['online_embed'] = data['online'].apply(lambda x: embedding_model.encode(x))

    # Combine embeddings with numerical features
    embeddings = np.vstack(data['category_embed'].values) + np.vstack(data['online_embed'].values)
    numerical_data = data[['amount']].values
    feature_data = np.hstack([embeddings, numerical_data])

    # Standardize data
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)

    # Split data and train autoencoder
    train_data, val_data = train_test_split(feature_data, test_size=0.2, random_state=42)
    input_dim = feature_data.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    st.write("Training model...")
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(train_data, dtype=torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    st.write("Model training complete.")
    return model, scaler

# Generate dummy transaction CSVs
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

# Button to spawn a dummy user 
if st.button("Spawn Dummy User"):
    dummy_data = generate_dummy_data()
    csv = dummy_data.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Dummy User Transactions", data=csv, file_name='dummy_user.csv', mime='text/csv')

uploaded_file = st.file_uploader("Upload User Transaction Data CSV", type=["csv"])
if uploaded_file:
    clear_session()
    data = load_data(uploaded_file)
    st.session_state['data'] = data
    st.session_state['user_id'] = f"user_{int(time.time())}"
    st.session_state['model'], st.session_state['scaler'] = train_autoencoder(data)

# Display user transaction history and analysis
if 'data' in st.session_state:
    st.subheader("Transaction History and Anomaly Detection")
    data_with_anomalies = st.session_state['data']
    fig = px.scatter(data_with_anomalies, x='timestamp', y='amount', title="Transaction Amount")
    st.plotly_chart(fig, use_container_width=True)

# begin purchasing simulation
if st.button("Start Purchasing"):
    st.write("Monitoring transactions...")
    transactions = []
    current_time = datetime.now()
    model = st.session_state.get('model')
    scaler = st.session_state.get('scaler')

    for i in range(10):
        amount = np.random.uniform(10, 1000)
        category = np.random.choice(["groceries", "electronics", "dining", "entertainment", "luxury"])
        online = np.random.choice(["online", "onsite"])

        # Embed and scale new transaction
        category_embed = embedding_model.encode(category)
        online_embed = embedding_model.encode(online)
        feature_vector = np.hstack([category_embed + online_embed, [amount]])
        scaled_vector = scaler.transform([feature_vector])

        # Predict anomaly
        model.eval()
        input_vector = torch.tensor(scaled_vector, dtype=torch.float32)
        reconstructed = model(input_vector).detach().numpy()
        reconstruction_error = mean_squared_error(scaled_vector, reconstructed)
        
        # Threshold for fraud detection
        fraud_threshold = 1.1
        is_fraudulent = reconstruction_error > fraud_threshold

        # Create new transaction entry
        new_transaction = {
            "timestamp": current_time,
            "amount": amount,
            "category": category,
            "online": online,
            "fraud_score": reconstruction_error,
            "anomaly": "Fraudulent" if is_fraudulent else "Normal"
        }

        # Real-time update on dashboard
        color = "red" if is_fraudulent else "black"
        st.write(f"Transaction Alert: {new_transaction['timestamp']} | Amount: ${new_transaction['amount']:.2f} | {new_transaction['anomaly']} (Score: {new_transaction['fraud_score']:.2f})", color=color)

        # Append transaction and update dashboard
        transactions.append(new_transaction)
        transaction_df = pd.DataFrame(transactions)
        # backend updation 
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
