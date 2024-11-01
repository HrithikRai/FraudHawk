from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# Simulated in-memory storage for transactions
transactions_db = {}

# Transaction data model
class Transaction(BaseModel):
    timestamp: str
    amount: float
    category: str
    online: str

@app.post("/transactions/{user_id}/")
def add_transaction(user_id: str, transaction: Transaction):
    # Add a transaction for a given user_id
    if user_id not in transactions_db:
        transactions_db[user_id] = []
    transactions_db[user_id].append(transaction.dict())
    return {"status": "transaction added", "user_id": user_id}

@app.get("/transactions/{user_id}")
def get_transactions(user_id: str):
    # Get all transactions for a given user_id
    if user_id in transactions_db:
        return {"transactions": transactions_db[user_id]}
    return {"status": "no transactions found"}
