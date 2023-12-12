import streamlit
import pandas as pd
from xgboost import XGBClassifier
import pickle

# Load model from file
filename = '../best_model.json'
best_model = XGBClassifier()
best_model.load_model(filename)




