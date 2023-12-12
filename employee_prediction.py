import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import pickle

# print the title for the app
st.title('Employee Attrition Prediction')
st.write('This app predicts the probability of employee attrition for Salifort Motors')


# get the columns and unique data for the dataset
df = pd.read_csv('HR_capstone_dataset.csv')
# change the column names to lower case
df.columns = df.columns.str.lower()
# fix typo in monthly hours
df = df.rename(columns={'average_montly_hours':'average_monthly_hours'})

# column names
col_names = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_monthly_hours', 'time_spend_company', 'work_accident', 'left',
       'promotion_last_5years', 'department', 'salary']

with st.form("Employee Data"):
    department = st.selectbox(
        "Department",
        options=df['department'].unique()
    )
    salary = st.selectbox(
        "Salary",
        options=df['salary'].unique()
    )
    
    st.form_submit_button("Submit")

# Load model from file
filename = 'best_model.json'
best_model = XGBClassifier()
best_model.load_model(filename)




