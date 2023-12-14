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
    satisfacion_level = st.slider(
        "Employee Satisfaction Level",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    last_evaluation = st.slider(
        "Employee Last Evaluation",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    number_of_projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=10,
        value=5
    )
    average_monthly_hours = st.number_input(
        "Average Monthly Hours",
        min_value=0,
        max_value=500,
        value=200
    )
    time_spend_company = st.number_input(
        "Years Spent at Salifort Motors",
        min_value=0,
        max_value=10,
        value=5
    )
    promotion_last_5years = st.selectbox(
        "Promotion in Last 5 Years",
        options=['Yes', 'No']
    )
    work_accident = st.selectbox(
        "Work Accident",
        options=['Yes', 'No']
    )
    
    st.form_submit_button("Submit")

# Load model from file
filename = 'best_model.json'
best_model = XGBClassifier()
best_model.load_model(filename)


# to delete later
satisfacion_level = 0.2
last_evaluation = 0.9
number_of_projects = 7
average_monthly_hours = 260
time_spend_company = 5
work_accident = 'No'
promotion_last_5years = 'No'
department = 'sales'
salary = 'low'




# create the dataframe from user input
input_df = pd.DataFrame([{col_names[0]:satisfacion_level, 
                          col_names[1]:last_evaluation, 
                          col_names[2]:number_of_projects,
                          col_names[3]:average_monthly_hours,
                          col_names[4]:time_spend_company,
                          col_names[5]:work_accident,
                          col_names[7]:promotion_last_5years,
                          col_names[8]:department,
                          col_names[9]:salary}])

# convert yes/no to 1/0
input_df['work_accident'] = input_df['work_accident'].apply(lambda x: 1 if x == 'Yes' else 0)
input_df['promotion_last_5years'] = input_df['promotion_last_5years'].apply(lambda x: 1 if x == 'Yes' else 0)


# add in the features engineered
# create the over 4 years and no promotion feature
input_df['over_4yr_no_promo'] = input_df.apply(lambda row: 1 if (row['promotion_last_5years'] == 0)&(row['time_spend_company'] > 4) else 0, axis=1)
# create the over worked high performer feature
input_df['over_worked_high_performer'] = input_df['number_project'] * input_df['average_monthly_hours'] * input_df['last_evaluation'] / input_df['satisfaction_level']

num_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'time_spend_company', 'over_worked_high_performer']
cat_cols = ['salary', 'department', 'over_4yr_no_promo']

# check that input_df has the necessary columns
for col in num_cols + cat_cols:
    assert col in input_df.columns, f'Column {col} not in input_df'
    
# create dummy variables
input_df = pd.get_dummies(input_df, columns=cat_cols)


# load the columns from the json file
with open('model_cols.json', 'r') as f:
    all_columns = json.load(f)

# find the missing columns
missing_cols = set(all_columns) - set(input_df.columns)

# set the missing columns to 0 for input_df
for col in missing_cols:
    input_df[col] = 0
    
# ensure the order of columns in the input_df is the same as the order in all_columns
input_df = input_df[all_columns]

# make the prediction
probability_leave = best_model.predict_proba(input_df)[0][1]
prediction = best_model.predict(input_df)

# print the prediction to the streamlit app
if prediction == 1:
    st.write('Employee is predicted to leave')
st.write(f'Probability of leaving: {probability_leave}')

