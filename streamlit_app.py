import streamlit as st
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
import shap
import pickle
import json

st.set_page_config(layout='centered')

# load the lottie json file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    

# load the lottie animation of a car driving
lottie_employee = load_lottiefile("./streamlit_data/car_animation.json")
st_lottie(lottie_employee, speed=1, loop=True, height=200)

# print the title for the app
st.title('Salifort Motors')

# get the columns and unique data for the dataset
df = pd.read_csv('./data/HR_capstone_dataset.csv')
# change the column names to lower case
df.columns = df.columns.str.lower()
# fix typo in monthly hours
df = df.rename(columns={'average_montly_hours':'average_monthly_hours'})
# drop duplicate rows
df = df.drop_duplicates(keep='first')

with st.sidebar.form("Employee Data"):
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


# column names
col_names = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_monthly_hours', 'time_spend_company', 'work_accident', 'left',
       'promotion_last_5years', 'department', 'salary']

tab1, tab2, tab3 = st.tabs(['Overview', 'Model Prediction', 'Data Visualizations'])

with tab1:
    st.subheader('Employee Attrition Prediction App')
    """
    This app uses the data from the **Google Advanced Data Analytics Capstone Project** 
    to predict whether an employee will leave the fictitious car company, Salifort Motors.
    The data was collected from the HR department of Salifort Motors. The data contains information 
    about the employees, such as their salary, department, and number of projects they have worked on. 
    The data also contains information about whether the employee left the company or not. 
    The data contains information on 14999 employees. 
    A machine learning classification model was trained on the data and provides
    a prediction of leaving, the probability of leaving, as well as a categorical grouping of the employee.
    
    #### Employee Leaving Categories:
    - **Low Satisfaction/High Performer**
    - **Medium Satisfaction/Low Performer**
    - **High Satisfaction/Medium Performer**
    - **General Employee**
    
    """
    
    st.write('The Employee Leaving Cohorts can be clearly seen in the plot below:')
    
    # look at the users who left with low satisfaction level and monthly hours
    user_left = df[df['left'] == 1]
    user_left.loc[:, 'left'] = 'Employee Left'
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=user_left, x='average_monthly_hours', y='satisfaction_level', hue='number_project')
    plt.title('Satisfaction Level/Monthly Hours/# of Projects for Employees Leaving', fontsize=12)
    plt.xlabel('Average Monthly Hours')
    plt.ylabel('Satisfaction Level')
    plt.legend(bbox_to_anchor= (1.0, 0.5), title='Number of Projects')
    st.pyplot(fig)
    
with tab2:
    st.subheader('Model Prediction')
    # Load model from file
    filename = './models/best_model.json'
    best_model = XGBClassifier()
    best_model.load_model(filename)

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
    with open('./streamlit_data/model_cols.json', 'r') as f:
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
    
        # categorize the employee
    if satisfacion_level > 0.7 and average_monthly_hours > 215 and average_monthly_hours < 275:
        employee_cat = 'High Satisfaction/Medium Performer'
    elif satisfacion_level < 0.2 and average_monthly_hours > 240:
        employee_cat = 'Low Satisfaction/High Performer'
    elif satisfacion_level > 0.35 and satisfacion_level < 0.45 and average_monthly_hours > 125 and average_monthly_hours < 165:
        employee_cat = 'Medium Satisfaction/Low Performer'
    else:
        employee_cat = 'General Employee'
    
    st.markdown(f'**Employee Category:** {employee_cat}')

    st.markdown(f'**Probability of leaving:** {round(100*probability_leave)}%')

    # print the prediction to the streamlit app
    if prediction == 1:
        st.markdown('**Employee is predicted to leave**')
    else:
        st.markdown('**Employee is predicted to stay**')


    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(best_model)
    input_shap_values = explainer.shap_values(input_df)


    # SHAP force plot for inputed instance predicted class
    st.subheader('Explained Prediction - SHAP Value Plot')


    force_plot = shap.force_plot(explainer.expected_value,
                    input_shap_values ,
                    input_df.columns,
                    matplotlib=True,
                    show=False)
    st.pyplot(force_plot, bbox_inches='tight', dpi=300)
    
    
    # SHAP force plot for inputed instance predicted class
    fig, ax = plt.subplots()
    shap.summary_plot(input_shap_values, input_df, plot_type="bar", show=False)
    plt.title('Feature Importance on Model Prediction', fontsize=14)
    plt.xlabel('')
    st.pyplot(fig)
    



with tab3:
    st.subheader('Data Visualizations')
    """
    The following are visualizations of distribution of **All Employees vs the Input Employee**
    """
    
          
    st.write('The Employee Leaving Cohorts can be clearly seen in the plot below:')
    
    # look at the users who left with low satisfaction level and monthly hours
    employee_df = df.copy()
    employee_df['left'] = employee_df['left'].apply(lambda x: 'Employee Left' if x == 1 else 'Employee Stayed')
    # randomly sample 1000 rows to make the plot more readable
    employee_df = employee_df.sample(1000, random_state=42)
    # concat the input employee data
    input_df['left'] = 'Input Employee'
    employee_df = pd.concat([employee_df, input_df])
        
    # Define a color palette dictionary
    colors = {'Employee Left': 'blue', 'Employee Stayed': 'grey', 'Input Employee': 'red'}
        
    # plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=employee_df, x='average_monthly_hours', y='satisfaction_level', hue='left', palette=colors)
    plt.title('Satisfaction Level and Monthly Hours for Employees', fontsize=12)
    plt.xlabel('Average Monthly Hours')
    plt.ylabel('Satisfaction Level')
    plt.legend(bbox_to_anchor= (1.0, 0.5), title='Employee Classification')
    st.pyplot(fig)


    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['satisfaction_level'], ax=ax, label='All Employees')
        plt.axvline(satisfacion_level, color='red', label='Input Employee Data')
        plt.title('Employee Satisfaction Level')
        plt.ylabel('Number of Employees')
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.histplot(df['number_project'], ax=ax, label='All Employees', bins=df['number_project'].nunique())
        plt.axvline(number_of_projects, color='red', label='Input Employee Data')
        plt.title('Number of Projects')
        plt.ylabel('Number of Employees')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df['last_evaluation'], ax=ax, label='All Employees')
        plt.axvline(last_evaluation, color='red', label='Input Employee Data')
        plt.title('Employee Last Evaluation')
        plt.ylabel('Number of Employees')
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.histplot(df['average_monthly_hours'], ax=ax, label='All Employees')
        plt.axvline(average_monthly_hours, color='red', label='Input Employee Data')
        plt.title('Average Monthly Hours')
        plt.ylabel('Number of Employees')
        st.pyplot(fig)
  