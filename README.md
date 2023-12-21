# Salifort_Motors_Employee_Retention_Project
The Capstone project for the Google Advanced Data Analytics Certificate

#### View the Interactive Web App Here: 
[Salifort Motors Project Web App](https://salifort-motors-employee-retention.streamlit.app/)

![Streamlit_App](./images/salifort_web_app_screenshot.png)

Dataset can be downloaded from Kaggle: [Salifort Motors Dataset](https://www.kaggle.com/datasets/leviiiest/salifort-motor-hr-dataset)

## Project Evaluation

- **Model Accuracy:**
   -  98% for classifying Employee Churn.

- **Model Recall:**
   - 93% of the time, the model will correctly predict the employees who will leave.
 

### There are three main groups of employees who are likely to leave:

1. **Low Satisfaction/High Performers:**
    - 4-5 years with the company but not promotion
    - Average Monthly Hours greater than 240
    - 6-7 projects
    - High Evaluation Scores
    - Low Satisfaction Scores

- **Recommendation:** These employees are very valuable to the company. They are due for a promotion and likely have too high of a work load, so reduce the work load and/or offer a promotion

2. **Mid Satisfaction/Low Performers:**
    - 3 years with the company
    - Average Monthly Hours less than 165
    - 2-3 projects
    - Low Evaluation Scores
    - Mid Satisfaction Scores

- **Recommendation:** These employees are currently not very valuable to the company. They can take some of the workload from the High  Performers. This could improve retention of the high performers and also possibly increase the retention of this work group. They are possibly bored with the low amount of work. Since this group is not as productive, if they still choose to leave, even after the increase in projects/hours, then they will not be as detrimental. They are typically newer employees and do not drain the company knowledge when they leave.

3. **High Satisfaction/Mid Performers:**
    - Greater than 5 years at the company
    - No Promotion
    - Average Monthly Hours between 215-275
    - 3-5 Projects

- **Recommenation:** These employees are valuable to the company. They have been at the company a long time and likely have a lot of knowledge of company practices. Without a promotion, they are likely to leave. They are not overworked so their workload should not be redistributed. By offering a promotion, their likelihood of leaving will be reduced.
       
       


#### Model Use Case:

**This model can predict employees that are likely to leave. Then the employee can be grouped into these three categories and managers can propose better changes that will positively impact the employee.**


## Exploratory Data Analysis:

### Imbalanced Target

![Employee_Target](./images/employee_target_pie_chart.png)

