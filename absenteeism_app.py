

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score, make_scorer, ConfusionMatrixDisplay, precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
import joblib

import warnings
warnings.filterwarnings(action='ignore')

st.header('Milestone 2')
st.write("Nama: Mitra M P Gurusinga")
st.write("Batch: HCK 006")
st.write("Milestone 2 Deployment Model")

#load data
@st.cache_data
def fetch_data():
    df = pd.read_excel('Absenteeism_at_work_Project.xls')
    return df

df = fetch_data()


Reason_for_absence = st.selectbox('Reason Type', df['Reason for absence'].unique())

Month_of_absence = st.selectbox('Month of Absence', df['Month of absence'].unique())
Day_of_the_week = st.selectbox('Day', df['Day of the week'].unique())

Seasons = st.selectbox('Seasons', df['Seasons'].unique())

Transportation_expense = st.slider('Transportation Expense', min_value=0, max_value=500, value=0, step=1)

Distance_from_Residence_to_Work = st.slider('Distance to Office', min_value=0, max_value=200, value=0, step=1)

Service_time = st.number_input('Service Time', value=0)
Age = st.number_input('Age', value=0)
Work_load_Average = st.slider('Work Load', min_value=200000, max_value=400000, value=200000, step=1)
Hit_target = st.number_input('Hit Target', value=0)
Disciplinary_failure = st.selectbox('Disciplinary Failure', df['Disciplinary failure'].unique())
Education = st.selectbox('Education', df['Education'].unique())
Son = st.number_input('Son', value=0)
Social_drinker = st.selectbox('Social Drinker', df['Social drinker'].unique())
Social_smoker = st.selectbox('Social Smoker', df['Social smoker'].unique())
Pet = st.number_input('Pet', value=0)
Height = st.number_input('Height', value=0)
Weight = st.number_input('Weight', value=0)
Body_mass_index = st.number_input('BMI', value=0)

df_input = {
    'Reason_for_absence': Reason_for_absence,
    'Month_of_absence' : Month_of_absence,
    'Day_of_the_week': Day_of_the_week,
    'Seasons': Seasons,
    'Transportation_expense': Transportation_expense,
    'Distance_from_Residence_to_Work': Distance_from_Residence_to_Work,
    'Service_time': Service_time,
    'Age': Age,
    'Work_load_Average/day_': Work_load_Average,
    'Hit_target': Hit_target,
    'Disciplinary_failure': Disciplinary_failure,
    'Education': Education,
    'Son': Son,
    'Social_drinker': Social_drinker,
    'Social_smoker': Social_smoker,
    'Pet': Pet,
    'Weight': Weight,
    'Height': Height,
    'Body_mass_index': Body_mass_index
}

input = pd.DataFrame(df_input, index=[0])


load_model = joblib.load("absent_pred.pkl")

if st.button('Predict'):
    prediction = load_model.predict(input)

    if prediction == 1:
        prediction = 'Absent More than 3 Hours'
    else:
        prediction = 'Absent Less than 3 Hours'

    st.write('The employee will: ')
    st.write(prediction)