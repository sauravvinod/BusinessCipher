import pandas as pd

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import mlflow.sklearn

import warnings

warnings.filterwarnings('ignore')

data_df = pd.read_excel("./data/Final Dataset.xlsx")
input_column = ['Main Hospital Expenses(y/n)', 'Pre& Post hospital Expense (y/n)', 'Gender ', 'Age',
                'Are you previously covered by any insurance', 'No of days admitted', 'Disease (y/n)', 'Expenses',
                'Are you  covered by any top policy', 'Are you covered in any other Mediclaim/Helath Insurance',
                'Annual Premium', 'Does the insurer have internal Claim Process(Y/N)',
                'Is the Hospital partnered with the Insurance company',
                'Name of the Hospital_BAPTIST HEALTH MEDICAL CENTER - LR',
                'Name of the Hospital_BAPTIST MEDICAL CENTER SOUTH',
                'Name of the Hospital_CALLAHAN EYE FOUNDATION HOSP', 'Name of the Hospital_COOPER GREEN MERCY HOSPITAL',
                'Name of the Hospital_DCH REGIONAL MEDICAL CENTER', 'Name of the Hospital_HUNTSVILLE HOSPITAL',
                'Name of the Hospital_INFIRMARY WEST', 'Name of the Hospital_MOBILE INFIRMARY MEDICAL CENTER',
                'Name of the Hospital_PRINCETON BAPTIST MEDICAL CENTER',
                'Name of the Hospital_PROVIDENCE ALASKA MEDICAL CENTER', 'Name of the Hospital_PROVIDENCE HOSPITAL',
                'Name of the Hospital_ST VINCENTS BIRMINGHAM', 'Name of the Hospital_ST. VINCENTS EAST',
                'Name of the Hospital_THE CHILDRENS HOSPITAL OF ALABAMA', 'Name of the Hospital_TRINITY MEDICAL CENTER',
                'Name of the Hospital_UNIV OF SOUTH ALABAMA MEDICAL CENTER',
                'Name of the Hospital_UNIVERSITY OF ALABAMA HOSPITAL',
                'Name of the Hospital_USA CHILDRENS AND WOMENS HOSPITAL',
                'Name of the Hospital_VAUGHAN REGIONAL MEDICAL CENTER', 'Type of room admitted_Deluxe',
                'Type of room admitted_Premium Deluxe', 'Type of room admitted_Premium Twin Sharing',
                'Type of room admitted_Suite', 'Type of room admitted_Twin Sharing', 'Type of Disease/Injury_Acute',
                'Type of Disease/Injury_Chronic', 'Type of Disease/Injury_Semi-Acute']


X_features = list(data_df.columns)
X_features.remove('Is_it_routed_through_TPA')
X_features.remove('UHID Number')
X_features.remove('Policy number')
X_features.remove('Policy CSL/Max Limit')
X_features.remove('Claim Amount')

encoded_data_df = pd.get_dummies(data_df[X_features], drop_first=True)
list(encoded_data_df.columns)

Y = data_df.Is_it_routed_through_TPA
X = sm.add_constant(encoded_data_df)
X = X[input_column]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

logit = sm.Logit(y_train, X_train)
logit_model = logit.fit()

print(list(X_train.columns))

mlflow.sklearn.save_model(logit_model, "../model/logit_v1")
