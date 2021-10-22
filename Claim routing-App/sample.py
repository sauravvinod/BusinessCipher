import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask

model = mlflow.sklearn.load_model("./model/logit_v1")

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

html_form_data = ['1', '1', '1', '30', '1', '5', '1', '555', '1', '1', '1000', '1', '1', '0', '0', '0',
                  '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0',
                  '0', '0', '0', '1', '0', '0']
converted_data = list(np.float_(html_form_data))
input_data = [converted_data]

df = pd.DataFrame(data=input_data, columns=input_column)

print(df.head())

pred = model.predict(df)

print(pred)
