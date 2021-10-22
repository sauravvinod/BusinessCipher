import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template, request

app = Flask(__name__)

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


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    data14 = request.form['n']
    data15 = request.form['o']
    data16 = request.form['p']
    data17 = request.form['q']
    data18 = request.form['r']
    data19 = request.form['s']
    data20 = request.form['t']
    data21 = request.form['u']
    data22 = request.form['v']
    data23 = request.form['w']
    data24 = request.form['x']
    data25 = request.form['y']
    data26 = request.form['z']
    data27 = request.form['aa']
    data28 = request.form['ab']
    data29 = request.form['ac']
    data30 = request.form['ad']
    data31 = request.form['ae']
    data32 = request.form['af']
    data33 = request.form['ag']
    data34 = request.form['ah']
    data35 = request.form['ai']
    data36 = request.form['aj']
    data37 = request.form['ak']
    data38 = request.form['al']
    data39 = request.form['am']
    data40 = request.form['an']
    html_form_data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
                      data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25,
                      data26, data27, data28, data29, data30, data31, data32, data33, data34, data35, data36, data37,
                      data38, data39, data40]
    converted_data = list(np.float_(html_form_data))
    input_data = [converted_data]
    df = pd.DataFrame(data=input_data, columns=input_column)
    prediction = model.predict(df)
    predicted_flag = int(round(prediction, 0))
    if predicted_flag:
        output_data = "Internal Claim Settlement"
    else:
        output_data = "External Claim Settlement"
    return render_template('after.html', data=output_data)


if __name__ == "__main__":
    app.run(debug=True)
