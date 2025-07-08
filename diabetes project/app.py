import pickle
from flask import Flask, request, render_template
import numpy as np


app = Flask(__name__)
model = pickle.load(open('NBModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pregnancies = int(request.form['Pregnancies'])
    glucose = int(request.form['Glucose'])
    bloodpressure = int(request.form['BloodPressure'])
    skinthickness = int(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    diabetespedigreefunction = float(request.form['DiabetesPedigreeFunction'])
    age = int(request.form['Age'])

    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])


    prediction = model.predict(input_data)

    # Determine the result based on the prediction
    if prediction[0] == 0:
        result = 'Non-Diabetic'
    else:
        result = 'Diabetic'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)