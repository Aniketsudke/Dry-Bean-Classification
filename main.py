from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle

# flask app
app = Flask(__name__)

# load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

beanList = {0: 'BARBUNYA', 1: 'BOMBAY', 2: 'CALI',
            3: 'DERMASON', 4: 'HOROZ', 5: 'SEKER', 6: 'SIRA'}


def get_predicted_value(arr):
    arr_reshaped = arr.reshape(1, -1)
    val = svc.predict(arr_reshaped)[0]
    return beanList[val]

# routes


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        Area = request.form.get('Area')
        MajorAxisLength = request.form.get('MajorAxisLength')
        MinorAxisLength = request.form.get('MinorAxisLength')
        ConvexArea = request.form.get('ConvexArea')
        EquivDiameter = request.form.get('EquivDiameter')
        Solidity = request.form.get('Solidity')
        Roundness = request.form.get('Roundness')
        ShapeFactor1 = request.form.get('ShapeFactor1')
        ShapeFactor2 = request.form.get('ShapeFactor2')
        ShapeFactor3 = request.form.get('ShapeFactor3')
        ShapeFactor4 = request.form.get('ShapeFactor4')

        if Area == '' or MajorAxisLength == '' or MinorAxisLength == '' or ConvexArea == '' or EquivDiameter == '' or Solidity == '' or Roundness == '' or ShapeFactor1 == '' or ShapeFactor2 == '' or ShapeFactor3 == '' or ShapeFactor4 == '':
            message = "Please Fill Correct all field in Numerical Format"
            return render_template('index.html', message=message)
        else:

            arr = np.array([float(Area), 855.283459, float(MajorAxisLength), float(MinorAxisLength), 1.583242, 0.750895, float(ConvexArea), float(EquivDiameter), 0.749733,
                            float(Solidity), float(Roundness), 0.799864, float(ShapeFactor1), float(ShapeFactor2), float(ShapeFactor3), float(ShapeFactor4)])
            predicted = get_predicted_value(arr)
            print(predicted)
            return render_template('index.html', predicted=predicted)

    return render_template('index.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
