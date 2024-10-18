from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

brand_encoder = joblib.load('encoders/brand_encoder.pkl')
fuel_type_encoder = joblib.load('encoders/fuel_type_encoder.pkl')
transmission_encoder = joblib.load('encoders/transmission_encoder.pkl')
ext_col_encoder = joblib.load('encoders/ext_col_encoder.pkl')
int_col_encoder = joblib.load('encoders/int_col_encoder.pkl')
accident_encoder = joblib.load('encoders/accident_encoder.pkl')
clean_title_encoder = joblib.load('encoders/clean_title_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/predict')
def predict():
    brand = request.form.get('brand')
    year = int(request.form.get('year'))
    mileage = int(request.form.get('mileage'))
    fuel_type = request.form.get('fuel_type')
    transmission = request.form.get('transmission')
    ext_col = request.form.get('ext_col').title()
    int_col = request.form.get('int_col').title()
    accident = request.form.get('accident')
    clean_title = request.form.get('clean_title')

    brand_encoded = brand_encoder.transform([brand])[0]
    fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
    transmission_encoded = transmission_encoder.transform([transmission])[0]
    ext_col_encoded = ext_col_encoder.transform([ext_col])[0]
    int_col_encoded = int_col_encoder.transform([int_col])[0]
    accident_encoded = accident_encoder.transform([accident])[0]
    clean_title_encoded = clean_title_encoder.transform([clean_title])[0]

    input_array = np.array([[brand_encoded, year, mileage, fuel_type_encoded, transmission_encoded, ext_col_encoded, int_col_encoded,  accident_encoded, clean_title_encoded]])

    model = tf.keras.models.load_model('model.keras')
    prediction = model.predict(input_array)

    return render_template('result.html.jinja', prediction=prediction[0][0])