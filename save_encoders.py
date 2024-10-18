import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('data_filled.csv')

brand_encoder = LabelEncoder()
fuel_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
ext_col_encoder = LabelEncoder()
int_col_encoder = LabelEncoder()
accident_encoder = LabelEncoder()
clean_title_encoder = LabelEncoder()

data['brand'] = brand_encoder.fit(data['brand'])
data['fuel_type'] = fuel_type_encoder.fit(data['fuel_type'])
data['transmission'] = transmission_encoder.fit(data['transmission'])
data['ext_col'] = ext_col_encoder.fit(data['ext_col'])
data['int_col'] = int_col_encoder.fit(data['int_col'])
data['accident'] = accident_encoder.fit(data['accident'])
data['clean_title'] = clean_title_encoder.fit(data['clean_title'])

joblib.dump(brand_encoder, 'encoders/brand_encoder.pkl')
joblib.dump(fuel_type_encoder, 'encoders/fuel_type_encoder.pkl')
joblib.dump(transmission_encoder, 'encoders/transmission_encoder.pkl')
joblib.dump(ext_col_encoder, 'encoders/ext_col_encoder.pkl')
joblib.dump(int_col_encoder, 'encoders/int_col_encoder.pkl')
joblib.dump(accident_encoder, 'encoders/accident_encoder.pkl')
joblib.dump(clean_title_encoder, 'encoders/clean_title_encoder.pkl')