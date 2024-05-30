import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import csv
from object.vehicle import Vehicle
import joblib

vehicles = []

with open('C:/Users/jiheo/OneDrive/Documents/School/WGU/Capstone/vehicle-prediction-mlm/data/train.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        columns = [row[0], row[1], row[2], row[3], row[4],  row[6], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[26], row[27], row[29], row[30], row[31], row[32], row[35]]
        if all(columns) and not row[35] == 'ot Priced':
            vehicle = Vehicle(*columns)
            vehicles.append(vehicle)
    
data = {
    'make': [v.make for v in vehicles],
    'model': [v.model for v in vehicles],
    'year': [v.year for v in vehicles],
    'mileage': [v.mileage for v in vehicles],
    'engine': [v.engine for v in vehicles],
    'transmission': [v.transmission for v in vehicles],
    'drivetrain': [v.drivetrain for v in vehicles],
    'min_mpg': [v.min_mpg for v in vehicles],
    'max_mpg': [v.max_mpg for v in vehicles],
    'damaged': [v.damaged for v in vehicles],
    'first_owner': [v.first_owner for v in vehicles],
    'personal': [v.personal for v in vehicles],
    'turbo': [v.turbo for v in vehicles],
    'cruise': [v.cruise for v in vehicles],
    'nav': [v.nav for v in vehicles],
    'power_lift': [v.power_lift for v in vehicles],
    'back_cam': [v.back_cam for v in vehicles],
    'keyless': [v.keyless for v in vehicles],
    'remote': [v.remote for v in vehicles],
    'sunroof': [v.sunroof for v in vehicles],
    'leather_seats': [v.leather_seats for v in vehicles],
    'memory_seats': [v.memory_seats for v in vehicles],
    'smart_info': [v.smart_info for v in vehicles],
    'bluetooth': [v.bluetooth for v in vehicles],
    'usb': [v.usb for v in vehicles],
    'heated_seats': [v.heated_seats for v in vehicles],
    'price': [v.price for v in vehicles]
}

df = pd.DataFrame(data)

#Splits the data into all the categories and the price
X = df.drop(columns='price')
y = df['price']

#List of categorical and numerical columns
categorical_cols = ['make', 'model', 'engine', 'transmission', 'drivetrain']
numerical_cols = ['year', 'mileage', 'min_mpg', 'max_mpg', 'damaged', 'first_owner', 'personal', 
                  'turbo', 'cruise', 'nav', 'power_lift', 'back_cam', 'keyless', 'remote', 
                  'sunroof', 'leather_seats', 'memory_seats', 'smart_info', 'bluetooth', 'usb', 'heated_seats']

# Preprocessing for numerical data
numerical_transformer = StandardScaler()


# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)



# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')


joblib.dump(pipeline, 'vehicle_price_predictor.pkl')

print('finished')

def hash_categorical_columns(df, columns):
    mappings = {}
    for column in columns:
        unique_value = df[column].unique()
        value_map = {val: i for i, val in enumerate(unique_value)}
        mappings[column] = value_map
        df[column] = df[column].map(value_map)
    return df, mappings
