import joblib
import pandas as pd
import csv 
from object.vehicle import Vehicle

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

pipeline = joblib.load('vehicle_price_predictor.pkl')

def predict_price(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Predict the price
    predicted_price = pipeline.predict(input_df)
    
    return predicted_price[0]

user_input1 = {
    'make': 'Mercedes-Benz', 'model': 'C-Class C 300', 'year': 2021, 'mileage': 100000, 'engine': '2.0L I4 16V GDI DOHC Turbo',
    'transmission': '9-Speed Automatic', 'drivetrain': 'Rear-wheel Drive', 'min_mpg': 25, 'max_mpg': 35, 'damaged': 0,
    'first_owner': 1, 'personal': 1, 'turbo': 0, 'cruise': 1, 'nav': 1, 'power_lift': 0, 'back_cam': 1,
    'keyless': 1, 'remote': 1, 'sunroof': 0, 'leather_seats': 1, 'memory_seats': 0, 'smart_info': 1,
    'bluetooth': 1, 'usb': 1, 'heated_seats': 1
}

user_input2 = {
    'make': 'Mercedes-Benz', 'model': 'C-Class C 300', 'year': 2021, 'mileage': 100000, 'engine': '2.0L I4 16V GDI DOHC Turbo',
    'transmission': '9-Speed Automatic', 'drivetrain': 'Rear-wheel Drive', 'min_mpg': 25, 'max_mpg': 35, 'damaged': 0,
    'first_owner': 1, 'personal': 1, 'turbo': 0, 'cruise': 1, 'nav': 1, 'power_lift': 0, 'back_cam': 1,
    'keyless': 1, 'remote': 1, 'sunroof': 0, 'leather_seats': 1, 'memory_seats': 0, 'smart_info': 1,
    'bluetooth': 1, 'usb': 1, 'heated_seats': 1
}

user_input3 = {
    'make': 'Mercedes-Benz', 'model': 'C-Class C 300', 'year': 2018, 'mileage': 100000, 'engine': '2.0L I4 16V GDI DOHC Turbo',
    'transmission': '9-Speed Automatic', 'drivetrain': 'Rear-wheel Drive', 'min_mpg': 25, 'max_mpg': 35, 'damaged': 0,
    'first_owner': 1, 'personal': 1, 'turbo': 0, 'cruise': 1, 'nav': 1, 'power_lift': 0, 'back_cam': 1,
    'keyless': 1, 'remote': 1, 'sunroof': 0, 'leather_seats': 1, 'memory_seats': 0, 'smart_info': 1,
    'bluetooth': 1, 'usb': 1, 'heated_seats': 1
}

predicted_price1 = predict_price(user_input1)
print(f'Predicted Price Low Mile: ${predicted_price1:.2f}')
predicted_price2 = predict_price(user_input2)
print(f'Predicted Price: ${predicted_price2:.2f}')
predicted_price3 = predict_price(user_input3)
print(f'Predicted Price High Mile: ${predicted_price3:.2f}')



