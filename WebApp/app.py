from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/apiEndpoint', methods=['POST'])
def apiEndpoint():
     data = request.get_json(force=True)
     year_model = data['year_model']
     total_kms_car_ran = data['total_kms_car_ran']
     car_company = data['car_company']
     fiscal_power = data['fiscal_power']
     fuel_type = data['fuel_type']
     user_input = {'year_model':year_model, 'total_kms_car_ran':total_kms_car_ran, 'fiscal_power':fiscal_power, 'fuel_type':fuel_type, 'car_company':car_company}
     
     one_hot_data = input_to_one_hot(user_input)
     predict_request = gbr.predict([one_hot_data])
     price_pred = predict_request[0]
     price_pred = round(price_pred, 2)
     #print(price_pred)
     return jsonify(price_pred)

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(61)
    # set the numerical input as they are
    enc_input[0] = data['year_model']
    enc_input[1] = data['total_kms_car_ran']
    enc_input[2] = data['fiscal_power']
    ##################### car_company #########################
    # get the array of car_companys categories
    cols = ['year_model', 'total_kms_car_ran', 'fiscal_power', 'fuel_type_Diesel',
       'fuel_type_Electrique', 'fuel_type_Essence', 'fuel_type_LPG',
       'car_company_Acura', 'car_company_Alfa Romeo', 'car_company_Audi', 'car_company_Autres', 'car_company_BMW',
       'car_company_BYD', 'car_company_Bentley', 'car_company_Cadillac', 'car_company_Changhe',
       'car_company_Chery', 'car_company_Chevrolet', 'car_company_Chrysler', 'car_company_Citroen',
       'car_company_Dacia', 'car_company_Daewoo', 'car_company_Daihatsu', 'car_company_Dodge', 'car_company_Fiat',
       'car_company_Ford', 'car_company_Foton', 'car_company_GMC', 'car_company_Geely', 'car_company_Honda',
       'car_company_Hummer', 'car_company_Hyundai', 'car_company_Infiniti', 'car_company_Isuzu',
       'car_company_Jaguar', 'car_company_Jeep', 'car_company_Kia', 'car_company_Land Rover', 'car_company_Lexus',
       'car_company_Maserati', 'car_company_Mazda', 'car_company_Mercedes-Benz', 'car_company_Mitsubishi',
       'car_company_Nissan', 'car_company_Opel', 'car_company_Peugeot', 'car_company_Pontiac',
       'car_company_Porsche', 'car_company_Renault', 'car_company_Rover', 'car_company_Seat', 'car_company_Skoda',
       'car_company_Ssangyong', 'car_company_Suzuki', 'car_company_Toyota', 'car_company_UFO',
       'car_company_Volkswagen', 'car_company_Volvo', 'car_company_Zotye', 'car_company_lancia',
       'car_company_mini']

    # redefine the the user inout to match the column name
    redefinded_user_input = 'car_company_'+data['car_company']
    # search for the index in columns name list 
    car_company_column_index = cols.index(redefinded_user_input)
    #print(car_company_column_index)
    # fullfill the found index with 1
    enc_input[car_company_column_index] = 1
    ##################### Fuel Type ####################
    # redefine the the user inout to match the column name
    redefinded_user_input = 'fuel_type_'+data['fuel_type']
    # search for the index in columns name list 
    fuelType_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[fuelType_column_index] = 1
    return enc_input

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    year_model = result['year_model']
    total_kms_car_ran = result['total_kms_car_ran']
    car_company = result['car_company']
    fiscal_power = result['fiscal_power']
    fuel_type = result['fuel_type']

    user_input = {'year_model':year_model, 'total_kms_car_ran':total_kms_car_ran, 'fiscal_power':fiscal_power, 'fuel_type':fuel_type, 'car_company':car_company}
    
    print(user_input)
    a = input_to_one_hot(user_input)
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    return json.dumps({'price':price_pred});

if __name__ == '__main__':
    app.run(port=8090, debug=True)






