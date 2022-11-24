import joblib
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import cross_origin

app = Flask(__name__, template_folder="templates")

model_fit = joblib.load(open("./models/com_random_forest.pkl", "rb"))
print("Random Forest Model Loaded")
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
print("Scaler Model Loaded")
encoder = joblib.load(open("./models/encoder.pkl", "rb"))
print("Encoder Model Loaded")
numeric_columns = joblib.load(open("./models/numeric_columns.pkl", "rb"))
print("Numerical column Model Loaded")
categorical_columns = joblib.load(open("./models/categorical_columns.pkl", "rb"))
print("Categorical column Model Loaded")
encoded_columns = joblib.load(open("./models/encoded_column.pkl", "rb"))
print("Encoded column Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("home.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        journey_year =  int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").year)
        #print(f'Date of journey: {journey_day} day, {journey_month} month and {journey_year} year')
        
        # Departure
        dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        #print(f'Departure : {dep_hour} hour and {dep_min} mins')

        # Arrival
        date_arr = request.form["Arrival_Time"]
        arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        #print(f"Arrival : {arrival_hour} hour and {arrival_min} mins")

        # Duration
        duration_hour = abs(arrival_hour - dep_hour)
        duration_minutes = abs(arrival_min - dep_min)
        #print(f"Duration : {duration_hour} hour and {duration_minutes} mins")
        
        # Airline
        Airline = request.form['Airline']
        #print('Airline:',Airline)
        
        #Source
        Source = request.form['Source']
        #print('Source:',Source)
        
        #Destination
        Destination = request.form['Destination']
        #print('Destination:',Destination)
        
        #Total_Stops
        Total_Stops = int(request.form['Stopage'])
        #print('Total_Stops:',Total_Stops)
        
        #Additional_Info
        Additional_Info = request.form['info']
        #print('Additional_Info:',Additional_Info)

    #     ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info',
    #    'journey_day', 'journey_month', 'journey_year', 'dep_hour',
    #    'dep_min', 'arrival_hour', 'arrival_min', 'duration_hour',
    #    'duration_minutes']	
    							
        new_input = {
            'Airline': Airline,
            'Source': Source,
            'Destination': Destination,
            'Total_Stops': Total_Stops,
            'Additional_Info': Additional_Info,
            'journey_day': journey_day,
            'journey_month': journey_month,
            'journey_year': journey_year,
            'dep_hour': dep_hour,
            'dep_min': dep_min,
            'arrival_hour': arrival_hour,
            'arrival_min': arrival_min,
            'duration_hour': duration_hour,
            'duration_minutes': duration_minutes
        }
        #print(new_input)

        def predict_input(input):
            input_df = pd.DataFrame([input])
            input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
            input_df[encoded_columns] = encoder.transform(input_df[categorical_columns])
            X_input = input_df[numeric_columns + encoded_columns]
            pred = model_fit.predict(X_input)[0]
            return pred
        prediction = predict_input(new_input)
        # prediction = round(prediction, 2)
        output = prediction

        if output>0:
                return {
                    "Source":Source,
                    "Destination":Destination,
                    "prediction": output,
                    "status": "success"
                }
        else:
                return render_template("index.html")
    return render_template("index.html")

if __name__=='__main__':
	app.run(debug=True)