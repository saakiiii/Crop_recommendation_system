from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import joblib

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    loaded_rf_model = joblib.load('rf_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')

    # Example prediction using loaded model
    new_data = pandas.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temp], 'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]})
    new_data = loaded_scaler.transform(new_data)  # Scale new data
    prediction = loaded_rf_model.predict(new_data)
    # print(N, P, K, temp, temp, humidity, ph, rainfall)

    # feature_list = [N, P, K, temp, humidity, ph, rainfall]
    # single_pred = np.array(feature_list).reshape(1, -1)

    # scaled_features = ms.transform(single_pred)
    # final_features = sc.transform(scaled_features)
    # prediction = model.predict(final_features)ss

    # crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    #              8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    #              14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    #              19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # if prediction[0] in crop_dict:
    #     crop = crop_dict[prediction[0]]
    #     result = "{} is the best crop to be cultivated right there".format(crop)
    # else:
    #     result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    print("Result:", prediction[0])
    return render_template('index.html',result = prediction[0], img=f"static/{prediction[0]}.jpg")




# python main
if __name__ == "__main__":
    app.run(debug=True)