# Import libraries
import numpy as np
from flask import Flask, request
import pickle as pk

app = Flask(__name__)

model = pk.load(open("model.pkl", "rb"))

@app.route("/api/home", methods=['GET'])
def home():
   return "Hello, Flask!"

@app.route("/api", methods = ['POST'])
def get_campaign_response():
    #get data from json
    data = request.get_json(force=True)

    array = []

    for app in data:
        array.append(data[app])
    
    npArray = np.array([array])

    print(npArray.shape)
    #make prediction
    prediction = model.predict(npArray)
    output = prediction[0]
    return str(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)