# Import libraries
import numpy as np
from flask import Flask, request, jsonify
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

    #make prediction
    prediction = model.predict([[np.array(data['Income', 'MntMeatProducts', 'MntFishProducts', 'Teenhome_1',
       'NumDealsPurchases_1', 'NumWebVisitsMonth_19', 'NumDealsPurchases_15',
       'Marital_Status_Married', 'Marital_Status_Together',
       'Marital_Status_Divorced', 'Marital_Status_Single',
       'NumWebVisitsMonth_4', 'NumDealsPurchases_2', 'NumDealsPurchases_3',
       'Kidhome_2', 'NumWebVisitsMonth_9', 'NumWebPurchases_1',
       'NumWebPurchases_2', 'NumWebVisitsMonth_2', 'NumDealsPurchases_8',
       'Marital_Status_Widow', 'NumWebVisitsMonth_14', 'NumCatalogPurchases_4',
       'NumCatalogPurchases_1', 'NumWebPurchases_27', 'NumCatalogPurchases_5',
       'AcceptedCmp3_1', 'NumCatalogPurchases_6', 'NumDealsPurchases_7',
       'NumCatalogPurchases_10', 'NumCatalogPurchases_11',
       'NumDealsPurchases_9', 'NumWebPurchases_9', 'NumDealsPurchases_13',
       'NumDealsPurchases_10', 'NumWebPurchases_10', 'NumWebVisitsMonth_13',
       'NumWebPurchases_25', 'NumWebPurchases_23'])]])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

#server request
#import requests
#url = 'http://localhost:5000/api'
#r = requests.post(url,json={'exp':1.8,})
#print(r.json())