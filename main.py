import numpy as np 
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import catboost


#load model
app = Flask(__name__)
api = Api(app)

model = joblib.load("joblib_model.sav")

class MakePrediction(Resource):
    @staticmethod
    def post():
        data = request.form.to_dict()
        rooms = data["rooms"]
        location = data["location"]
        
        result = np.exp(model.predict([rooms, location]))
        result = np.round(result, 2)

        return jsonify({
            'Expected Rent is': result
        })
    
    
api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
