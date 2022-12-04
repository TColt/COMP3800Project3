import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
knn_model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@flask_app.route("/")
def index():
	return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])
def prediction():
    features = [int(x) for x in request.form.values()]
    feat_array = [np.array(features)]
    scaled_array = scaler.transform(feat_array)
    result = knn_model.predict(scaled_array)
    return render_template("index.html", x=death_result(result))
    
def death_result(array):
    result = ""
    if array[0] == 1:
        result = "Death Event"
    else:
        result = "Alive"
    return result

if __name__ == "__main__":
	flask_app.run(debug =True)
 
