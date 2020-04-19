import pickle as pkl

import numpy as np
from flask import Flask, render_template, request

app = Flask(
    __name__, template_folder="/home/hackerearth/Desktop/rest_api_demo/template"
)


@app.route("/")
def index():
    return render_template("structure.html")


@app.route("/prediction/", methods=["GET", "POST"])
def makecalc():
    data = request.form["data"]
    data = np.array(eval(data))
    prediction = np.array2string(model.predict(data))
    return """<h1>The prediction is : {}</h1>""".format(prediction)


if __name__ == "__main__":
    model = pkl.load(open("final_prediction.pickle", "rb"))
    app.run(debug=True, host="0.0.0.0")
