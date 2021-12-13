from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='web')


@app.route('/')
def student():
    return render_template("home.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('model_negative.sav')
    result = loaded_model.predict(to_predict)
    new_result = result[0] / 100 - 1
    to_predict2 = np.array(new_result).reshape(-1, 1)
    loaded_model2 = joblib.load('model.sav')
    result = loaded_model2.predict(to_predict2)
    return result[0]


@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = round(float(ValuePredictor(to_predict_list)), 2)
        print(result)
        if result >= 0:
            return render_template("home.html", result='aman')
        else :
            return render_template("home.html", result='tidak aman')


if __name__ == '__main__':
    app.run(debug=True)