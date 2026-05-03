from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load correct model file
model = pickle.load(open("iris_model.pkl", "rb"))

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# ---------------- PREDICTION ---------------- #

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        features = np.array([[sl, sw, pl, pw]])

        result = model.predict(features)[0]

        # ✅ Match YOUR image names
        if result == 0:
            flower = "Setosa"
            image = "setosa.jpg"
        elif result == 1:
            flower = "Versicolor"
            image = "versicolor.webp"
        else:
            flower = "Virginica"
            image = "virginica.webp"

        return render_template(
            'prediction.html',
            prediction_text=f"The predicted flower is {flower}",
            flower_image=image
        )

    except Exception as e:
        return render_template(
            'prediction.html',
            prediction_text="Error: Please enter valid numeric values"
        )


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)