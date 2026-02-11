from flask import Flask, render_template, request
import pickle
import numpy as np

# 1. Initialize Flask
app = Flask(__name__)

# 2. Load the trained model
model = pickle.load(open("model.pkl", "rb"))


# 3. Home page – input form
@app.route('/')
def home():
    return render_template("index.html")


# 4. Predict page – shows result
@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form as float numbers
    features = [float(x) for x in request.form.values()]

    # Convert to array for model
    final_features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)

    # Show result
    return render_template('predict.html', prediction_text=f'Predicted Class: {prediction[0]}')


# 5. Run app
if __name__ == "__main__":
    app.run(debug=True)
