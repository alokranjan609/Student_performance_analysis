from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        sleep_hours = float(request.form["sleep_hours"])

        # Convert input to NumPy array
        input_data = np.array([[hours_studied, previous_scores, sleep_hours]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction=f"{prediction:.2f}")

    except Exception as e:
        return render_template("index.html", prediction="Invalid input. Please enter valid numbers.")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
