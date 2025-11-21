from flask import Flask, render_template, request
import joblib

# Load GBC model and vectorizer
GBC = joblib.load("GBC_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    # Vectorize input text
    new_xv_test = vectorizer.transform([text])

    # Model prediction
    pred = GBC.predict(new_xv_test)[0]

    # Convert to readable label
    result = "Fake News" if pred == 1 else "Real News"

    return render_template("index.html", result=result, news=text)

if __name__ == "__main__":
    app.run(debug=True)
