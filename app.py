import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# load the trained model and the dropdown options
model = joblib.load("house_price_model.joblib")
with open("options.json") as f:
    options = json.load(f)


def format_price(lakhs):
    # show the price in Lakhs, and in Crores when it is 100 Lakhs or more
    if lakhs >= 100:
        return str(round(lakhs / 100, 2)) + " Cr"
    return str(round(lakhs, 2)) + " Lakhs"


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form = {}

    if request.method == "POST":
        form = request.form.to_dict()
        try:
            bhk = int(form["bhk"])
            area = int(form["area"])
            if bhk < 1 or area < 100:
                raise ValueError

            row = pd.DataFrame([{
                "bhk": bhk,
                "area": area,
                "type": form["type"],
                "region": form["region"],
                "status": form["status"],
                "age": form["age"],
            }])
            price = model.predict(row)[0]
            result = format_price(price)
        except (ValueError, KeyError):
            error = "Please enter a valid BHK (1 or more) and area (100 sqft or more)."

    return render_template("index.html", options=options, result=result, error=error, form=form)


if __name__ == "__main__":
    app.run(debug=True)
