# deploy_Flask.py

from flask import Flask, render_template, request
import pickle
import dataset

app = Flask(__name__)

# Load the model
def load_model():
    """Load the trained model from a file"""
    with open('trained_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

@app.route('/')
def home():
    """Render the index page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request"""
    try:
        # Get the input from the form
        sms_input = request.form['sms']

        # Ensure the input is a string
        sms_input = str(sms_input)

        cleaned_sms = dataset.clean_text(sms_input, apply_stemming=True, apply_lemmatization=False)

        # Make a prediction using the model
        result = model.predict([cleaned_sms])[0]

        # Render the result
        return render_template('index.html', result=result)
    except ValueError:
        # Handle invalid input
        return render_template('index.html', result='Invalid input')

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8000)