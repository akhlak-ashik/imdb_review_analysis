# app.py

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('imdb_sentiment_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the request
        text = request.json['text']

        # Make prediction
        prediction = model.predict([text])[0]

        # Return the prediction
        return jsonify({'sentiment': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
