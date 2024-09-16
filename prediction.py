from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_location = joblib.load('label_encoder_location.pkl')
label_encoder_amenities = joblib.load('label_encoder_amenities.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()

            
            required_keys = ['area', 'rooms', 'location', 'amenities']
            for key in required_keys:
                if key not in data:
                    return jsonify({'error': f"Missing key: {key}"}), 400

            
            area = float(data['area'])
            rooms = int(data['rooms'])
            location = data['location']
            amenities = data['amenities']

           
            if location not in label_encoder_location.classes_:
                return jsonify({'error': f"Invalid location: {location}. Valid labels: {label_encoder_location.classes_.tolist()}"}), 400
            if amenities not in label_encoder_amenities.classes_:
                return jsonify({'error': f"Invalid amenities: {amenities}. Valid labels: {label_encoder_amenities.classes_.tolist()}"}), 400

            
            location_encoded = label_encoder_location.transform([location])[0]
            amenities_encoded = label_encoder_amenities.transform([amenities])[0]

            
            input_data = pd.DataFrame({
                'Acres': [area],
                'Rooms': [rooms],
                'Location': [location_encoded],
                'Amenities': [amenities_encoded]
            })

            
            input_data_scaled = scaler.transform(input_data)

           
            prediction = model.predict(input_data_scaled)
            predicted_price = prediction[0]

            return jsonify({'predicted_price': predicted_price})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif request.method == 'GET':
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
