import pickle
import pandas as pd
from flask import Flask, request, jsonify

model_file = 'model_rf.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('wine-quality')

@app.route('/predict', methods=['POST'])
def predict():
    wine_data = request.get_json()    
    df = pd.DataFrame([wine_data])

    y_pred = model.predict_proba(df)[0, 1]
    high_quality = y_pred >= 0.5
    
    result = {
        'high_quality_probability': float(y_pred),
        'is_high_quality': bool(high_quality)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)