import requests

url = 'http://localhost:9696/predict'
#url = 'https://wine-quality-ml-0xpm.onrender.com/predict'

wine_data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "is_red": 1
}

response = requests.post(url, json=wine_data).json()

print(f"High quality probability: {response['high_quality_probability']:.4f}")
print(f"¿Is high quality?: {response['is_high_quality']}")