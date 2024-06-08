from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your trained model
model = GaussianNB()

data = pd.read_csv('./mushrooms.csv')
data.columns = [col.strip() for col in data.columns]
null_data_count = data.isnull().sum().sum()

if (null_data_count == 0):
    print("No missing data")
else:
    print(f"{null_data_count} missing data.")

replace_dict = {
    'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
    'cap-surface': {'f': 0, 'g': 1, 'y': 2, 's': 3},
    'cap-color': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
    'bruises': {'t': 0, 'f': 1},
    'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
    'gill-attachment': {'a': 0, 'd': 1, 'f': 2, 'n': 3},
    'gill-spacing': {'c': 0, 'w': 1, 'd': 2},
    'gill-size': {'b': 0, 'n': 1},
    'gill-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
    'stalk-shape': {'e': 0, 't': 1},
    'stalk-root': {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6},
    'stalk-surface-above-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
    'stalk-surface-below-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
    'stalk-color-above-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'stalk-color-below-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'veil-type': {'p': 0, 'u': 1},
    'veil-color': {'n': 0, 'o': 1, 'w': 2, 'y': 3},
    'ring-number': {'n': 0, 'o': 1, 't': 2},
    'ring-type': {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
    'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
    'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
    'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6},
    'class': {'p': 0, 'e': 1}
}

data.replace(replace_dict, inplace=True)

X = data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 
          'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 
          'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
          'spore-print-color', 'population', 'habitat']]
y = data['class']

model.fit(X, y)

@app.route('/classify', methods=['POST'])
def classify():
    content = request.json
    features = [
        content['cap-shape'],
        content['cap-surface'],
        content['cap-color'],
        content['bruises'],
        content['odor'],
        content['gill-attachment'],
        content['gill-spacing'],
        content['gill-size'],
        content['gill-color'],
        content['stalk-shape'],
        content['stalk-root'],
        content['stalk-surface-above-ring'],
        content['stalk-surface-below-ring'],
        content['stalk-color-above-ring'],
        content['stalk-color-below-ring'],
        content['veil-type'],
        content['veil-color'],
        content['ring-number'],
        content['ring-type'],
        content['spore-print-color'],
        content['population'],
        content['habitat']
    ]
    prediction = model.predict([features])[0]
    result = "Edible" if prediction == 1 else "Poisonous"
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)
