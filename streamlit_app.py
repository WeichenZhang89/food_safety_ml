import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import base64

def encode_decode_image(path):
    with open(path, "rb") as image_file:
        read_image = image_file.read()
    encode_image = base64.b64encode(read_image)
    decode_rep = encode_image.decode()
    return decode_rep

data = pd.read_csv('./mushrooms.csv')

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
data = data.infer_objects(copy=False)

X = data.drop(columns='class')
y = data['class']

model = CategoricalNB()
model.fit(X, y)

st.title("Food Safety")
st.write("Select the option for characteristic")

input_data = {}
option_des = {
        'cap-shape': 'bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s',
        'cap-surface': 'fibrous=f, grooves=g, scaly=y, smooth=s',
        'cap-color': 'brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y',
        'bruises': 'bruises=t, no=f',
        'odor': 'almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s',
        'gill-attachment': 'attached=a, descending=d, free=f, notched=n',
        'gill-spacing': 'close=c, crowded=w, distant=d',
        'gill-size': 'broad=b, narrow=n',
        'gill-color': 'black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y',
        'stalk-shape': 'enlarging=e, tapering=t',
        'stalk-root': 'bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?',
        'stalk-surface-above-ring': 'fibrous=f, scaly=y, silky=k, smooth=s',
        'stalk-surface-below-ring': 'fibrous=f, scaly=y, silky=k, smooth=s',
        'stalk-color-above-ring': 'brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y',
        'stalk-color-below-ring': 'brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y',
        'veil-type': 'partial=p, universal=u',
        'veil-color': 'brown=n, orange=o, white=w, yellow=y',
        'ring-number': 'none=n, one=o, two=t',
        'ring-type': 'cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z',
        'spore-print-color': 'black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y',
        'population': 'abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y',
        'habitat': 'grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d'
    }
for feature in X.columns:
    input_data[feature] = st.selectbox(
        f"{feature} ({option_des[feature]})",
        options=list(replace_dict[feature].keys())
    )

input_values = [replace_dict[feature][input_data[feature]] for feature in X.columns]

input_df = pd.DataFrame([input_values], columns=X.columns)

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    if prediction[0] == 1:
        st.success("the mushroom is edible.")
    else:
        st.error("the mushroom is toxic.")
    st.write(f"prediction probabilities: edible: {prediction_proba[0][1]:.2f}, toxic: {prediction_proba[0][0]:.2f}")

def add_background(path):
    encoded_image = encode_decode_image(path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_background('mushroom_replace.jpeg')
