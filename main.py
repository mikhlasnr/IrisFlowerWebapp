import streamlit as st
from keras.models import load_model
import numpy as np


model = load_model("model.h5")
labels = np.load("labels.npy")

st.image("./images/img-bunga.png")
st.title("Selamat datang di Aplikasi Prediksi Bunga")
st.subheader("Masukkan Data")
a = float(st.number_input("Panjang sepal dalam cm"))
b = float(st.number_input("Lebar sepal dalam cm"))
c = float(st.number_input("Panjang petal dalam cm"))
d = float(st.number_input("Lebar petal dalam cm"))

btn = st.button("Prediksi")

if btn:
	pred = model.predict(np.array([a, b, c, d]).reshape(1, -1))
	pred = labels[np.argmax(pred)]
	st.subheader(pred)

	if pred == "Iris Setosa":
		st.image("./images/setosa.jpg")
	elif pred == "Iris Versicolour":
		st.image("./images/versicolor.jpg")
	else:
		st.image("./images/verginca.jpg")
