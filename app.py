import pathlib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import tensorflow as tf


def get_input():
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#ffffff",
        height=200,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    button_disabled = True
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img_array = np.array(img)

        if not np.all(img_array == 255):
            button_disabled = False

    read = st.button('Read', disabled=button_disabled)
    if read:
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        return img
    else:
        st.write('Please write a math symbol!')
        return None


def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model


def get_prediction(img):
    target_dir = 'dataset'
    data_dir = pathlib.Path(target_dir)
    class_names = list(sorted([item.name for item in data_dir.glob("*")]))
    class_names[10] = '+'
    class_names[11] = '/'
    class_names[12] = '*'
    class_names[13] = '-'

    model = load_model()
    probs = model.predict(img)
    prediction = class_names[int(np.argmax(probs, 1))]
    return probs, prediction


def show_page():
    st.title("Handwritten Math Symbols Recognition")
    input_data = get_input()
    if input_data is not None:
        probs, prediction = get_prediction(input_data)
        if probs is not None and prediction is not None:
            st.write(f"You just wrote: {prediction}")


show_page()
