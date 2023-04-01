import random
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from KB_lab03_activation_function import tanh, tanh_der
from KB_lab_03_layer import Network, FCLayer, ActivationLayer, mse, mse_prime


def create_node(size, name):
    mean = random.uniform(-1, 1)
    standard_deviation = random.uniform(0, 0.1)
    classes = np.full(size, name)

    rng = np.random.default_rng()
    x = rng.normal(mean, standard_deviation, size)
    y = rng.normal(pow(standard_deviation, 2), standard_deviation, size)
    return [np.array((x, y)).T], classes


x2, classes2 = create_node(5, 1)
x1, classes1 = create_node(5, 0)

x_latest = np.array(np.concatenate((x1, x2)))
classes_latest = np.concatenate((classes1, classes2))

st.subheader('Katarzyna Bielicka AI Lab03 - Shallow Neural Network')

with st.form("neuron_form"):
    size_slider_val = st.slider("Node size", min_value=100, max_value=400, step=50)
    layer_slider_val = st.slider("Number of Hidden layers", min_value=1, max_value=3)
    neuron_slider_val = st.slider("Number of neurons", min_value=1, max_value=10)

    submitted = st.form_submit_button("Submit")
    if submitted:
        net = Network()
        net.add(FCLayer(2, neuron_slider_val))
        net.add(ActivationLayer(tanh, tanh_der))
        temp = neuron_slider_val
        for i in range(layer_slider_val):
            print(temp//2)
            net.add(FCLayer(temp, temp//2))
            net.add(ActivationLayer(tanh, tanh_der))
            temp = temp //2
        net.add(FCLayer(temp, 1))
        net.add(ActivationLayer(tanh, tanh_der))
        net.use(mse, mse_prime)
        net.fit(x_latest, classes_latest, epochs=1000, learning_rate=0.1)

        new_x1, new_classes1 = create_node(size_slider_val, 'class 1')
        new_x2, new_classes2 = create_node(size_slider_val, 'class 2')

        x_latest = np.concatenate((new_x1, new_x2))
        classes_latest = np.concatenate((new_classes1, new_classes2))

        x_min, x_max = x_latest[:, 0].min() - 1, x_latest[:, 0].max() + 1
        y_min, y_max = x_latest[:, 1].min() - 1, x_latest[:, 1].max() + 1
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = net.predict(np.c_[xx.ravel(), yy.ravel()])
        z = np.reshape(z, xx.shape)
        fig, ax = plt.subplots()
        x1 = np.array(list(new_x1)).reshape(size_slider_val, 2)
        x2 = np.array(list(new_x2)).reshape(size_slider_val, 2)
        ax.scatter(x1[:, 0], x1[:, 1], c='b', label='class 1')
        ax.scatter(x2[:, 0], x2[:, 1], c='r', label='class 2')
        ax.contourf(yy, xx, z, cmap='RdGy', alpha=0.3)
        ax.legend(loc='upper left')
        st.pyplot(fig)
