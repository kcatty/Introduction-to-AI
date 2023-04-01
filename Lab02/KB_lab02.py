import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from KB_lab02_neuron import Perceptron


def create_node(size, name):
    mean = random.uniform(-1, 1)
    standard_deviation = random.uniform(0, 0.1)
    classes = np.full(size, name)

    rng = np.random.default_rng()
    x = rng.normal(mean, standard_deviation, size)
    y = rng.normal(pow(standard_deviation, 2), standard_deviation, size)
    return np.vstack((x, y)).T, classes


x2, classes2 = create_node(100, 1)
x1, classes1 = create_node(100, 0)

x_latest = np.concatenate((x1, x2))
classes_latest = np.concatenate((classes1, classes2))
options = ('Heaviside', 'Logistic', 'sin', 'tanh', 'sign', 'ReLu', 'Leaky ReLu')

st.subheader('Katarzyna Bielicka AI Lab02')
with st.form("neuron_form"):
    size_slider_val = st.slider("Node size", min_value=100, max_value=400, step=50)
    node1_slider_val = st.slider("Number of nodes class 1", min_value=1, max_value=3)
    node2_slider_val = st.slider("Number of nodes class 2", min_value=1, max_value=3)
    index = st.selectbox("selectbox", range(len(options)), format_func=lambda x: options[x])

    submitted = st.form_submit_button("Submit")
    if submitted:
        neuron = Perceptron(learning_rate=0.2, n_iters=100, function_option=index)
        neuron.fit(x_latest, classes_latest)
        new_x1, new_x2, new_classes1 = [], [], []
        new_classes2 = []
        for number in range(node1_slider_val):
            a, c = create_node(size_slider_val, 'class 1')
            new_x1 = np.concatenate((new_x1, a))
            new_classes1 = np.append(new_classes1, c)

        for number in range(node2_slider_val):
            a, c = create_node(size_slider_val, 'class 2')
            new_x2 = np.concatenate((new_x2, a))
            new_classes2 = np.append(new_classes2, c)

        x_latest = np.concatenate((new_x1, new_x2))
        classes_latest = np.concatenate((new_classes1, new_classes2))

        x = np.arange(-1, 1, 0.001)
        y = np.arange(-1, 1, 10)

        x_min, x_max = x_latest[:, 0].min() - 1, x_latest[:, 0].max() + 1
        y_min, y_max = x_latest[:, 1].min() - 1, x_latest[:, 1].max() + 1
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = neuron.predict(np.c_[xx.ravel(), yy.ravel()])

        z = z.reshape(xx.shape)
        fig, ax = plt.subplots()
        ax.scatter(new_x1[:, 0], new_x1[:, 1], c='b', label='class 1')
        ax.scatter(new_x2[:, 0], new_x2[:, 1], c='r', label='class 2')
        ax.contourf(xx, yy, z, cmap='RdGy', alpha=0.3)
        ax.legend(loc='upper left')
        st.pyplot(fig)
