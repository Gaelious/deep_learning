import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Capa densa es que se conectan a todas las neuronas de la capa anterior
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)    

modelo.fit(celsius, farenheit, epochs=500, verbose= False)

plt.xlabel('Generacion')
plt.ylabel('Magnitud de la perdida')
plt.plot(modelo.history.history['loss'])