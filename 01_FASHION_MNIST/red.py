#Definicion de la red neuronal
import tensorflow as tf 
import matplotlib.pyplot as plt
from data import imagenes_train_normalizadas, etiquetas_train


# creo una red_neuronal secuencial
red_neuronal = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #capa de entrada, aplanar la imagen de 28x28 pixeles en un vector de 784 pixeles
    tf.keras.layers.Dense(128, activation='relu'), #capa oculta con 128 neuronas y funcion de activacion relu
    tf.keras.layers.Dense(10, activation='softmax') #capa de salida con 10 neuronas y funcion de activacion softmax
])

red_neuronal.summary() #muestra un resumen de la red neuronal, capas, parametros y conexiones

#compilacion de la red neuronal, se compila con el optimizador adam que ajusta el learning rate automaticamente,
# la funcion de perdida sparse_categorical_crossentropy que convierte los valores de salida en probabilidades 0-1 
# la metrica accuracy muestra el porcentaje de aciertos
red_neuronal.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

#entrenar a la red neuronal, y lo guardo en una variable para poder ver la evolucion de la perdida y la precision durante el entrenamiento
#validation_split= es que coge un batch del 20% de todos los datos de entrenamiento
historial = red_neuronal.fit(imagenes_train_normalizadas, etiquetas_train, epochs=10, validation_split=0.2)

#historial guarda un diccionario interno con todas las metricas de Pérdida en entrenamiento, Precisión en entrenamiento, Pérdida en validación,Precisión en validación
historial.history.keys() #muestra las claves del historial, que son loss, accuracy, val_loss y val_accuracy

# ---------------------------------------------------------
# 1. GRÁFICA DE PÉRDIDA (LOSS)
# ---------------------------------------------------------
plt.figure() # Crea un "lienzo" en blanco nuevo

# Dibujamos ambas líneas en el mismo lienzo (el 'label' es el nombre para la leyenda)
plt.plot(historial.history['loss'], label='Pérdida en entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida en validación')

# Ponemos los nombres a los ejes y el título
plt.title('Evolución de la Función de Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Valor de Pérdida')

# Activamos la leyenda (el cuadrito que muestra los colores y los labels)
plt.legend() 

# Guardamos la foto terminada (asegúrate de usar la ruta correcta según lo que elegiste antes)
plt.savefig('01_FASHION_MNIST/results/grafica_perdida.png')

plt.plot(historial.history['val_loss'], label='Pérdida en validación') 
plt.savefig('01_FASHION_MNIST/results/perdida_val.png')
# ---------------------------------------------------------
# 2. GRÁFICA DE PRECISIÓN (ACCURACY)
# ---------------------------------------------------------
plt.figure() # Crea OTRA ventana en blanco para no mezclar con la anterior

# Dibujamos las líneas de precisión
plt.plot(historial.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión en validación')

# Ponemos los textos
plt.title('Evolución de la Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Porcentaje de Acierto')
plt.legend()

plt.savefig('01_FASHION_MNIST/results/grafica_precision.png')
plt.plot(historial.history['val_accuracy'], label='Precisión en validación') 
plt.savefig('01_FASHION_MNIST/results/precision_val.png')