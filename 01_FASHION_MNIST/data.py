# cargar y preparar datos
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.keras.datasets.fashion_mnist.load_data()

#las tuplas son tensores de numpy
(imagenes_train, etiquetas_train), (imagenes_test, etiquetas_test) = tf.keras.datasets.fashion_mnist.load_data() #devuelve 2 tuplas, una para entrenar y otra para practicar, cada tupla cntiene el dato y la etiqueta


# 1. Ver las dimensiones (shapes) de las cuatro variables osea las matrices ya que son tensores
print("\nShape de imagenes_train:", imagenes_train.shape)#una matriz y otra matriz de 2 dimensiones
print("Shape de etiquetas_train:", etiquetas_train.shape)
print("Shape de imagenes_test:", imagenes_test.shape)
print("Shape de etiquetas_test:", etiquetas_test.shape)

# 2. Ver el tipo de dato informático que usan 
print("\nTipo de dato de las imágenes:", imagenes_train.dtype)
print("Tipo de dato de las etiquetas:", etiquetas_train.dtype)

# 3. Comprobar los valores de los píxeles
print("\nValor mínimo del píxel:", imagenes_train.min())#significa negro 0 
print("Valor máximo del píxel:", imagenes_train.max()) #significa blanco 100

nombre_ropa = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Cazadora",
                "Sandalia", "Camisa", "Zapato deportivo", "Bolso", "Bota"] # cada indice corresponde a una categoria ya que los datasheets convierten los nombre en numeros del 0 al 9

#ahora normalizo las imagenes en vez de 0-255 de 0-1 ya que asi converge mejor lo pongo con decimal para obligar a cambiar el tipo de dato a float,
#ya que la red necesita decimales para el descenso del gradiente
 
imagenes_train_normalizadas = imagenes_train/255.0
imagenes_test_normalizadas = imagenes_test/255.0

print("\nTipo de dato de las imágenes normalizadas:", imagenes_train_normalizadas.dtype)

plt.figure(figsize=(10,10))# Crear una pantalla nueva especificando el tamaño


#este bucle le uso para los primeros 25 datos de los 60000 que tiene el dataset
for i in range(25):

    # Crear una sub-cuadrícula de 5 filas y 5 columnas, y posicionarnos en la celda i+1
    plt.subplot(5,5,i+1)

    #quita los numeros (representan los pixeles) de los ejes 
    plt.xticks([])
    plt.yticks([])

    plt.grid(False)

    #coloca cada imagen y las pone en gris
    plt.imshow(imagenes_train_normalizadas[i], cmap='grey')

    #pone el nombre buscando en las etiquetas del train el numero del 0-9
    plt.title(nombre_ropa[etiquetas_train[i]])

plt.show()