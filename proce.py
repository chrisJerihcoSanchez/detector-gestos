import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ruta donde están las imágenes y las etiquetas
dataset_path = r"C:/Users/ESTUDIANTE/Desktop/ejercicio samuel/gesto/dataset"  # Actualiza esto con la ruta correcta a tu dataset
gestures = ['A', 'OK', 'BIEN', 'HOLA', 'FUCK']  # Lista de los gestos

# Lista para almacenar las imágenes y las etiquetas
images = []
labels = []

# Leer las imágenes y las etiquetas
for label in gestures:
    gesture_folder = os.path.join(dataset_path, label)
    
    for filename in os.listdir(gesture_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Asegúrate de que las imágenes están en .png o .jpg
            image_path = os.path.join(gesture_folder, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar la imagen en escala de grises
            img_resized = cv2.resize(img, (64, 64))  # Redimensionar a 64x64
            images.append(img_resized)
            labels.append(gestures.index(label))  # Etiqueta numérica para el gesto

# Convertir las listas a arrays de numpy
X = np.array(images)
y = np.array(labels)

# Aplanar las imágenes para que tengan una sola dimensión
X = X.reshape(X.shape[0], -1)

# Normalizar las imágenes (opcional)
X = X / 255.0  # Normalización entre 0 y 1

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar cuántas imágenes hay para entrenamiento y prueba
print(f"Conjunto de entrenamiento: {X_train.shape[0]} imágenes")
print(f"Conjunto de prueba: {X_test.shape[0]} imágenes")

# Entrenar un modelo con Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
modelo_guardado = 'gesture_recognition_model.pkl'
joblib.dump(model, modelo_guardado)

print(f"Modelo guardado como: {modelo_guardado}")

# Ahora el modelo está guardado, puedes cargarlo en el futuro y usarlo para hacer predicciones
