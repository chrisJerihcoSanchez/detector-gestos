import cv2
import numpy as np
import joblib
import os

# Cargar el modelo entrenado desde la misma carpeta del script
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(directorio_actual, 'gesture_recognition_model.pkl')
model = joblib.load(ruta_modelo)

# Lista de los gestos (A, OK, BIEN, HOLA, FUCK)
gestures = ['A', 'OK', 'BIEN', 'HOLA', 'FUCK']

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Verifica si la cámara está abierta correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    # Captura el fotograma de la cámara
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma.")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar la imagen para que tenga el mismo tamaño que las imágenes del dataset
    img_resized = cv2.resize(gray, (64, 64))  # Redimensionar a 64x64
    img_flattened = img_resized.flatten()  # Aplanar la imagen

    # Normalizar la imagen (opcional, pero recomendable si el modelo fue entrenado con imágenes normalizadas)
    img_normalized = img_flattened / 255.0  # Escalar los valores de los píxeles entre 0 y 1

    # Realizar la predicción usando el modelo
    prediction = model.predict([img_normalized])  # Usamos la imagen normalizada

    # Mostrar el gesto detectado en la pantalla
    gesture = gestures[prediction[0]]  # Mapeamos la predicción al gesto correspondiente

    # Mostrar el texto con el gesto detectado
    cv2.putText(frame, f'Gesto detectado: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen en tiempo real
    cv2.imshow("Frame", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
