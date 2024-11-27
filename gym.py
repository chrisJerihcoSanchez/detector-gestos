import cv2
import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Definir las carpetas para cada gesto
gestures = ['A', 'OK', 'BIEN', 'HOLA', 'FUCK']
dataset_path = 'dataset'  # Ruta donde se almacenarán las imágenes

# Verificar si las carpetas existen, si no, crearlas
for gesture in gestures:
    gesture_folder = os.path.join(dataset_path, gesture)
    if not os.path.exists(gesture_folder):
        os.makedirs(gesture_folder)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Verifica si la cámara está abierta correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Contador de imágenes capturadas
image_count = 0
current_gesture = None

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

    # Mostrar la imagen en tiempo real
    cv2.imshow("Frame", frame)

    # Instrucciones y lista de gestos disponibles
    cv2.putText(frame, "Selecciona un gesto:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for i, gesture in enumerate(gestures, 1):
        cv2.putText(frame, f"{i}. {gesture}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar el gesto seleccionado en la pantalla
    if current_gesture:
        cv2.putText(frame, f"Gesto seleccionado: {current_gesture}", 
                    (10, 60 + len(gestures) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tomar imagen cuando se presiona una tecla
    key = cv2.waitKey(1) & 0xFF

    # Si se presiona 1-5, seleccionamos el gesto correspondiente
    if key == ord('1'):
        current_gesture = 'A'
    elif key == ord('2'):
        current_gesture = 'OK'
    elif key == ord('3'):
        current_gesture = 'BIEN'
    elif key == ord('4'):
        current_gesture = 'HOLA'
    elif key == ord('5'):
        current_gesture = 'FUCK'

    # Si se presiona la tecla 'c', se captura la imagen
    if key == ord('c') and current_gesture:
        # Guardar la imagen en la carpeta correspondiente
        gesture_folder = os.path.join(dataset_path, current_gesture)
        image_count += 1
        img_path = os.path.join(gesture_folder, f'{current_gesture}_{image_count}.png')
        cv2.imwrite(img_path, img_resized)  # Guardar la imagen

        print(f"Imagen guardada como {img_path}")
    
    # Si se presiona 't', entrenamos el modelo
    if key == ord('t'):
        # Cargar todas las imágenes y etiquetas
        X = []
        y = []
        for gesture in gestures:
            gesture_folder = os.path.join(dataset_path, gesture)
            for filename in os.listdir(gesture_folder):
                if filename.endswith('.png'):
                    img = cv2.imread(os.path.join(gesture_folder, filename), cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (64, 64))  # Asegurarse de que todas tengan el mismo tamaño
                    X.append(img_resized.flatten())
                    y.append(gestures.index(gesture))

        X = np.array(X)
        y = np.array(y)

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar un modelo
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión del modelo: {accuracy * 100:.2f}%')

        # Guardar el modelo
        joblib.dump(model, 'gesture_recognition_model.pkl')
        print("Modelo guardado como 'gesture_recognition_model.pkl'")

    # Salir si se presiona la tecla 'q'
    if key == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()