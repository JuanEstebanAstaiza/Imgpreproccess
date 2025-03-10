import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 📂 Obtener la ruta del directorio de trabajo (workspace)
workspace = os.getcwd()  # 🏢 Obtiene la ruta donde se ejecuta el script

# 📂 Carpeta de salida dentro del mismo workspace
carpeta_salida = os.path.join(workspace, "imagenes_procesadas")

# 📂 Crear la carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# 🎨 Rango de colores en HSV para detectar señales de tránsito
rangos_colores = {
    "rojo1": ((0, 70, 50), (10, 255, 255)),
    "rojo2": ((170, 70, 50), (180, 255, 255)),
    "azul": ((100, 100, 50), (140, 255, 255)),
    "amarillo": ((15, 100, 100), (35, 255, 255))
}

# 🔄 Procesar todas las imágenes en el workspace
for archivo in os.listdir(workspace):
    if archivo.endswith((".jpg", ".png", ".jpeg")):  # 📌 Filtrar solo imágenes
        ruta_imagen = os.path.join(workspace, archivo)

        # 1️⃣ Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        imagen_copia = imagen_rgb.copy()  # Copia para dibujar

        # 2️⃣ Convertir a HSV y aplicar máscaras de color
        imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        mascara_total = np.zeros_like(imagen_hsv[:, :, 0])
        for rango in rangos_colores.values():
            mascara = cv2.inRange(imagen_hsv, rango[0], rango[1])
            mascara_total = cv2.bitwise_or(mascara_total, mascara)

        # 3️⃣ Mejorar contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris_mejorado = clahe.apply(gris)

        # 4️⃣ Filtro bilateral para suavizar sin perder bordes
        suavizado = cv2.bilateralFilter(gris_mejorado, 9, 75, 75)

        # 5️⃣ Detección de bordes adaptativa (Canny)
        sigma = 0.33
        mediana = np.median(suavizado)
        limite_inferior = int(max(0, (1.0 - sigma) * mediana))
        limite_superior = int(min(255, (1.0 + sigma) * mediana))
        bordes = cv2.Canny(suavizado, limite_inferior, limite_superior)

        # 6️⃣ Detección de círculos (Transformada de Hough)
        circulos = cv2.HoughCircles(
            suavizado, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=15, maxRadius=150
        )

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for i in circulos[0, :]:
                cv2.circle(imagen_copia, (i[0], i[1]), i[2], (0, 255, 0), 3)
                cv2.putText(imagen_copia, "Circulo", (i[0] - 20, i[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7️⃣ Detección de polígonos (triángulos, rectángulos)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacion = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

            if len(aproximacion) == 3:  # Triángulo
                cv2.drawContours(imagen_copia, [aproximacion], 0, (255, 0, 0), 3)
                cv2.putText(imagen_copia, "Triangulo", tuple(aproximacion[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            elif len(aproximacion) == 4:  # Rectángulo/Cuadrado
                cv2.drawContours(imagen_copia, [aproximacion], 0, (0, 255, 255), 3)
                cv2.putText(imagen_copia, "Rectangulo", tuple(aproximacion[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 8️⃣ Guardar imagen procesada en el mismo directorio (dentro de la carpeta de salida)
        ruta_salida = os.path.join(carpeta_salida, f"procesado_{archivo}")
        cv2.imwrite(ruta_salida, cv2.cvtColor(imagen_copia, cv2.COLOR_RGB2BGR))

        print(f"✅ Imagen procesada y guardada: {ruta_salida}")

        # 9️⃣ Mostrar resultados (opcional)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(mascara_total, cmap='gray')
        axs[0].set_title("Máscara de color")
        axs[0].axis("off")

        axs[1].imshow(bordes, cmap='gray')
        axs[1].set_title("Bordes detectados")
        axs[1].axis("off")

        axs[2].imshow(imagen_copia)
        axs[2].set_title("Señales detectadas")
        axs[2].axis("off")

        plt.show()
