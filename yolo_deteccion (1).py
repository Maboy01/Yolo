# ==========================================================
# PROYECTO: Detección de Objetos en Tiempo Real con YOLO
# ==========================================================
# Estudiantes: César Aguirre - Esteban Fonseca
# Asignatura: Deep Learning (Fundamentos)
# Fecha: Marzo 2026
#
# DESCRIPCIÓN:
# Este sistema implementa detección de objetos en tiempo real utilizando YOLOv8
# junto con técnicas de tracking para realizar un conteo preciso de objetos únicos.
# Además, se integran métricas de rendimiento y visualizaciones gráficas para analizar
# el comportamiento del modelo en un entorno dinámico.
# ==========================================================


# ==========================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
import cv2
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# cv2 -> Captura y procesamiento de video en tiempo real
# YOLO -> Modelo de detección de objetos
# time -> Medición de tiempos (FPS, rendimiento)
# matplotlib -> Visualización de resultados (gráficas)


# ==========================================================
# 2. VARIABLE GLOBAL DE CONTROL
# ==========================================================
salir = False
# Variable que permite finalizar el sistema mediante interacción del usuario


# ==========================================================
# 3. FUNCIÓN DE EVENTO DE MOUSE (BOTÓN SALIR)
# ==========================================================
def click_event(event, x, y, flags, param):
    global salir
    if event == cv2.EVENT_LBUTTONDOWN:
        # Se verifica si el clic ocurre dentro del botón "SALIR"
        if 500 <= x <= 630 and 10 <= y <= 50:
            print("Botón SALIR presionado")
            salir = True


# ==========================================================
# 4. CARGA DEL MODELO YOLOv8
# ==========================================================
model = YOLO("yolov8n.pt")

# Se utiliza la versión YOLOv8n (nano), optimizada para velocidad
# Ideal para aplicaciones en tiempo real con bajo consumo de recursos


# ==========================================================
# 5. CONEXIÓN CON DROIDCAM
# ==========================================================
url = "http://10.239.62.53:4747/video"
cap = cv2.VideoCapture(url)

# Se reduce el buffer para disminuir la latencia del video
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verificación de conexión
if not cap.isOpened():
    print("No se pudo conectar a la cámara")
    exit()


# ==========================================================
# 6. CONFIGURACIÓN DE CLASES DE INTERÉS
# ==========================================================
clases_interes = ["person", "laptop", "backpack", "chair"]

# Se filtran únicamente los objetos relevantes para el análisis


# ==========================================================
# 7. INICIALIZACIÓN DE MÉTRICAS
# ==========================================================
conteo_total = {clase: 0 for clase in clases_interes}
# Conteo real de objetos únicos detectados

fps_list = []
# Lista para almacenar FPS a lo largo del tiempo

confianzas = []
# Lista de todas las confianzas detectadas

detecciones_por_frame = []
# Número de objetos detectados en cada frame

conf_por_clase = {clase: [] for clase in clases_interes}
# Confianza separada por cada tipo de objeto

objetos_contados = set()
# Conjunto que almacena IDs únicos para evitar duplicados

inicio_total = time.time()

print("Iniciando detección...")


# ==========================================================
# 8. BUCLE PRINCIPAL (PROCESAMIENTO EN TIEMPO REAL)
# ==========================================================
while True:
    start_frame = time.time()

    ret, frame = cap.read()

    # ======================================================
    # 9. MANEJO DE ERRORES (RECONEXIÓN)
    # ======================================================
    if not ret:
        print("Reconectando...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(url)
        continue

    # ======================================================
    # 10. DETECCIÓN + TRACKING
    # ======================================================
    results = model.track(frame, persist=True)

    # persist=True permite mantener el ID de los objetos entre frames

    conteo_frame = {clase: 0 for clase in clases_interes}

    # ======================================================
    # 11. PROCESAMIENTO DE RESULTADOS
    # ======================================================
    for r in results:
        for box in r.boxes:

            # Clase detectada
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Nivel de confianza
            confidence = float(box.conf[0])

            # Validación de ID (tracking)
            if box.id is None:
                continue

            obj_id = int(box.id[0])

            # ==================================================
            # 12. FILTRADO POR CLASE
            # ==================================================
            if label in clases_interes:

                conteo_frame[label] += 1

                # ==================================================
                # 13. CONTEO REAL (EVITA DUPLICADOS)
                # ==================================================
                if obj_id not in objetos_contados:
                    conteo_total[label] += 1
                    objetos_contados.add(obj_id)

                # Registro de métricas
                confianzas.append(confidence)
                conf_por_clase[label].append(confidence)

                # Coordenadas del objeto
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Mostrar etiqueta + ID + confianza
                texto = f"{label} ID:{obj_id} {confidence:.2f}"
                cv2.putText(frame, texto, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ======================================================
    # 14. MÉTRICAS POR FRAME
    # ======================================================
    detecciones_por_frame.append(sum(conteo_frame.values()))

    fps = 1 / (time.time() - start_frame)
    fps_list.append(fps)

    # Mostrar FPS en pantalla
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Mostrar conteo por clase
    y_offset = 60
    for clase, count in conteo_frame.items():
        cv2.putText(frame, f"{clase}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y_offset += 25

    # ======================================================
    # 15. BOTÓN INTERACTIVO
    # ======================================================
    cv2.rectangle(frame, (500, 10), (630, 50), (0,0,255), -1)
    cv2.putText(frame, "SALIR", (520, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("YOLO PRO FINAL", frame)
    cv2.setMouseCallback("YOLO PRO FINAL", click_event)

    # ======================================================
    # 16. CONDICIÓN DE SALIDA
    # ======================================================
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or salir:
        break


# ==========================================================
# 17. FINALIZACIÓN
# ==========================================================
cap.release()
cv2.destroyAllWindows()

tiempo_total = time.time() - inicio_total

print("\nRESULTADOS FINALES:")
for clase, total in conteo_total.items():
    print(f"{clase}: {total}")


# ==========================================================
# 18. CÁLCULO DE MÉTRICAS FINALES
# ==========================================================
promedio_confianza = sum(confianzas) / len(confianzas) if confianzas else 0
detecciones_seg = sum(detecciones_por_frame) / tiempo_total

print(f"\nConfianza promedio: {promedio_confianza:.2f}")
print(f"Detecciones por segundo: {detecciones_seg:.2f}")


# ==========================================================
# 19. GRÁFICAS Y SU INTERPRETACIÓN
# ==========================================================

# ----------------------------------------------------------
# 1. Detecciones Totales
# Muestra cuántos objetos únicos fueron detectados por clase
# ----------------------------------------------------------
plt.figure()
plt.bar(conteo_total.keys(), conteo_total.values())
plt.title("Detecciones Totales (Reales)")
plt.show()


# ----------------------------------------------------------
# 2. FPS en el tiempo
# Permite analizar la estabilidad del sistema
# ----------------------------------------------------------
plt.figure()
plt.plot(fps_list)
plt.title("FPS en el tiempo")
plt.grid()
plt.show()


# ----------------------------------------------------------
# 3. Histograma de Confianza
# Muestra qué tan seguro es el modelo en sus predicciones
# ----------------------------------------------------------
plt.figure()
plt.hist(confianzas, bins=20)
plt.title("Distribución de Confianza")
plt.show()


# ----------------------------------------------------------
# 4. Detecciones por Frame
# Muestra la cantidad de objetos detectados en cada instante
# ----------------------------------------------------------
plt.figure()
plt.plot(detecciones_por_frame)
plt.title("Detecciones por Frame")
plt.show()


# ----------------------------------------------------------
# 5. Confianza Promedio por Clase
# Permite evaluar qué objetos son detectados con mayor precisión
# ----------------------------------------------------------
promedios = {c: (sum(v)/len(v) if v else 0) for c, v in conf_por_clase.items()}

plt.figure()
plt.bar(promedios.keys(), promedios.values())
plt.title("Confianza Promedio por Clase")
plt.show()


# ----------------------------------------------------------
# 6. Detecciones Acumuladas
# Muestra el crecimiento total de detecciones a lo largo del tiempo
# ----------------------------------------------------------
acumulado = []
total = 0
for d in detecciones_por_frame:
    total += d
    acumulado.append(total)

plt.figure()
plt.plot(acumulado)
plt.title("Detecciones Acumuladas")
plt.show()