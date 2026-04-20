# ============================================================
# SISTEMA DE VIGILANCIA INTELIGENTE CON CÁMARA
# Este programa usa inteligencia artificial para detectar
# personas, sillas, laptops y maletas en tiempo real
# a través de la cámara de un celular.
# ============================================================

# --- IMPORTAR HERRAMIENTAS ---
# Estas líneas "llaman" a los programas que necesitamos usar
from ultralytics import YOLO  # La inteligencia artificial que reconoce objetos
import cv2                    # Herramienta para manejar video e imágenes
import time                   # Herramienta para medir el tiempo (no se usa aquí todavía)

# --- CARGAR EL CEREBRO DE LA IA ---
# "yolov8n.pt" es un archivo que contiene una IA ya entrenada
# para reconocer más de 80 tipos de objetos distintos
model = YOLO("yolov8n.pt")

# --- DEFINIR QUÉ OBJETOS NOS INTERESAN ---
# De todos los objetos que la IA puede detectar,
# solo queremos prestar atención a estos 5:
clases_interes = ["person", "laptop", "backpack", "suitcase", "chair"]
#                  persona    laptop    mochila     maleta      silla

# --- ASIGNAR UN COLOR A CADA TIPO DE OBJETO ---
# Cada objeto detectado se dibujará con un rectángulo de diferente color
# Los colores están en formato BGR (Azul, Verde, Rojo) — al revés del normal
colores = {
    "person":   (0, 255, 0),    # Verde  → personas
    "laptop":   (255, 0, 0),    # Azul   → laptops
    "backpack": (0, 0, 255),    # Rojo   → mochilas
    "suitcase": (255, 255, 0),  # Cyan   → maletas
    "chair":    (255, 0, 255)   # Magenta→ sillas
}

# --- CONECTARSE A LA CÁMARA DEL CELULAR ---
# La dirección "http://10.28.135.184:4747/video" es la IP del celular
# usando la app DroidCam o similar para transmitir video por WiFi
cap = cv2.VideoCapture("http://10.28.135.184:4747/video")

# Verificar que la conexión con la cámara funcionó
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()  # Si no hay cámara, el programa se detiene aquí

# ============================================================
# BUCLE PRINCIPAL — El programa repite esto una y otra vez
# hasta que el usuario presione la tecla "q" para salir
# ============================================================
while True:

    # --- CAPTURAR UN FOTOGRAMA ---
    # "ret" indica si la captura fue exitosa (True/False)
    # "frame" es la imagen del momento actual de la cámara
    ret, frame = cap.read()

    # Si la cámara dejó de enviar imagen, salir del bucle
    if not ret:
        break

    # --- AJUSTAR EL TAMAÑO DE LA IMAGEN ---
    # Redimensionar a 640x480 píxeles para que la IA trabaje más rápido
    frame = cv2.resize(frame, (640, 480))

    # --- ANALIZAR LA IMAGEN CON LA IA ---
    # La IA examina la imagen y devuelve todos los objetos que encontró
    results = model(frame)

    # Hacer una copia de la imagen donde dibujaremos los rectángulos
    annotated_frame = frame.copy()

    # ============================================================
    # SECCIÓN DE MÉTRICAS — Contadores y listas para el análisis
    # ============================================================

    # Contador de cuántos objetos de cada tipo hay en la imagen
    conteo = {c: 0 for c in clases_interes}
    # Resultado: {"person": 0, "laptop": 0, "backpack": 0, ...}

    # Listas para guardar las posiciones (coordenadas) de cada objeto
    personas = []   # Posición de cada persona detectada
    sillas   = []   # Posición de cada silla detectada
    mochilas = []   # Posición de cada mochila o maleta detectada

    # ============================================================
    # PROCESAR CADA OBJETO DETECTADO POR LA IA
    # ============================================================
    for r in results:               # Para cada resultado de la IA...
        for box in r.boxes:         # Para cada objeto encontrado...

            cls   = int(box.cls[0])       # Número que identifica el tipo de objeto
            label = model.names[cls]      # Convertir ese número al nombre ("person", "chair"...)
            conf  = float(box.conf[0])    # Nivel de confianza: qué tan segura está la IA (0 a 1)

            # --- FILTRO DE CALIDAD ---
            # Si la IA no está al menos 50% segura, ignorar este objeto
            # Esto evita falsos positivos (creer ver algo que no está)
            if conf < 0.5:
                continue  # Saltar al siguiente objeto

            # Solo procesar si el objeto es uno de los que nos interesan
            if label in clases_interes:

                # Sumar 1 al contador de ese tipo de objeto
                conteo[label] += 1

                # Obtener las coordenadas del rectángulo que rodea al objeto
                # (x1,y1) = esquina superior izquierda
                # (x2,y2) = esquina inferior derecha
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Guardar la posición según el tipo de objeto
                # (lo usaremos más adelante para el análisis de proximidad)
                if label == "person":
                    personas.append((x1, y1, x2, y2))
                elif label == "chair":
                    sillas.append((x1, y1, x2, y2))
                elif label in ["backpack", "suitcase"]:
                    mochilas.append((x1, y1, x2, y2))

                # --- DIBUJAR EN LA IMAGEN ---
                color = colores[label]

                # Dibujar el rectángulo alrededor del objeto
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Escribir el nombre del objeto y el % de confianza encima del rectángulo
                cv2.putText(
                    annotated_frame,
                    f"{label} {conf:.2f}",   # Ej: "person 0.87"
                    (x1, y1 - 10),           # Posición: justo arriba del rectángulo
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,                     # Tamaño del texto
                    color,
                    2                        # Grosor del texto
                )

    # ============================================================
    # ANÁLISIS: ¿CUÁNTAS SILLAS ESTÁN OCUPADAS?
    # Lógica: si una persona y una silla están muy cerca en la imagen,
    # asumimos que esa persona está sentada en esa silla
    # ============================================================
    puestos_ocupados = 0

    for silla in sillas:
        sx1, sy1, sx2, sy2 = silla      # Coordenadas de la silla

        for persona in personas:
            px1, py1, px2, py2 = persona  # Coordenadas de la persona

            # Si la diferencia de posición es menor a 100 píxeles,
            # consideramos que están en el mismo lugar → puesto ocupado
            if abs(sx1 - px1) < 100 and abs(sy1 - py1) < 100:
                puestos_ocupados += 1
                break  # Una persona ya ocupa esta silla, no seguir buscando

    # Las sillas libres son el total de sillas menos las que tienen persona
    puestos_libres = max(len(sillas) - puestos_ocupados, 0)

    # ============================================================
    # ANÁLISIS: ¿HAY MALETAS ABANDONADAS?
    # Lógica: si una mochila/maleta NO tiene a ninguna persona cerca,
    # puede estar abandonada — lo cual es una alerta de seguridad
    # ============================================================
    maletas_solas = 0

    for mochila in mochilas:
        mx1, my1, mx2, my2 = mochila    # Coordenadas de la mochila
        cerca = False                    # Suponemos que no hay nadie cerca

        for persona in personas:
            px1, py1, px2, py2 = persona

            # Si hay una persona a menos de 100 píxeles de distancia,
            # la mochila NO está abandonada
            if abs(mx1 - px1) < 100 and abs(my1 - py1) < 100:
                cerca = True
                break  # Ya encontramos al dueño, no seguir buscando

        # Si ninguna persona estuvo cerca, la maleta está sola
        if not cerca:
            maletas_solas += 1

    # ============================================================
    # MOSTRAR ESTADÍSTICAS EN PANTALLA
    # Se escriben en la esquina superior izquierda del video
    # ============================================================
    y = 20  # Posición vertical inicial del primer texto

    # Función auxiliar para escribir una línea de texto y bajar la posición
    def escribir(texto):
        global y
        cv2.putText(annotated_frame, texto, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25  # Bajar 25 píxeles para la siguiente línea

    # Mostrar conteos básicos
    escribir(f"Personas: {conteo['person']}")
    escribir(f"Laptops: {conteo['laptop']}")
    escribir(f"Sillas: {conteo['chair']}")
    escribir(f"Maletas: {conteo['backpack'] + conteo['suitcase']}")

    # Mostrar análisis de puestos
    escribir(f"Puestos ocupados: {puestos_ocupados}")
    escribir(f"Puestos libres: {puestos_libres}")

    # Mostrar alerta de maletas
    escribir(f"Maletas solas: {maletas_solas}")

    # --- ALERTA VISUAL ROJA ---
    # Si hay al menos una maleta sin dueño cerca, mostrar advertencia grande
    if maletas_solas > 0:
        cv2.putText(annotated_frame, "ALERTA: OBJETO SOLO",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # --- MOSTRAR LA IMAGEN EN UNA VENTANA ---
    # Abre (o actualiza) una ventana llamada "YOLO Inteligente" con el video anotado
    cv2.imshow("YOLO Inteligente", annotated_frame)

    # --- CONTROL DE SALIDA ---
    # Esperar 1 milisegundo. Si el usuario presiona "q", cerrar el programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# CIERRE DEL PROGRAMA
# Liberar la cámara y cerrar todas las ventanas correctamente
# ============================================================
cap.release()           # Desconectarse de la cámara
cv2.destroyAllWindows() # Cerrar la ventana de video