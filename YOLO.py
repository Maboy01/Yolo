from ultralytics import YOLO
import cv2
import time

# Modelo
model = YOLO("yolov8n.pt")

# Clases de interés
clases_interes = ["person", "laptop", "backpack", "suitcase", "chair"]

# Colores por clase (BGR)
colores = {
    "person": (0, 255, 0),
    "laptop": (255, 0, 0),
    "backpack": (0, 0, 255),
    "suitcase": (255, 255, 0),
    "chair": (255, 0, 255)
}

# Cámara celular
cap = cv2.VideoCapture("http://10.194.227.245:4747/video")

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    annotated_frame = frame.copy()

    # ===== MÉTRICAS =====
    conteo = {c: 0 for c in clases_interes}
    personas = []
    sillas = []
    mochilas = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # FILTRO DE CONFIANZA
            if conf < 0.5:
                continue

            if label in clases_interes:
                conteo[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Guardar para lógica avanzada
                if label == "person":
                    personas.append((x1, y1, x2, y2))
                elif label == "chair":
                    sillas.append((x1, y1, x2, y2))
                elif label in ["backpack", "suitcase"]:
                    mochilas.append((x1, y1, x2, y2))

                # Dibujar
                color = colores[label]

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

    # ===== PUESTOS OCUPADOS =====
    puestos_ocupados = 0

    for silla in sillas:
        sx1, sy1, sx2, sy2 = silla
        for persona in personas:
            px1, py1, px2, py2 = persona

            if abs(sx1 - px1) < 100 and abs(sy1 - py1) < 100:
                puestos_ocupados += 1
                break

    puestos_libres = max(len(sillas) - puestos_ocupados, 0)

    # ===== MALAS ABANDONADAS =====
    maletas_solas = 0

    for mochila in mochilas:
        mx1, my1, mx2, my2 = mochila
        cerca = False

        for persona in personas:
            px1, py1, px2, py2 = persona

            if abs(mx1 - px1) < 100 and abs(my1 - py1) < 100:
                cerca = True
                break

        if not cerca:
            maletas_solas += 1

    # ===== DENSIDAD =====
    densidad = conteo["person"] / (conteo["chair"] + 1)

    # ===== ÍNDICE OCUPACIÓN =====
    total_puestos = puestos_ocupados + puestos_libres
    indice_ocupacion = (puestos_ocupados / total_puestos) if total_puestos > 0 else 0

    # ===== MOSTRAR MÉTRICAS EN PANTALLA =====
    y = 20

    def escribir(texto):
        global y
        cv2.putText(annotated_frame, texto, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    escribir(f"Personas: {conteo['person']}")
    escribir(f"Laptops: {conteo['laptop']}")
    escribir(f"Sillas: {conteo['chair']}")
    escribir(f"Maletas: {conteo['backpack'] + conteo['suitcase']}")

    escribir(f"Puestos ocupados: {puestos_ocupados}")
    escribir(f"Puestos libres: {puestos_libres}")

    escribir(f"Maletas solas: {maletas_solas}")

    escribir(f"Densidad: {densidad:.2f}")
    escribir(f"Indice ocupacion: {indice_ocupacion:.2f}")

    # ALERTA VISUAL
    if maletas_solas > 0:
        cv2.putText(annotated_frame, "ALERTA: OBJETO SOLO",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("YOLO Inteligente", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()