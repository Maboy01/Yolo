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
# Incluye alerta automática cuando se detecta un objeto sin supervisión adecuada,
# detección heurística de "Computador de Mesa" y estado de sillas (sola/ocupada).
# ==========================================================


# ==========================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
import cv2
from ultralytics import YOLO
import time


# ==========================================================
# 2. VARIABLE GLOBAL DE CONTROL
# ==========================================================
salir = False


# ==========================================================
# 3. FUNCIÓN DE EVENTO DE MOUSE (BOTÓN SALIR)
# ==========================================================
def click_event(event, x, y, flags, param):
    global salir
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= x <= 630 and 10 <= y <= 50:
            print("Botón SALIR presionado")
            salir = True


# ==========================================================
# 4. HELPER: SOLAPAMIENTO DE BOUNDING BOXES
# Retorna True si dos bboxes se intersectan
# ==========================================================
def hay_solapamiento(bbox1, bbox2):
    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2
    return not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a)


# ==========================================================
# 5. CARGA DEL MODELO YOLOv8
# ==========================================================
model = YOLO("yolov8n.pt")


# ==========================================================
# 6. CONEXIÓN CON DROIDCAM
# ==========================================================
url = "http://10.28.135.113:4747/video"
cap = cv2.VideoCapture(url)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("No se pudo conectar a la cámara")
    exit()


# ==========================================================
# 7. CONFIGURACIÓN DE CLASES
# ==========================================================
clases_principales = ["person", "laptop", "backpack", "chair"]
clases_pc          = ["tv", "keyboard", "mouse"]
clases_objeto      = ["laptop", "backpack", "computador_mesa"]
# Las sillas se excluyen de la alerta general: tienen su propia lógica


# ==========================================================
# 8. INICIALIZACIÓN DE MÉTRICAS
# ==========================================================
UMBRAL_CONFIANZA = 0.55

clases_display = ["person", "laptop", "backpack", "chair", "computador_mesa"]
conteo_total   = {c: 0 for c in clases_display}
objetos_contados = set()

# Estado histórico de sillas (por ID único)
sillas_vistas_solas    = set()   # IDs de sillas que alguna vez estuvieron vacías
sillas_vistas_ocupadas = set()   # IDs de sillas que alguna vez estuvieron ocupadas

fps_list              = []
confianzas            = []
detecciones_por_frame = []

ultimo_alerta   = 0
COOLDOWN_ALERTA = 3

inicio_total = time.time()
print("Iniciando detección...")
print("Regla PC escritorio : tv + (keyboard o mouse) = Computador de Mesa")
print("Regla silla ocupada : bbox silla solapa con bbox persona")


# ==========================================================
# 9. BUCLE PRINCIPAL
# ==========================================================
while True:
    start_frame = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Reconectando...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(url)
        continue

    # ======================================================
    # 10. DETECCIÓN + TRACKING
    # ======================================================
    results = model.track(frame, persist=True, conf=UMBRAL_CONFIANZA)

    conteo_frame = {c: 0 for c in clases_display}

    detecciones_tv         = []   # (obj_id, bbox, conf)
    detecciones_periferico = []   # (obj_id, bbox, conf)
    detecciones_sillas     = []   # (obj_id, bbox, conf)
    detecciones_personas   = []   # (obj_id, bbox, conf)  — para solapamiento
    detecciones_otras      = []   # (label, obj_id, bbox, conf)

    # ======================================================
    # 11. RECOLECCIÓN DE DETECCIONES
    # ======================================================
    for r in results:
        for box in r.boxes:

            cls_id     = int(box.cls[0])
            label      = model.names[cls_id]
            confidence = float(box.conf[0])

            if box.id is None:
                continue

            obj_id = int(box.id[0])
            bbox   = tuple(map(int, box.xyxy[0]))

            if label == "chair":
                detecciones_sillas.append((obj_id, bbox, confidence))
                confianzas.append(confidence)

            elif label == "person":
                detecciones_personas.append((obj_id, bbox, confidence))
                detecciones_otras.append((label, obj_id, bbox, confidence))
                confianzas.append(confidence)

            elif label in clases_principales:   # laptop, backpack
                detecciones_otras.append((label, obj_id, bbox, confidence))
                confianzas.append(confidence)
            # "chair" y "person" ya fueron manejados arriba

            elif label in clases_pc:
                confianzas.append(confidence)
                if label == "tv":
                    detecciones_tv.append((obj_id, bbox, confidence))
                else:
                    detecciones_periferico.append((obj_id, bbox, confidence))

    # ======================================================
    # 12. ESTADO DE SILLAS (sola / ocupada)
    # ======================================================
    sillas_solas_frame    = 0
    sillas_ocupadas_frame = 0

    for silla_id, silla_bbox, silla_conf in detecciones_sillas:

        conteo_frame["chair"] += 1

        if ("chair", silla_id) not in objetos_contados:
            conteo_total["chair"] += 1
            objetos_contados.add(("chair", silla_id))

        ocupada = any(hay_solapamiento(silla_bbox, p_bbox)
                      for _, p_bbox, _ in detecciones_personas)

        x1, y1, x2, y2 = silla_bbox

        if ocupada:
            sillas_vistas_ocupadas.add(silla_id)
            sillas_ocupadas_frame += 1
            color      = (0, 165, 255)          # naranja
            etiqueta   = f"Silla Ocupada ID:{silla_id} {silla_conf:.2f}"
        else:
            sillas_vistas_solas.add(silla_id)
            sillas_solas_frame += 1
            color      = (0, 255, 255)           # amarillo
            etiqueta   = f"Silla Sola    ID:{silla_id} {silla_conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, etiqueta, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ======================================================
    # 13. HEURÍSTICA: COMPUTADOR DE MESA
    # ======================================================
    es_pc_escritorio = len(detecciones_tv) > 0 and len(detecciones_periferico) > 0

    if es_pc_escritorio:
        tv_id, tv_bbox, tv_conf = detecciones_tv[0]
        x1, y1, x2, y2 = tv_bbox

        conteo_frame["computador_mesa"] += 1

        clave_pc = f"pc_{tv_id}"
        if clave_pc not in objetos_contados:
            conteo_total["computador_mesa"] += 1
            objetos_contados.add(clave_pc)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, f"Computador de Mesa ID:{tv_id} {tv_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    else:
        for tv_id, tv_bbox, tv_conf in detecciones_tv:
            x1, y1, x2, y2 = tv_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 0), 2)
            cv2.putText(frame, f"Monitor? ID:{tv_id} {tv_conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 0), 2)

    # ======================================================
    # 14. DIBUJAR OTRAS CLASES (person, laptop, backpack)
    # ======================================================
    for label, obj_id, bbox, confidence in detecciones_otras:
        conteo_frame[label] += 1

        if (label, obj_id) not in objetos_contados:
            conteo_total[label] += 1
            objetos_contados.add((label, obj_id))

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID:{obj_id} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ======================================================
    # 15. MÉTRICAS POR FRAME
    # ======================================================
    detecciones_por_frame.append(sum(conteo_frame.values()))

    fps = 1 / (time.time() - start_frame)
    fps_list.append(fps)

    # ======================================================
    # 16. ALERTA "OBJETO SOLO"
    # ======================================================
    objetos_en_frame  = sum(conteo_frame[c] for c in clases_objeto if c in conteo_frame)
    personas_en_frame = conteo_frame["person"]

    alerta_activa = objetos_en_frame > 0 and personas_en_frame <= 1

    if alerta_activa:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 65), (w, h), (0, 0, 180), -1)
        msg_alerta = "ALERTA: OBJETO SIN SUPERVISION" if personas_en_frame == 0 \
                     else "ALERTA: OBJETO CON 1 PERSONA"
        cv2.putText(frame, msg_alerta, (10, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        ahora = time.time()
        if ahora - ultimo_alerta > COOLDOWN_ALERTA:
            print(f"[ALERTA] {msg_alerta} — personas en frame: {personas_en_frame}")
            ultimo_alerta = ahora

    # ======================================================
    # 17. HUD DE ESTADO
    # ======================================================
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    y_offset = 60
    for clase, count in conteo_frame.items():
        if clase in ("chair", "person"):
            continue   # se muestran por separado más abajo
        if count > 0:
            cv2.putText(frame, f"{clase}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

    # Desglose de sillas en el frame actual
    if conteo_frame["chair"] > 0:
        cv2.putText(frame, f"Sillas solas   : {sillas_solas_frame}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"Sillas ocupadas: {sillas_ocupadas_frame}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    cv2.putText(frame, f"Personas: {conteo_frame['person']}",
                (10, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    # ======================================================
    # 18. BOTÓN SALIR
    # ======================================================
    cv2.rectangle(frame, (500, 10), (630, 50), (0, 0, 255), -1)
    cv2.putText(frame, "SALIR", (520, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO PRO FINAL", frame)
    cv2.setMouseCallback("YOLO PRO FINAL", click_event)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or salir:
        break


# ==========================================================
# 19. FINALIZACIÓN Y RESULTADOS
# ==========================================================
cap.release()
cv2.destroyAllWindows()

tiempo_total = time.time() - inicio_total

print("\n" + "=" * 40)
print("       RESULTADOS FINALES")
print("=" * 40)

print("\nConteo de objetos unicos:")
for clase, total in conteo_total.items():
    if clase == "chair":
        continue
    print(f"  {clase:20s}: {total}")

print(f"\n  {'chair (total)':20s}: {conteo_total['chair']}")
print(f"  {'  -> solas alguna vez':20s}: {len(sillas_vistas_solas)}")
print(f"  {'  -> ocupadas alguna vez':20s}: {len(sillas_vistas_ocupadas)}")

promedio_confianza = sum(confianzas) / len(confianzas) if confianzas else 0
detecciones_seg    = sum(detecciones_por_frame) / tiempo_total if tiempo_total > 0 else 0

print(f"\nConfianza promedio : {promedio_confianza:.2f}")
print(f"Detecciones/seg    : {detecciones_seg:.2f}")
print(f"Tiempo total       : {tiempo_total:.1f} s")
print("=" * 40)
