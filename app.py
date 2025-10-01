import os
import cv2
import time
import torch
import numpy as np
import threading
from flask import Flask, render_template, Response, abort, jsonify
from ultralytics import YOLO

# (Opsional) diamkan log TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# === Jika Anda butuh TF untuk classifier huruf/angka ===
from tensorflow.keras.models import load_model

# === Initialize Flask App ===
app = Flask(__name__)

# === Cek Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device yang digunakan:", device)
if device == 'cuda':
    try:
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Memori GPU tersedia:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
    except Exception:
        pass

# === Path Model dan Direktori ===
MODEL_DETECT_PATH = r"C:\Users\EagleEyes\belajar\model\best.pt"
MODEL_PLAT_DETECT_PATH = r"plat_nomor.pt"
MODEL_HURUF_PATH = r"huruf_classifier.h5"
MODEL_ANGKA_PATH = r"angka_classifier.h5"

VIDEO_FOLDER = r"C:\Users\EagleEyes\belajar\video"
video_files = ["jpo_pondok_bambu.mp4", "jpo_lenteng_agung.mp4", "jpo_tanjung_barat.mov", "jpo_jorr_pesangrahan.mp4"]
video_location = ["JPO PONDOK BAMBU","JPO LENTENG AGUNG","JPO TANJUNG BARAT","JPO JORR PESANGRAHAN"]

# === Kelas Pelanggaran dan Label ===
violation_motor_labels = {"motor_1_nohelmet", "motor_2_nohelmet", "motor_more_2"}
violation_car_labels = {"driver_unbuckled", "passanger_unbuckled"}
unknown_car_labels = {"driver_unknown", "passanger_unknown"}

class_names = [
    "car", "driver_buckled", "driver_unbuckled", "driver_unknown", "kaca",
    "motor_1_helmet", "motor_1_nohelmet", "motor_2_helmet", "motor_2_nohelmet",
    "motor_more_2", "passanger_buckled", "passanger_unbuckled",
    "passanger_unknown", "plat_nomor"
]

# === Load Models (sekali saja) ===
detect_model = YOLO(MODEL_DETECT_PATH).to(device)

model_contains = None
model_plat_huruf = None
model_plat_angka = None
try:
    if os.path.exists(MODEL_PLAT_DETECT_PATH):
        model_contains = YOLO(MODEL_PLAT_DETECT_PATH).to(device)
    if os.path.exists(MODEL_HURUF_PATH):
        model_plat_huruf = load_model(MODEL_HURUF_PATH)
    if os.path.exists(MODEL_ANGKA_PATH):
        model_plat_angka = load_model(MODEL_ANGKA_PATH)
except Exception as e:
    print("Peringatan: model pembaca plat belum siap ->", e)

plat_angka = list("0123456789")
plat_huruf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["~"]

# ====== STORAGE utk Ringkasan (dibaca detail.html) ======
latest_detection_data = {}
data_lock = threading.Lock()

# === Util: cek child di dalam parent bbox ===
def is_inside(parent_box, child_box):
    px1, py1, px2, py2 = parent_box
    cx1, cy1, cx2, cy2 = child_box
    return (cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2)

# === Util: OCR sederhana plat ===
def check_platenumber(plate_crop):
    if model_contains is None or model_plat_huruf is None or model_plat_angka is None:
        return "N/A"

    plate_crop = cv2.resize(plate_crop, (400, 200), interpolation=cv2.INTER_CUBIC)
    result_contains = model_contains(plate_crop, verbose=False)[0]
    detected = []
    for box_contain in result_contains.boxes:
        x1, y1, x2, y2 = map(int, box_contain.xyxy[0].tolist())
        if x2 > x1 and y2 > y1:
            detected.append((x1, y1, x2, y2, plate_crop[y1:y2, x1:x2]))

    detected.sort(key=lambda b: b[0])
    if not detected:
        return "N/A"

    text = ""
    for (_, _, _, _, img) in detected:
        ci = cv2.resize(img, (64, 64)).astype("float32") / 255.0
        ci = np.expand_dims(ci, axis=0)
        p_num = model_plat_angka.predict(ci, verbose=0)[0]
        p_chr = model_plat_huruf.predict(ci, verbose=0)[0]
        if float(np.max(p_num)) >= float(np.max(p_chr)):
            text += plat_angka[int(np.argmax(p_num))]
        else:
            text += plat_huruf[int(np.argmax(p_chr))]
    return text

# === Stream sederhana (MJPEG) utk halaman utama ===
def gen_frames(video_file):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        abort(404, description=f"Video tidak ditemukan: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        abort(500, description=f"Gagal membuka video: {video_path}")

    no_grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with no_grad_ctx:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 70 == 0:
                results = detect_model.predict(frame, conf=0.5, verbose=False)
                annotated = frame.copy()
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0]); conf = float(box.conf[0])
                        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        is_violation = (label in violation_motor_labels) or (label in violation_car_labels)
                        color = (0, 0, 255) if is_violation else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                annotated = cv2.resize(annotated, (800, 600))
                ok, buf = cv2.imencode('.jpg', annotated)
                if ok:
                    frame_bytes = buf.tobytes()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           frame_bytes + b"\r\n")
            frame_num += 1
    cap.release()

# === Stream DETAIL: analitik + simpan ringkasan utk tabel ===
def gen_detail_frames(idx):
    video_file = video_files[idx]
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        abort(404, description=f"Video tidak ditemukan: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        abort(500, description=f"Gagal membuka video: {video_path}")

    no_grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with no_grad_ctx:
        frame_num = 0
        last_annotated_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_num += 1
            
            # Lakukan deteksi hanya setiap 20 frame untuk efisiensi
            if frame_num % 20 == 0:
                yolo_out = detect_model(frame, verbose=False)[0]
                detections = []
                for box in yolo_out.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = (0, 255, 0) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2),  color , 2)
                    detections.append({"label": label, "box": (x1, y1, x2, y2)})

                rows = []
                annotated_frame = frame.copy() # Salin frame untuk digambari
                
                # Analisis dan gambar per objek utama (mobil/motor)
                for obj in [d for d in detections if d["label"] == "car" or d["label"].startswith("motor")]:
                    parent_box = obj["box"]
                    children = [d for d in detections if d is not obj and is_inside(parent_box, d["box"])]
                    
                    vehicle_type = "Mobil" if obj["label"] == "car" else "Motor"
                    plate_text = "N/A"
                    conditions = []
                    
                    # Kumpulkan semua label yang relevan untuk ditampilkan
                    labels_to_show = []
                    
                    if vehicle_type == "Motor":
                        labels_to_show.append(obj["label"])
                        if obj["label"] in violation_motor_labels:
                            conditions.append(obj["label"])
                    else: # Mobil
                        labels_to_show.append(obj["label"])
                        
                    for ch in children:
                        if ch["label"] == "plat_nomor":
                            x1, y1, x2, y2 = ch["box"]
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                plate_text = check_platenumber(crop)
                        else:
                            labels_to_show.append(ch["label"])
                            if vehicle_type == "Mobil" and (ch["label"] in violation_car_labels or ch["label"] in unknown_car_labels):
                                conditions.append(ch["label"])

                    status = "-"
                    if any(c in violation_car_labels or c in violation_motor_labels for c in conditions):
                        status = "Violation"
                    elif any(c in unknown_car_labels for c in conditions):
                        status = "Unknown"
                    
                    # Gabungkan label untuk display di frame
                    display_label = " - ".join(labels_to_show)

                    # Tentukan warna kotak dan teks
                    color = (0, 0, 255) if status == "Violation" else ((0, 165, 255) if status == "Unknown" else (0, 255, 0))

                    # Gambar kotak dan label di frame
                    cv2.rectangle(annotated_frame, (parent_box[0], parent_box[1]), (parent_box[2], parent_box[3]), color, 3)
                    cv2.putText(
                        annotated_frame,
                        f"{display_label}",
                        (parent_box[0], max(0, parent_box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    # Siapkan data untuk tabel
                    rows.append({
                        "vehicle": vehicle_type,
                        "plate": plate_text,
                        "condition": ", ".join(conditions) if conditions else "-",
                        "note": status
                    })
                
                # Simpan data deteksi terbaru untuk endpoint JSON
                with data_lock:
                    latest_detection_data[idx] = rows
                
                last_annotated_frame = annotated_frame
            
            # Kirim frame beranotasi terakhir (jika sudah ada)
            frame_to_send = last_annotated_frame if last_annotated_frame is not None else frame
            frame_to_send = cv2.resize(frame_to_send, (1024, 600))
            
            ok, buf = cv2.imencode(".jpg", frame_to_send)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()

# ===================== ROUTES =====================

@app.route("/")
def index():
    items = list(enumerate(video_files))
    return render_template("index.html", items=items, video_location=video_location)

@app.route("/video_feed/<int:idx>")
def video_feed(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return Response(gen_frames(video_files[idx]),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers=headers)

@app.route("/detail/<int:idx>")
def detail(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    title = video_location[idx] if idx < len(video_location) else video_files[idx]
    return render_template("detail.html", idx=idx, title=title)

@app.route("/detail_feed/<int:idx>")
def detail_feed(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    return Response(gen_detail_frames(idx),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers=headers)

# === Endpoint JSON utk tabel Ringkasan Deteksi ===
@app.route("/detection_data/<int:idx>")
def detection_data(idx):
    with data_lock:
        data = latest_detection_data.get(idx, [])
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)