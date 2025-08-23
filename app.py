from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, Response, jsonify
)
import cv2
import numpy as np
from flask_mysqldb import MySQL
from datetime import datetime
from functools import wraps
import collections
from collections import OrderedDict
import os
import time
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.spatial import distance as dist

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # wajib untuk session

# ================= Konfigurasi MySQL =================
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'cctv_monitoring'
mysql = MySQL(app)

# ================= YOLO Setup =================
model = YOLO('saved_model/best.pt')
class_names = [
    'car', 'driver_buckled', 'driver_unbuckled', 'driver_unknown', 'kaca',
    'motor_1_helmet', 'motor_1_nohelmet', 'motor_2_helmet', 'motor_2_nohelmet',
    'motor_more_2', 'passanger_buckled', 'passanger_unbuckled', 'passanger_unknown',
    'plat_nomor'
]

video_paths = {
    1: "assets/cam1.mp4",
    2: "assets/cam2.mp4",
    3: "assets/cctv3.mp4",
    4: "assets/cctv4.mp4"
}

# Menyimpan data deteksi secara smoothing
deque_object_counts = {i: collections.deque(maxlen=30) for i in range(1, 5)}
object_counts = {i: {} for i in range(1, 5)}

# ================= Fungsi Database =================
def save_to_db(cam_id, plat_number, vehicle_type, violation, vehicle_speed, image_path):
    """
    Menyimpan data pelanggaran ke database.
    Fungsi ini harus dipanggil di dalam app_context Flask.
    """
    try:
        timestamp = datetime.now()
        cur = mysql.connection.cursor()
        sql = """INSERT INTO pelanggaran_kendaraan
                 (cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path)
                 VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cur.execute(sql, (cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path))
        mysql.connection.commit()
        cur.close()
        print(f"✅ Pelanggaran tersimpan: {violation} di cam {cam_id} dengan kecepatan {vehicle_speed:.2f} km/jam")
    except Exception as e:
        print(f"❌ Error menyimpan ke database: {e}")

# ================= Fungsi Helper =================
def detect_objects(frame):
    """Melakukan deteksi objek pada frame menggunakan model YOLO."""
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf > 0.5:
                label = class_names[cls] if cls < len(class_names) else str(cls)
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({"label": label, "conf": conf, "bbox": xyxy})
    return detections

def _draw_boxes(frame, detections, tracked_objects):
    """Menggambar bounding box dan info kecepatan pada frame."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{det['label']} {det['conf']:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Menampilkan ID dan kecepatan objek yang dilacak
    for objectID, data in tracked_objects.items():
        centroid = data["centroid"]
        speed_kmh = data.get("speed_kmh", 0)
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # if speed_kmh > 0:
        #     cv2.putText(frame, f"{speed_kmh:.2f} km/jam", (centroid[0] + 15, centroid[1]),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if speed_kmh > 0:
            pass  # simpan ke database saja, tidak ditampilkan di layar


def _update_deque_and_average(feed_number, counts_dict):
    """Memperbarui deque dan menghitung rata-rata untuk smoothing."""
    dq = deque_object_counts.get(feed_number)
    dq.append(counts_dict)
    avg_counts = {}
    for label in class_names:
        values = [c.get(label, 0) for c in dq]
        avg_counts[label] = int(np.mean(values)) if values else 0
    return avg_counts

def _update_object_count(feed_number, smooth_counts):
    """Memperbarui jumlah objek global."""
    object_counts[feed_number] = smooth_counts

def _generate_mjpeg_frame(frame):
    """Mengubah frame OpenCV menjadi format MJPEG."""
    _, buffer = cv2.imencode('.jpg', frame)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= Streaming + Save =================
def generate_stream(video_path, feed_number, detail=False):
    """Generator untuk streaming video, deteksi objek, dan penyimpanan ke DB."""
    width, height = (800, 500) if detail else (640, 380)
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        print(f"❌ Tidak bisa membuka video {video_path}")
        return
    fps = stream.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    display_counts = {}
    snapshot_folder = f"static/snapshots/cam{feed_number}"
    os.makedirs(snapshot_folder, exist_ok=True)

    # --- Variabel untuk Pelacakan dan Kecepatan ---
    tracked_objects = OrderedDict()
    next_object_id = 0
    # CATATAN: Nilai ini SANGAT PENTING dan perlu dikalibrasi.
    # Ukur objek di dunia nyata (misal lebar mobil = 1.8m) dan ukur lebarnya dalam piksel di video.
    # PIXELS_PER_METER = (lebar mobil dalam piksel) / (lebar mobil dalam meter)
    PIXELS_PER_METER = 20.0  # Contoh nilai, HARUS DIKALIBRASI

    while True:
        ret, frame = stream.read()
        if not ret:
            stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        frame_resized = cv2.resize(frame, (width, height))
        current_time = time.time()
        
        detections = []
        # Deteksi objek setiap 5 frame untuk pelacakan yang lebih baik
        if frame_count % 5 == 0:
            roi_start = int(height * 0.4)
            roi = frame_resized[roi_start:, :]
            frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            detections = detect_objects(frame_rgb)
            
            for det in detections:
                det["bbox"][1] += roi_start
                det["bbox"][3] += roi_start

            # --- Logika Pelacakan Objek (Centroid Tracker) ---
            input_centroids = np.zeros((len(detections), 2), dtype="int")
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det["bbox"])
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                input_centroids[i] = (cX, cY)

            if len(tracked_objects) == 0:
                for i in range(len(input_centroids)):
                    tracked_objects[next_object_id] = {"centroid": input_centroids[i], "timestamp": current_time, "bbox": detections[i]["bbox"]}
                    next_object_id += 1
            else:
                object_ids = list(tracked_objects.keys())
                previous_centroids = np.array([tracked_objects[oid]["centroid"] for oid in object_ids])
                
                D = dist.cdist(previous_centroids, input_centroids)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows, used_cols = set(), set()
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    
                    object_id = object_ids[row]
                    
                    # --- Perhitungan Kecepatan ---
                    time_diff = current_time - tracked_objects[object_id]["timestamp"]
                    dist_pixels = dist.euclidean(tracked_objects[object_id]["centroid"], input_centroids[col])
                    
                    if time_diff > 0:
                        dist_meters = dist_pixels / PIXELS_PER_METER
                        speed_mps = dist_meters / time_diff
                        speed_kmh = speed_mps * 3.6
                        tracked_objects[object_id]["speed_kmh"] = speed_kmh
                    
                    tracked_objects[object_id]["centroid"] = input_centroids[col]
                    tracked_objects[object_id]["timestamp"] = current_time
                    tracked_objects[object_id]["bbox"] = detections[col]["bbox"]
                    
                    used_rows.add(row)
                    used_cols.add(col)

                unused_rows = set(range(D.shape[0])).difference(used_rows)
                unused_cols = set(range(D.shape[1])).difference(used_cols)

                # Hapus objek yang hilang
                for row in unused_rows:
                    object_id = object_ids[row]
                    del tracked_objects[object_id]

                # Tambah objek baru
                for col in unused_cols:
                    tracked_objects[next_object_id] = {"centroid": input_centroids[col], "timestamp": current_time, "bbox": detections[col]["bbox"]}
                    next_object_id += 1
            
            # --- Logika Penyimpanan Pelanggaran ---
            counts = {}
            pelanggaran_terdeteksi = ["plat_nomor","motor_1_nohelmet", "motor_2_nohelmet", "driver_unbuckled"]
            for det in detections:
                label = det["label"]
                counts[label] = counts.get(label, 0) + 1

                if label in pelanggaran_terdeteksi:
                    # Cari objek yang dilacak yang sesuai dengan deteksi ini
                    det_x1, det_y1, det_x2, det_y2 = map(int, det["bbox"])
                    det_center_x = (det_x1 + det_x2) / 2
                    
                    found_speed = 0
                    for obj_id, data in tracked_objects.items():
                        obj_x1, _, obj_x2, _ = data["bbox"]
                        obj_center_x = (obj_x1 + obj_x2) / 2
                        # Asumsikan deteksi adalah objek yang sama jika pusat horizontalnya dekat
                        if abs(det_center_x - obj_center_x) < 50: 
                            found_speed = data.get("speed_kmh", 0)
                            break
                    
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{label}_{timestamp_str}.jpg"
                    snapshot_path_save = os.path.join(snapshot_folder, filename)
                    snapshot_path_db = os.path.join(f"snapshots/cam{feed_number}", filename).replace("\\", "/")
                    
                    # Simpan frame dengan bounding box
                    temp_frame = frame_resized.copy()
                    _draw_boxes(temp_frame, [det], {}) # Gambar hanya box pelanggaran
                    cv2.imwrite(snapshot_path_save, temp_frame)

                    with app.app_context():
                        save_to_db(
                            cam_id=feed_number,
                            plat_number="N/A",
                            vehicle_type="Motor" if "motor" in label else "Mobil",
                            violation=label,
                            vehicle_speed=found_speed,
                            image_path=snapshot_path_db
                        )

            display_counts = counts
            smooth_counts = _update_deque_and_average(feed_number, counts)
            _update_object_count(feed_number, smooth_counts)

        _draw_boxes(frame_resized, detections, tracked_objects)
        
        y_text = 40
        for label, cnt in display_counts.items():
            cv2.putText(frame_resized, f"{label}: {cnt}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_text += 25

        time.sleep(1.0 / fps)
        yield _generate_mjpeg_frame(frame_resized)

    stream.release()

# ================= Decorators =================
def login_required(f):
    """Decorator untuk memastikan user sudah login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ================= Routes =================
@app.route('/')
@login_required
def index():
    return render_template('index.html', video_paths=video_paths)

@app.route('/object_count')
def get_object_count():
    return jsonify(object_counts=object_counts)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id not in video_paths:
        return "CCTV tidak tersedia", 404
    return Response(generate_stream(video_paths[cam_id], cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detail/<int:cam_id>')
@login_required
def detail(cam_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path "
                "FROM pelanggaran_kendaraan WHERE cam_id=%s ORDER BY timestamp DESC", (cam_id,))
    logs = cur.fetchall()
    cur.close()
    return render_template('detail.html', cam_id=cam_id, logs=logs)

@app.route('/detail_feed/<int:cam_id>')
def detail_feed(cam_id):
    if cam_id not in video_paths:
        return "CCTV tidak tersedia", 404
    return Response(generate_stream(video_paths[cam_id], cam_id, detail=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/arsip')
@login_required
def arsip():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path "
                "FROM pelanggaran_kendaraan ORDER BY timestamp DESC")
    logs = cur.fetchall()
    cur.close()
    return render_template('arsip.html', logs=logs)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            flash('Username sudah digunakan.', 'danger')
            cur.close()
            return redirect(url_for('register'))

        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        mysql.connection.commit()
        cur.close()
        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session['admin_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('index'))
        flash("Username atau password salah.", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.", "success")
    return redirect(url_for('login'))

# ================= Main =================
if __name__ == '__main__':
    # Pastikan Anda sudah menginstal scipy: pip install scipy
    app.run(host='0.0.0.0', port=5001, debug=True)
