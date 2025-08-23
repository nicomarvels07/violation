from flask import Flask, render_template, jsonify, Response, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from datetime import datetime
from functools import wraps
import cv2
import time
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # wajib untuk session

# ---------------- MySQL Config ----------------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'cctv_monitoring'
mysql = MySQL(app)

# ---------------- Fungsi Simpan Pelanggaran ----------------
def simpan_ke_db(cam_id, plat_number, vehicle_type, violation, vehicle_speed, image_path=""):
    """Simpan data pelanggaran kendaraan ke database"""
    try:
        cur = mysql.connection.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = """INSERT INTO pelanggaran_kendaraan
                 (cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path)
                 VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        val = (cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path)
        cur.execute(sql, val)
        mysql.connection.commit()
        cur.close()
        print(f"Pelanggaran tersimpan: {plat_number}, {violation}")
    except Exception as e:
        print("Error menyimpan ke database:", e)

# ---------------- Fungsi Ambil Data ----------------
def get_vehicle_data(cam_id=None):
    cur = mysql.connection.cursor()
    if cam_id:
        cur.execute("SELECT id, cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path "
                    "FROM pelanggaran_kendaraan WHERE cam_id=%s ORDER BY timestamp DESC", (cam_id,))
    else:
        cur.execute("SELECT id, cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path "
                    "FROM pelanggaran_kendaraan ORDER BY timestamp DESC")
    logs = cur.fetchall()
    cur.close()
    return logs

# ---------------- VIDEO STREAM ----------------
video_paths = {
    1: "assets/cctv1.mp4",
    2: "assets/cctv2.mp4",
    3: "assets/cctv3.mp4",
    4: "assets/cctv4.mp4"
}

def generate_frames(cam_id):
    """Stream video MJPEG per CCTV dan simpan pelanggaran simulasi ke DB"""
    if cam_id not in video_paths:
        return

    cap = cv2.VideoCapture(video_paths[cam_id])

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Simulasi penyimpanan pelanggaran setiap 200 frame
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 200 == 0:
            simpan_ke_db(cam_id, "B1234XYZ", "Mobil", "Parkir Liar", 60, "snapshots/dummy.jpg")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        time.sleep(1.0 / fps)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------- ROUTES ----------------
@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    return Response(generate_frames(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/vehicle_data/<int:cam_id>")
def vehicle_data(cam_id):
    logs = get_vehicle_data(cam_id)
    return jsonify([{
        "id": log[0],
        "cam_id": log[1],
        "timestamp": log[2],
        "plat_number": log[3],
        "vehicle_type": log[4],
        "violation": log[5],
        "vehicle_speed": log[6],
        "image_path": log[7]
    } for log in logs])

@app.route("/detail/<int:cam_id>")
def detail(cam_id):
    logs = get_vehicle_data(cam_id)
    return render_template("detail.html", cam_id=cam_id, logs=logs)

@app.route('/history')
def history():
    logs = get_vehicle_data()
    return render_template('detail.html', logs=logs)

# ---------------- MAIN ----------------
if __name__ == '__main__':
    app.run(debug=True)
